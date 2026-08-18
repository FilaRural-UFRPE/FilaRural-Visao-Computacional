"""
Validação do pipeline de visão computacional em condições de produção.

Replica exatamente o que acontece em produção quando a câmera captura um
frame e a API analisa:

    1. Captura um frame real da câmera (mesma função de capture_and_analyze.py);
    2. Roda o mesmo YoloONNX usado pela API (mesmos thresholds, ROI e rotação);
    3. Classifica a fila com a mesma lógica de main.py;
    4. Mostra cada detecção com a confiança e o desenho das caixas;
    5. (Opcional) envia o frame para a API real e confere a resposta no
       /queue/status — a cadeia completa até o que o site exibe.

Como usar (na mesma rede da câmera):
    export RTSP_PASSWORD="sua_senha_aqui"
    export RTSP_IP="192.168.1.50"
    python validate_pipeline.py                 # só valida a visão
    python validate_pipeline.py --api           # valida a cadeia inteira
    python validate_pipeline.py --frame foto.jpg  # usa um arquivo, sem câmera

Saídas:
    - prints no terminal com pessoas/status/espera e detecções detalhadas;
    - validated_frame.jpg (com caixas e ROI desenhados) para conferência visual.
"""
import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from yolo import YoloONNX

API_URL = os.environ.get(
    "API_URL", "https://filarural-visao-computacional-1.onrender.com/analyze"
)
QUEUE_STATUS_URL = API_URL.rsplit("/", 1)[0] + "/queue/status"
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.onnx")


def classify_queue(people: int) -> tuple:
    if people == 0:
        return "vazia", 0
    elif people <= 5:
        return "pequena", people * 2
    elif people <= 15:
        return "média", people * 2
    else:
        return "grande", people * 2


def build_rtsp_url() -> str:
    user = os.environ.get("RTSP_USER", "admin")
    password = os.environ.get("RTSP_PASSWORD", "")
    ip = os.environ.get("RTSP_IP", "")
    port = int(os.environ.get("RTSP_PORT", 554))
    return f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0"


def draw_validation_image(image: np.ndarray, detections, roi_box) -> str:
    """Desenha o ROI e as caixas de detecção para conferência visual."""
    annotated = image.copy()

    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, "ROI", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for i, (bx1, by1, bx2, by2, conf) in enumerate(detections, 1):
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 0, 80), 2)
        cv2.putText(annotated, f"#{i} {conf:.2f}", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 80), 1)

    output_path = "validated_frame.jpg"
    cv2.imwrite(output_path, annotated)
    return output_path


def print_detections(detections, roi_box, original_shape):
    h, w = original_shape[:2]
    roi_x, roi_y = roi_box[0], roi_box[1]
    roi_w, roi_h = roi_box[2] - roi_box[0], roi_box[3] - roi_box[1]

    print(f"  {'#':>2} {'confiança':>9}  {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}  zona")
    for i, (x1, y1, x2, y2, conf) in enumerate(sorted(detections, key=lambda d: -d[4]), 1):
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if roi_x <= center_x < roi_x + roi_w and roi_y <= center_y < roi_y + roi_h:
            zone = "DENTRO do ROI"
        else:
            zone = "FORA do ROI"
        print(f"  {i:>2} {conf:>9.3f}  {x1:>5} {y1:>5} {x2:>5} {y2:>5}  {zone}")

    print(f"  Tamanho do frame original: {w}x{h}")
    print(f"  ROI aplicado: x {roi_x}..{roi_x + roi_w}, y {roi_y}..{roi_y + roi_h} "
          f"({roi_w / w * 100:.0f}% da largura, {roi_h / h * 100:.0f}% da altura)")


def validate_frame(detector: YoloONNX, image: np.ndarray) -> None:
    """Roda o mesmo caminho de yolo.read() e imprime o que a API responderia."""
    detector.image = image

    cropped, offset = detector._crop_roi(image)
    roi_box = (offset[0], offset[1], offset[0] + cropped.shape[1], offset[1] + cropped.shape[0])

    blob, scale, roi_size = detector._preprocess(cropped)
    outputs = detector.session.run(None, {detector.input_name: blob})
    roi_detections = detector._postprocess(outputs, scale, roi_size)

    used_fallback = False
    if not roi_detections and cropped.shape[:2] != image.shape[:2]:
        blob, scale, full_size = detector._preprocess(image)
        outputs = detector.session.run(None, {detector.input_name: blob})
        detector.detections = detector._postprocess(outputs, scale, full_size)
        detector.roi_offset = (0, 0)
        used_fallback = True
        print("  ! ROI retornou zero -> caiu para inferência no frame COMPLETO (fallback)")
    else:
        off_x, off_y = detector.roi_offset
        detector.detections = [
            (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y, conf)
            for (x1, y1, x2, y2, conf) in roi_detections
        ]

    people = len(detector.detections)
    status, waiting = classify_queue(people)

    print(f"\n===== RESULTADO (igual ao que a API responderia) =====")
    print(f"  Pessoas detectadas : {people}")
    print(f"  Status da fila     : {status}")
    print(f"  Espera estimada    : ~{waiting} min")
    print(f"  Usou fallback      : {'sim' if used_fallback else 'não'}")

    if detector.detections:
        print("\n  Detecções:")
        print_detections(detector.detections, roi_box, image.shape)
    else:
        print("\n  Nenhuma pessoa detectada.")

    output_path = draw_validation_image(image, detector.detections, roi_box)
    print(f"\n  Imagem com caixas/ROI salva em: {output_path}")


def validate_via_api(image: np.ndarray) -> None:
    """Envia o frame para a API real e confere a resposta do /queue/status."""
    import requests

    print("\n===== VALIDAÇÃO DA CADEIA COMPLETA (API real) =====")
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        print("  Falha ao codificar frame como JPEG.")
        return

    print(f"  Enviando frame para {API_URL} ...")
    try:
        response = requests.post(
            API_URL,
            files={"file": ("validacao.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            print(f"  ! API retornou JSON inesperado: {payload!r}")
            return
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  ! Falha ao chamar a API: {e}")
        return

    print(f"  Resposta do /analyze: {payload}")
    print(f"  -> people_in_line       : {payload.get('people_in_line')}")
    print(f"  -> status               : {payload.get('status')}")
    print(f"  -> waiting_time_minutes : {payload.get('waiting_time_minutes')}")
    print(f"  -> db_saved             : {payload.get('db_saved')}")

    print(f"\n  Consultando {QUEUE_STATUS_URL} ...")
    try:
        status = requests.get(QUEUE_STATUS_URL, timeout=30)
        status.raise_for_status()
        body = status.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  ! Falha ao consultar /queue/status: {e}")
        return

    print(f"  Resposta do /queue/status: {body}")
    if body.get("available"):
        match = body.get("people_in_line") == payload.get("people_in_line")
        print(f"  -> available             : {body.get('available')}")
        print(f"  -> people_in_line        : {body.get('people_in_line')}")
        print(f"  -> status                : {body.get('status')}")
        print(f"  -> captured_at           : {body.get('captured_at')}")
        print(f"  -> is_stale              : {body.get('is_stale')}")
        print(f"\n  {'✔ Site mostra o mesmo número da visão.' if match else '✘ DISCREPÂNCIA: site difere da análise!'}")
    else:
        print(f"  -> available             : {body.get('available')}")
        print(f"  -> message               : {body.get('message')}")
        if body.get("is_stale"):
            print("  ! Dados antigos corretamente NÃO exibidos como estado atual.")


def main():
    parser = argparse.ArgumentParser(description="Valida o pipeline de visão em produção.")
    parser.add_argument("--frame", help="Usar um arquivo de imagem em vez de capturar da câmera.")
    parser.add_argument("--api", action="store_true", help="Valida também a cadeia até a API real.")
    args = parser.parse_args()

    detector = YoloONNX(model_path=MODEL_PATH)
    print(f"Modelo: {MODEL_PATH}")
    print(f"CONF_THRESHOLD={detector.CONF_THRESHOLD} | NMS={detector.NMS_THRESHOLD} | "
          f"ROTATE={detector.ROTATE_DEGREES}")
    print(f"ROI: top={detector.ROI_TOP} bottom={detector.ROI_BOTTOM} "
          f"left={detector.ROI_LEFT} right={detector.ROI_RIGHT}")

    if args.frame:
        print(f"\nCarregando frame do arquivo: {args.frame}")
        image = cv2.imread(args.frame)
        if image is None:
            print(f"Erro: não foi possível ler {args.frame}")
            sys.exit(1)
    else:
        print("\nCapturando frame da câmera (mesmo caminho de produção)...")
        rtsp_url = build_rtsp_url()
        if not os.environ.get("RTSP_PASSWORD") or not os.environ.get("RTSP_IP"):
            print("Defina RTSP_PASSWORD e RTSP_IP antes de rodar sem --frame.")
            sys.exit(1)
        from capture_and_analyze import capture_frame
        image = capture_frame(rtsp_url)
        if image is None:
            print("Erro: captura falhou.")
            sys.exit(1)
        print(f"Frame capturado: {image.shape[1]}x{image.shape[0]} às {datetime.now().strftime('%H:%M:%S')}")

    validate_frame(detector, image)

    if args.api:
        validate_via_api(image)

    print("\nConferência manual: abra validated_frame.jpg e compare o número de")
    print("caixas com a fila real na câmera. Se faltarem pessoas, ajuste o ROI")


if __name__ == "__main__":
    main()
