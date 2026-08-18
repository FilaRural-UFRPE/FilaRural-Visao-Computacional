"""
Captura periódica da câmera Intelbras Mibo (via RTSP) e envio para a API
FilaRural de análise de fila.

Este script precisa rodar em uma máquina conectada à MESMA REDE LOCAL da
câmera (a Intelbras recomenda RTSP apenas em rede local, não pela internet).
Pode ser um PC, notebook ou Raspberry Pi ligado 24/7 no RU.

Como usar:
    pip install opencv-python-headless requests
    export RTSP_PASSWORD="sua_senha_aqui"
    export RTSP_IP="192.168.1.50"
    python capture_and_analyze.py
"""
import os
import time
import logging
from datetime import datetime
import cv2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO --------------------------------------------------------
# Credenciais lidas de variáveis de ambiente — nunca hardcoded, já que este
# repositório é público no GitHub. Configure via .env local (fora do
# controle de versão) ou diretamente no ambiente do serviço/systemd que
# roda este script 24/7.
RTSP_USER = os.environ.get("RTSP_USER", "admin")
RTSP_PASSWORD = os.environ.get("RTSP_PASSWORD", "")
RTSP_IP = os.environ.get("RTSP_IP", "")
RTSP_PORT = int(os.environ.get("RTSP_PORT", 554))

if not RTSP_PASSWORD or not RTSP_IP:
    logger.error(
        "RTSP_PASSWORD e RTSP_IP precisam estar definidos como variáveis de "
        "ambiente antes de rodar este script."
    )
    raise SystemExit(1)

RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/cam/realmonitor?channel=1&subtype=0"

API_URL = os.environ.get(
    "API_URL", "https://filarural-visao-computacional-1.onrender.com/analyze"
)
CAPTURE_INTERVAL_SECONDS = int(os.environ.get("CAPTURE_INTERVAL_SECONDS", 5 * 60))  # a cada 5 minutos
# --------------------------------------------------------------------------


def capture_frame(rtsp_url: str) -> "cv2.typing.MatLike | None":
    """Conecta na câmera, captura um único frame e fecha a conexão."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            logger.error("Não foi possível conectar à câmera RTSP.")
            return None

        # Descarta frames iniciais potencialmente antigos. Uma falha transitória
        # não deve abortar imediatamente: retorna o último frame válido somente
        # depois de conseguir leituras consecutivas do stream atual.
        frame = None
        valid_frames = 0
        for _ in range(10):
            ret, candidate = cap.read()
            if not ret or candidate is None:
                valid_frames = 0
                continue
            frame = candidate
            valid_frames += 1
            if valid_frames >= 5:
                break

        if valid_frames < 5:
            logger.error("Stream RTSP não forneceu frames válidos suficientes.")
            return None
        return frame
    finally:
        cap.release()


def send_to_api(frame, api_url: str) -> dict | None:
    """Envia o frame capturado para a API de análise e retorna o resultado.

    Observação: a rotação do frame (a câmera entrega invertido, 180°) é
    corrigida do lado do servidor, dentro de yolo.py — não é preciso rotacionar
    aqui antes de enviar.
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        logger.error("Falha ao codificar o frame como JPEG.")
        return None

    try:
        files = {"file": ("captura.jpg", buffer.tobytes(), "image/jpeg")}
        response = requests.post(api_url, files=files, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            logger.error("API retornou JSON em formato inesperado.")
            return None
        return payload
    except (requests.exceptions.RequestException, ValueError):
        logger.exception("Erro ao chamar a API de análise")
        return None


def run_once():
    logger.info("Capturando frame da câmera...")
    frame = capture_frame(RTSP_URL)
    if frame is None:
        logger.error("Captura falhou — pulando este ciclo.")
        return

    logger.info("Frame capturado, enviando para a API...")
    result = send_to_api(frame, API_URL)
    if result is None:
        logger.error("Análise falhou — pulando este ciclo.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        "[%s] %s | %s pessoas | ~%s min de espera | salvo no banco: %s",
        timestamp,
        result.get("status"),
        result.get("people_in_line"),
        result.get("waiting_time_minutes"),
        result.get("db_saved"),
    )
    # O resultado já é persistido pela própria API (/analyze -> queue_status
    # no Postgres), que é de onde o frontend/dashboard lê via /queue/status.


def main():
    logger.info("Iniciando captura periódica a cada %s segundos...", CAPTURE_INTERVAL_SECONDS)
    while True:
        run_once()
        time.sleep(CAPTURE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
