"""
Captura periódica da câmera Intelbras Mibo (via RTSP) e envio para a API
FilaRural de análise de fila.

Este script precisa rodar em uma máquina conectada à MESMA REDE LOCAL da
câmera (a Intelbras recomenda RTSP apenas em rede local, não pela internet).
Pode ser um PC, notebook ou Raspberry Pi ligado 24/7 no RU.

Como usar:
    pip install opencv-python-headless requests
    python capture_and_analyze.py
"""

import time
import logging
from datetime import datetime

import cv2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO --------------------------------------------------------

# Monte a URL a partir dos dados obtidos no app Mibo Smart (ver Configurações
# avançadas > Redes > Informações de rede, e a chave de acesso da câmera).
RTSP_USER = "admin"
RTSP_PASSWORD = "SUA_SENHA_AQUI"
RTSP_IP = "SEU_IP_AQUI"          # ex: 192.168.1.50
RTSP_PORT = 554                   # porta RTSP anotada no app
RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/cam/realmonitor?channel=1&subtype=0"

API_URL = "https://filarural-visao-computacional-1.onrender.com/analyze"

CAPTURE_INTERVAL_SECONDS = 5 * 60  # captura a cada 5 minutos — ajuste conforme necessário

# --------------------------------------------------------------------------


def capture_frame(rtsp_url: str) -> "cv2.typing.MatLike | None":
    """Conecta na câmera, captura um único frame e fecha a conexão."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Não foi possível conectar à câmera RTSP.")
        return None

    # Descarta alguns frames iniciais — o primeiro frame de um stream RTSP
    # às vezes vem corrompido ou desatualizado (buffer antigo).
    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            frame = None
            break

    cap.release()
    return frame


def send_to_api(frame, api_url: str) -> dict | None:
    """Envia o frame capturado para a API de análise e retorna o resultado."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        logger.error("Falha ao codificar o frame como JPEG.")
        return None

    try:
        files = {"file": ("captura.jpg", buffer.tobytes(), "image/jpeg")}
        response = requests.post(api_url, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
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
        "[%s] %s | %s pessoas | ~%s min de espera",
        timestamp,
        result.get("status"),
        result.get("people_in_line"),
        result.get("waiting_time_minutes"),
    )
    # TODO: aqui é onde o resultado deve ser salvo/enviado para onde o
    # frontend/dashboard vai ler (ex: banco de dados, planilha, outro
    # endpoint da própria aplicação FilaRural).


def main():
    logger.info("Iniciando captura periódica a cada %s segundos...", CAPTURE_INTERVAL_SECONDS)
    while True:
        run_once()
        time.sleep(CAPTURE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
