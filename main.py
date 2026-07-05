import logging
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from yolo import Yolo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Estado global simples: o modelo YOLO é carregado uma única vez, na
# inicialização da aplicação, e reutilizado em todas as requisições.
# Recarregá-lo a cada chamada (como acontecia antes) é lento e caro em memória.
yolo_model: Yolo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model
    logger.info("Carregando modelo YOLOv8...")
    yolo_model = Yolo()
    logger.info("Modelo carregado com sucesso.")
    yield
    yolo_model = None


app = FastAPI(
    title="FilaRural — Visão Computacional",
    description="API para análise de fila do RU usando YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)


def _classify_queue(people_in_line: int) -> str:
    if people_in_line == 0:
        return "vazia"
    if people_in_line <= 5:
        return "pequena"
    if people_in_line <= 15:
        return "média"
    return "grande"


def analyze_image(filepath: str) -> tuple | None:
    """
    Executa a detecção de pessoas na imagem e monta o resultado da análise.

    Retorna uma tupla (people_in_line, waiting_time, status) ou None em caso de erro.
    Reutiliza a instância global do modelo YOLO em vez de recriá-la a cada chamada.
    """
    try:
        if yolo_model is None:
            raise RuntimeError("Modelo YOLO ainda não foi inicializado")

        if yolo_model.read(filepath) != 0:
            return None

        yolo_model.save()
        people_in_line = yolo_model.get_num_of_people()

        # Estimativa de tempo de espera (2 minutos por pessoa)
        waiting_time = people_in_line * 2
        status = _classify_queue(people_in_line)

        return (people_in_line, waiting_time, status)
    except Exception:
        logger.exception("Erro ao analisar imagem")
        return None


@app.get("/health")
def health():
    return {"status": "GOOD", "service": "FilaRural Visão Computacional"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Recebe uma imagem da fila e retorna:
    - Número de pessoas
    - Tempo de espera estimado (minutos)
    - Classificação da fila (vazia, pequena, média, grande)
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. "
            f"Tipos aceitos: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # Nome de arquivo temporário gerado pelo servidor (nunca confiar no nome
    # enviado pelo cliente: evita path traversal e colisões entre requisições
    # concorrentes). A extensão original é preservada apenas para o OpenCV
    # conseguir inferir o formato corretamente.
    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        original_ext = ".jpg"
    temp_path = os.path.join(tempfile.gettempdir(), f"filarural_{uuid.uuid4().hex}{original_ext}")
    output_path = os.path.join(os.path.dirname(temp_path), f"MODIFIED_{os.path.basename(temp_path)}")

    try:
        contents = await file.read(MAX_FILE_SIZE_BYTES + 1)
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Arquivo excede o tamanho máximo permitido (10MB)")
        if not contents:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        with open(temp_path, "wb") as f:
            f.write(contents)

        # A inferência do YOLO é CPU-bound e bloqueante; roda em threadpool
        # para não travar o event loop do FastAPI.
        result = await run_in_threadpool(analyze_image, temp_path)

        if result is None:
            return JSONResponse(status_code=500, content={"error": "Erro ao processar imagem"})

        people, waiting_time, status = result
        return {
            "people_in_line": people,
            "waiting_time_minutes": waiting_time,
            "status": status,
            "message": f"Fila {status} — {people} pessoas — ~{waiting_time} minutos de espera",
        }
    finally:
        # Remove os arquivos temporários (original e anotado), sucesso ou erro.
        for path in (temp_path, output_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Não foi possível remover arquivo temporário: %s", path)


@app.get("/")
def root():
    return {
        "service": "FilaRural Visão Computacional",
        "endpoints": {
            "GET /health": "Verifica se o serviço está online",
            "POST /analyze": "Envia imagem para análise da fila",
        },
    }


# Mantém compatibilidade com execução via linha de comando
if __name__ == "__main__":
    if len(sys.argv) > 1:
        yolo_model = Yolo()
        result = analyze_image(sys.argv[1])
        print("Retorno da função analyze_image:", result)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
