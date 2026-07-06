import sys
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from yolo import YoloONNX

app = FastAPI(
    title="FilaRural — Visão Computacional (ONNX)",
    description="API para análise de fila do RU usando YOLOv8 ONNX Runtime",
    version="1.0.0",
)

# Carrega o modelo uma vez ao arrancar
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.onnx")
yolo = YoloONNX(model_path=MODEL_PATH)


def classify_queue(people: int) -> tuple:
    """Classifica a fila e estima tempo de espera."""
    if people == 0:
        return "vazia", 0
    elif people <= 5:
        return "pequena", people * 2
    elif people <= 15:
        return "média", people * 2
    else:
        return "grande", people * 2


@app.get("/")
def root():
    return {
        "service": "FilaRural Visão Computacional",
        "runtime": "ONNX",
        "endpoints": {
            "GET /health": "Verifica se o serviço está online",
            "POST /analyze": "Envia imagem para análise da fila",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "FilaRural Visão Computacional", "runtime": "ONNX"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Recebe uma imagem da fila e retorna:
    - Número de pessoas
    - Tempo de espera estimado (minutos)
    - Classificação da fila (vazia, pequena, média, grande)
    """
    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        original_ext = ".jpg"
    temp_path = f"temp_{uuid.uuid4().hex}{original_ext}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = yolo.read(temp_path)
        if result != 0:
            return JSONResponse(status_code=500, content={"error": "Erro ao processar imagem"})

        people = yolo.get_num_of_people()
        status, waiting_time = classify_queue(people)

        return {
            "people_in_line":       people,
            "waiting_time_minutes": waiting_time,
            "status":               status,
            "message":              f"Fila {status} — {people} pessoas — ~{waiting_time} min de espera",
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        yolo.read(sys.argv[1])
        people = yolo.get_num_of_people()
        status, waiting_time = classify_queue(people)
        print(f"Pessoas: {people} | Status: {status} | Espera: {waiting_time} min")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
