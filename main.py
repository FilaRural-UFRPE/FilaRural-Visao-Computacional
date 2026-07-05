import sys
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from yolo import Yolo

app = FastAPI(
    title="FilaRural — Visão Computacional",
    description="API para análise de fila do RU usando YOLOv8",
    version="1.0.0",
)


def read_line(filepath: str) -> tuple:
    """
    Lê a imagem da fila do RU e retorna uma tupla com dados da fila.
    O primeiro elemento é o número de pessoas e o segundo é o tempo de espera estimado.
    Retorna None em caso de erro.
    """
    try:
        yolo = Yolo()
        yolo.read(filepath)
        yolo.save()
        people_in_line = yolo.get_num_of_people()

        # Estimativa de tempo de espera (2 minutos por pessoa)
        waiting_time = people_in_line * 2

        # Classificação da fila
        if people_in_line == 0:
            status = "vazia"
        elif people_in_line <= 5:
            status = "pequena"
        elif people_in_line <= 15:
            status = "média"
        else:
            status = "grande"

        return (people_in_line, waiting_time, status)
    except Exception as e:
        print(f"Erro: {e}")
        return None


@app.get("/health")
def health():
    return {"status": "ok", "service": "FilaRural Visão Computacional"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Recebe uma imagem da fila e retorna:
    - Número de pessoas
    - Tempo de espera estimado (minutos)
    - Classificação da fila (vazia, pequena, média, grande)
    """
    # Salva a imagem temporariamente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = read_line(temp_path)

    # Remove ficheiro temporário
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if result is None:
        return JSONResponse(status_code=500, content={"error": "Erro ao processar imagem"})

    people, waiting_time, status = result

    return {
        "people_in_line": people,
        "waiting_time_minutes": waiting_time,
        "status": status,
        "message": f"Fila {status} — {people} pessoas — ~{waiting_time} minutos de espera",
    }


@app.get("/")
def root():
    return {
        "service": "FilaRural Visão Computacional",
        "endpoints": {
            "GET /health": "Verifica se o serviço está online",
            "POST /analyze": "Envia imagem para análise da fila",
        }
    }


# Mantém compatibilidade com execução via linha de comando
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = read_line(sys.argv[1])
        print("Retorno da função read_line:", result)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
