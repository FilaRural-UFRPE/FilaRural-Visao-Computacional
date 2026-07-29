import sys
import os
import shutil
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from yolo import YoloONNX

app = FastAPI(
    title="FilaRural — Visão Computacional (ONNX)",
    description="API para análise de fila do RU usando YOLOv8 ONNX Runtime",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://semdesperdicio.smartru.com.br",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.onnx")
yolo = YoloONNX(model_path=MODEL_PATH)

POSTGRES_HOST     = os.environ.get("SUPABASE_HOST", "")
POSTGRES_PORT     = int(os.environ.get("SUPABASE_PORT", 5432))
POSTGRES_DB       = os.environ.get("SUPABASE_DB", "postgres")
POSTGRES_USER     = os.environ.get("SUPABASE_USER", "")
POSTGRES_PASSWORD = os.environ.get("SUPABASE_PASSWORD", "")


def get_connection():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor,
    )


def ensure_table():
    """Cria a tabela de status da fila se não existir."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS queue_status (
                        id SERIAL PRIMARY KEY,
                        people_in_line INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        waiting_time_minutes INTEGER NOT NULL,
                        captured_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            conn.commit()
    except Exception as e:
        print(f"Aviso: não foi possível criar tabela queue_status: {e}")


def save_queue_status(people: int, status: str, waiting_time: int) -> bool:
    """Guarda a leitura mais recente da fila no banco."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO queue_status (people_in_line, status, waiting_time_minutes)
                       VALUES (%s, %s, %s)""",
                    (people, status, waiting_time),
                )
            conn.commit()
        return True
    except Exception as e:
        print(f"Aviso: não foi possível guardar status da fila: {e}")
        return False


def classify_queue(people: int) -> tuple:
    if people == 0:
        return "vazia", 0
    elif people <= 5:
        return "pequena", people * 2
    elif people <= 15:
        return "média", people * 2
    else:
        return "grande", people * 2


@app.on_event("startup")
async def startup():
    ensure_table()


@app.get("/")
def root():
    return {
        "service": "FilaRural Visão Computacional",
        "runtime": "ONNX",
        "endpoints": {
            "GET /health": "Verifica se o serviço está online",
            "POST /analyze": "Envia imagem para análise da fila (usado pela câmera)",
            "GET /queue/status": "Retorna o estado mais recente da fila (usado pelos estudantes)",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "FilaRural Visão Computacional", "runtime": "ONNX"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Recebe uma imagem da fila, analisa com YOLOv8 e guarda o resultado no banco.
    Chamado pelo script capture_and_analyze.py rodando junto à câmera.
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

        db_saved = save_queue_status(people, status, waiting_time)

        return {
            "people_in_line":       people,
            "waiting_time_minutes": waiting_time,
            "status":               status,
            "message":              f"Fila {status} — {people} pessoas — ~{waiting_time} min de espera",
            "db_saved":             db_saved,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/queue/status")
def queue_status():
    """
    Retorna o estado mais recente da fila.
    Usado pelo frontend para mostrar aos estudantes antes de irem ao RU.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT people_in_line, status, waiting_time_minutes, captured_at
                    FROM queue_status
                    ORDER BY captured_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()

        if not row:
            return {
                "available": False,
                "message": "Ainda não há dados da fila disponíveis.",
            }

        # Verifica se a leitura é recente (menos de 20 minutos)
        age_minutes = (datetime.utcnow() - row["captured_at"].replace(tzinfo=None)).total_seconds() / 60
        is_stale = age_minutes > 20

        return {
            "available":            True,
            "people_in_line":       row["people_in_line"],
            "status":               row["status"],
            "waiting_time_minutes": row["waiting_time_minutes"],
            "captured_at":          row["captured_at"].isoformat(),
            "is_stale":             is_stale,
            "age_minutes":          round(age_minutes, 1),
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Erro ao consultar estado da fila: {str(e)}",
        }


@app.get("/queue/history")
def queue_history(hours: int = 3):
    """Retorna o histórico da fila nas últimas N horas — útil para gráficos."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT people_in_line, status, waiting_time_minutes, captured_at
                    FROM queue_status
                    WHERE captured_at > NOW() - INTERVAL '%s hours'
                    ORDER BY captured_at ASC
                """, (hours,))
                rows = cur.fetchall()

        return {
            "available": len(rows) > 0,
            "history": [
                {
                    "people_in_line": r["people_in_line"],
                    "status": r["status"],
                    "waiting_time_minutes": r["waiting_time_minutes"],
                    "captured_at": r["captured_at"].isoformat(),
                }
                for r in rows
            ],
        }
    except Exception as e:
        return {"available": False, "message": str(e)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        yolo.read(sys.argv[1])
        people = yolo.get_num_of_people()
        status, waiting_time = classify_queue(people)
        print(f"Pessoas: {people} | Status: {status} | Espera: {waiting_time} min")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
