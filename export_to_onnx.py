"""
Script para exportar o modelo YOLOv8n (nano) para ONNX.

Executa UMA VEZ localmente antes de fazer deploy:
    pip install ultralytics
    python export_to_onnx.py

Isso gera o ficheiro yolov8n.onnx que deves adicionar ao repositório
ou guardar num bucket S3/GCS para o Render descarregar.

Por que o nano ("n") e não o "small" ou "medium": o "small" consome bem mais
RAM e CPU durante a inferência, o que causava OOM/timeout no plano gratuito
do Render (512MB RAM / 0.1 CPU). O nano é a opção recomendada pela própria
Ultralytics para ambientes com poucos recursos — e, combinado com a correção
de rotação e o threshold de confiança ajustado em yolo.py, já é suficiente
para o cenário real da câmera do RU.
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("Modelo exportado para yolov8n.onnx!")
