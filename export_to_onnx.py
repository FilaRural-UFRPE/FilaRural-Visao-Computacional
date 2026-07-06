"""
Script para exportar o modelo YOLOv8n (nano) para ONNX.
Executa UMA VEZ localmente antes de fazer deploy:

    pip install ultralytics
    python export_to_onnx.py

Isso gera o ficheiro yolov8n.onnx que deves adicionar ao repositório
(substituindo o yolov8s.onnx) ou guardar num bucket S3/GCS para o Render
descarregar.

Por que trocar de "s" para "n": o modelo "small" consome bem mais RAM e CPU
durante a inferência que o "nano", o que estava causando OOM/timeout no
plano gratuito do Render (512MB RAM / 0.1 CPU). O nano é a opção
recomendada pela própria Ultralytics para ambientes com poucos recursos.
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("Modelo exportado para yolov8n.onnx!")
