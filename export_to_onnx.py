"""
Script para exportar o modelo YOLOv8s para ONNX.
Executa UMA VEZ localmente antes de fazer deploy:

    pip install ultralytics
    python export_to_onnx.py

Isso gera o ficheiro yolov8s.onnx que deves adicionar ao repositório
ou guardar num bucket S3/GCS para o Render descarregar.
"""
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("Modelo exportado para yolov8s.onnx!")
