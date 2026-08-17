"""
Script opcional: exporta o YOLOv8s (small) para ONNX e depois quantiza para
INT8, na tentativa de caber no limite de RAM do Render free tier (512MB)
com uma precisão melhor que o nano puro.

SÓ USE ISSO SE, mesmo depois do recorte de ROI em yolo.py, o yolov8n ainda
não estiver detectando bem o suficiente. Teste o ROI primeiro — é de graça
e pode já resolver.

Como usar:
    pip install ultralytics onnx onnxruntime
    python export_quantized.py

Isso gera:
    yolov8s.onnx       — modelo "small" original (referência, ~44MB)
    yolov8s_int8.onnx  — modelo quantizado (bem menor, ideal ~11-15MB)

Depois, no Render:
    MODEL_PATH = yolov8s_int8.onnx

IMPORTANTE: teste local antes do deploy — quantização INT8 pode reduzir
um pouco a precisão em troca do menor uso de memória. Compare os
resultados com o yolov8n atual usando as mesmas imagens de teste da fila.
"""
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

print("1/2 — Exportando YOLOv8s para ONNX...")
model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("yolov8s.onnx gerado.")

print("2/2 — Quantizando para INT8...")
quantize_dynamic(
    model_input="yolov8s.onnx",
    model_output="yolov8s_int8.onnx",
    weight_type=QuantType.QInt8,
)
print("yolov8s_int8.onnx gerado — pronto para testar localmente antes do deploy.")
