#!/usr/bin/env python3
"""
Simple executable for FilaRural YOLO object detection.
Rodar na máquina conectada à câmera Intelbras.
"""
import sys
import os

# Set default thresholds if not already set
if 'YOLO_CONF_THRESHOLD' not in os.environ:
    os.environ['YOLO_CONF_THRESHOLD'] = '0.04'
if 'CAMERA_ROTATE_DEGREES' not in os.environ:
    os.environ['CAMERA_ROTATE_DEGREES'] = '180'

from yolo import YoloONNX

def main():
    if len(sys.argv) < 2:
        print('Uso: python yolo_executable.py <caminho_da_imagem>')
        print('   ou: python yolo_executable.py --capture para captura RTSP')
        sys.exit(1)

    command = sys.argv[1]

    if command == '--capture':
        # Modo captura RTSP - captura única
        print('Modo captura RTSP - nao implementado diretamente aqui')
        print('Use capture_and_analyze.py para captura periódica')
    else:
        filepath = command
        try:
            detector = YoloONNX()
            result = detector.read(filepath)
            if result == 0:
                people = detector.get_num_of_people()
                print(f'Pessoas detectadas: {people}')
                for (x1, y1, x2, y2, conf) in detector.detections:
                    print(f'  - Conf: {conf:.2f} | Box: ({x1}, {y1}) - ({x2}, {y2})')
            else:
                print('Erro ao processar imagem')
        except Exception as e:
            print(f'Erro: {e}')

if __name__ == '__main__':
    main()