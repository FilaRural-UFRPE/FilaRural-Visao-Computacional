import cv2
import numpy as np
import onnxruntime as ort

class YoloONNX:
    """
    Detector de pessoas usando YOLOv8 exportado para ONNX.
    Muito mais leve que o Ultralytics — ideal para servidores com pouca RAM.
    """

    # Parâmetros do modelo YOLOv8
    INPUT_WIDTH  = 640
    INPUT_HEIGHT = 640
    CONF_THRESHOLD = 0.4  # confiança mínima para detetar pessoa
    NMS_THRESHOLD  = 0.5  # threshold para Non-Maximum Suppression
    PERSON_CLASS   = 0    # classe 0 = pessoa no COCO dataset

    def __init__(self, model_path: str = "yolov8s.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.filepath    = None
        self.image       = None
        self.detections  = []  # lista de (x1, y1, x2, y2, confidence)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Redimensiona e normaliza a imagem para o formato do modelo."""
        h, w = image.shape[:2]
        scale = min(self.INPUT_WIDTH / w, self.INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Padding para 640x640
        canvas = np.full((self.INPUT_HEIGHT, self.INPUT_WIDTH, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Normaliza para [0, 1] e converte para CHW
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)
        return blob, scale, (w, h)

    def _postprocess(self, outputs: np.ndarray, scale: float, orig_size: tuple) -> list:
        """Processa os outputs do modelo e aplica NMS."""
        orig_w, orig_h = orig_size
        predictions = outputs[0][0].T  # (8400, 84)

        boxes, scores = [], []
        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if class_id != self.PERSON_CLASS or confidence < self.CONF_THRESHOLD:
                continue

            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) / scale)
            y1 = int((cy - bh / 2) / scale)
            x2 = int((cx + bw / 2) / scale)
            y2 = int((cy + bh / 2) / scale)

            # Garante que as coordenadas estão dentro da imagem
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)

        if not boxes:
            return []

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.CONF_THRESHOLD, self.NMS_THRESHOLD)
        detections = []
        for i in indices:
            x, y, w, h = boxes[i]
            detections.append((x, y, x + w, y + h, scores[i]))

        return detections

    def read(self, filepath: str) -> int:
        """
        Processa a imagem e preenche self.detections.
        Retorna 0 em sucesso, 1 em erro.
        """
        try:
            self.filepath = filepath
            self.image    = cv2.imread(filepath)
            if self.image is None:
                return 1

            blob, scale, orig_size = self._preprocess(self.image)
            outputs = self.session.run(None, {self.input_name: blob})
            self.detections = self._postprocess(outputs, scale, orig_size)
            return 0
        except Exception as e:
            print(f"Erro em read(): {e}")
            return 1

    def save(self) -> int:
        """
        Salva a imagem com as caixas desenhadas.
        Retorna 0 em sucesso, 1 em erro.
        """
        try:
            image_copy = self.image.copy()
            for (x1, y1, x2, y2, conf) in self.detections:
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 80), 2)
                cv2.putText(image_copy, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 80), 1)

            output_path = f"MODIFIED_{self.filepath}"
            cv2.imwrite(output_path, image_copy)
            print(f"Imagem salva em {output_path}")
            return 0
        except Exception as e:
            print(f"Erro em save(): {e}")
            return 1

    def get_num_of_people(self) -> int:
        """Retorna o número de pessoas detectadas."""
        return len(self.detections)
