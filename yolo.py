import os
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

    # Confirmado por testes reais com a câmera do RU: 0.1 recupera bem mais
    # gente distante/pequena sem gerar falsos positivos em excesso (testado
    # numa cena vazia: só 2 detecções, consideradas aceitáveis).
    CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.1"))
    NMS_THRESHOLD  = 0.5  # threshold para Non-Maximum Suppression
    PERSON_CLASS   = 0    # classe 0 = pessoa no COCO dataset

    # Confirmado por testes: a câmera Intelbras já corrige a orientação
    # internamente (flip de firmware) mesmo estando fisicamente instalada de
    # cabeça para baixo — o RTSP já chega certo. Rotação extra aqui só
    # atrapalha. Mantido configurável caso a câmera/config mude no futuro.
    ROTATE_DEGREES = int(os.environ.get("CAMERA_ROTATE_DEGREES", "0"))

    # --- Região de interesse (ROI) ------------------------------------ #
    # A câmera tem lente fisheye/grande angular montada bem perto do teto:
    # boa parte do frame é teto, árvores e prédio, e a calçada onde a fila
    # forma fica numa faixa relativamente pequena da imagem. Recortando essa
    # faixa ANTES de redimensionar para 640x640, as pessoas (inclusive as
    # mais distantes) ocupam proporcionalmente mais pixels na entrada do
    # modelo — sem precisar de um modelo maior ou mais RAM.
    #
    # Valores em porcentagem (0.0-1.0) da imagem original, calibrados a
    # partir dos frames reais de teste. Ajustar aqui se a câmera for
    # reposicionada ou o enquadramento mudar.
    ROI_TOP    = float(os.environ.get("ROI_TOP_PCT", "0.45"))
    ROI_BOTTOM = float(os.environ.get("ROI_BOTTOM_PCT", "1.0"))
    ROI_LEFT   = float(os.environ.get("ROI_LEFT_PCT", "0.0"))
    ROI_RIGHT  = float(os.environ.get("ROI_RIGHT_PCT", "1.0"))

    # Liga/desliga os prints de debug (confidências brutas) nos logs.
    # Deixado desligado por padrão para não poluir os logs em produção.
    DEBUG_DETECTIONS = os.environ.get("DEBUG_DETECTIONS", "false").lower() == "true"

    def __init__(self, model_path: str = "yolov8n.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.filepath    = None
        self.image       = None      # imagem original completa (sem crop)
        self.roi_offset  = (0, 0)    # (x, y) do canto superior esquerdo do ROI, para converter as caixas de volta
        self.detections  = []        # lista de (x1, y1, x2, y2, confidence) — já em coordenadas da imagem ORIGINAL

    def _crop_roi(self, image: np.ndarray):
        """Recorta a região de interesse configurada, e devolve também o
        offset (x, y) do recorte, para converter as caixas de volta para as
        coordenadas da imagem original depois."""
        h, w = image.shape[:2]
        top    = int(h * self.ROI_TOP)
        bottom = int(h * self.ROI_BOTTOM)
        left   = int(w * self.ROI_LEFT)
        right  = int(w * self.ROI_RIGHT)
        cropped = image[top:bottom, left:right]
        return cropped, (left, top)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Converte BGR para RGB, redimensiona e normaliza para o modelo."""
        h, w = image.shape[:2]
        scale = min(self.INPUT_WIDTH / w, self.INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # cv2.imread devolve BGR, enquanto os modelos YOLO da Ultralytics são
        # treinados/exportados esperando RGB. Manter BGR reduz sensivelmente a
        # confiança e pode transformar uma fila cheia em zero detecções.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (new_w, new_h))

        # Padding para 640x640
        canvas = np.full((self.INPUT_HEIGHT, self.INPUT_WIDTH, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Normaliza para [0, 1] e converte para CHW
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 640, 640)
        return blob, scale, (w, h)

    def _postprocess(self, outputs: np.ndarray, scale: float, roi_size: tuple) -> list:
        """Processa os outputs do modelo e aplica NMS. Retorna caixas em
        coordenadas RELATIVAS AO ROI recortado (o offset é aplicado depois,
        em read())."""
        roi_w, roi_h = roi_size
        predictions = outputs[0][0].T  # (8400, 84)

        if self.DEBUG_DETECTIONS:
            person_confidences = predictions[:, 4 + self.PERSON_CLASS]
            top5 = np.sort(person_confidences)[::-1][:5]
            print(f"[DEBUG] Top 5 confidências de 'person' nesta imagem: {top5.tolist()}", flush=True)
            print(f"[DEBUG] CONF_THRESHOLD atual: {self.CONF_THRESHOLD}", flush=True)
            print(f"[DEBUG] ROTATE_DEGREES atual: {self.ROTATE_DEGREES}", flush=True)
            print(f"[DEBUG] ROI: top={self.ROI_TOP} bottom={self.ROI_BOTTOM} left={self.ROI_LEFT} right={self.ROI_RIGHT}", flush=True)

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

            # Garante que as coordenadas estão dentro do ROI
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(roi_w, x2), min(roi_h, y2)

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
            image = cv2.imread(filepath)
            if image is None:
                return 1

            # Corrige a rotação da câmera, se configurado (default: sem rotação)
            if self.ROTATE_DEGREES == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif self.ROTATE_DEGREES == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif self.ROTATE_DEGREES == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.image = image

            # Recorta a região de interesse antes de mandar pro modelo
            cropped, offset = self._crop_roi(image)
            self.roi_offset = offset

            blob, scale, roi_size = self._preprocess(cropped)
            outputs = self.session.run(None, {self.input_name: blob})
            roi_detections = self._postprocess(outputs, scale, roi_size)

            # O ROI melhora pessoas pequenas na área habitual da fila, porém a
            # câmera pode mudar de posição ou a fila pode ocupar outra parte do
            # quadro. Um resultado vazio dispara uma segunda inferência no
            # frame completo para evitar publicar um falso zero.
            if not roi_detections and cropped.shape[:2] != image.shape[:2]:
                blob, scale, full_size = self._preprocess(image)
                outputs = self.session.run(None, {self.input_name: blob})
                self.detections = self._postprocess(outputs, scale, full_size)
                self.roi_offset = (0, 0)
                return 0

            # Converte as caixas de volta para coordenadas da imagem ORIGINAL,
            # somando o offset do recorte — importante para o save() desenhar
            # certo e para qualquer consumidor que espere coordenadas completas.
            off_x, off_y = self.roi_offset
            self.detections = [
                (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y, conf)
                for (x1, y1, x2, y2, conf) in roi_detections
            ]
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
