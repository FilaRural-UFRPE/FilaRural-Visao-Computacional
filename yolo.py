import cv2
from ultralytics import YOLO as YOLO_MODEL

class Yolo:
    def __init__(self):
        self.model = YOLO_MODEL("yolov8s.pt") # Modelo

        self.filepath = None
        self.image = None
        self.results = None

    def read(self, filepath) -> int:
        """
        Esta função modifica as variáveis da classe, atualizando seus valores conforme
        o processamento da imagem passada como argumento.

        Argumentos:
            filepath: localização da imagem

        Retorno:
            0 em caso de sucesso,
            1 em caso de erro
        """

        try:
            self.filepath = filepath
            self.image = cv2.imread(filepath)
            self.results = self.model(self.image, classes=0) # "Classes" igual a 0 significa detecção de apenas pessoas

            return 0
        except:
            return 1

    def save(self) -> int:
        """
        Esta função salva a foto processada pela função self.read, com as caixas de identificação de pessoas.
        O local de salvamento do arquivo é o mesmo onde está a foto original, mas com o prefixo "MODIFIED_".

        Retorno:
            0 em caso de sucesso,
            1 em caso de erro
        """
                
        try:
            for result in self.results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 80), 2)

            output_path = f"MODIFIED_{self.filepath}"
            cv2.imwrite(output_path, self.image)
            print(f"Imagem salva em {output_path}")
        except:
            return 1
        
    def get_num_of_people(self) -> int:
        """
        Retorna o número de pessoas encontradas na imagem.
        """

        try:
            count = 0
            for result in self.results:
                boxes = result.boxes
                
                for box in boxes:
                    count += 1

            return count
        except:
            return 0

