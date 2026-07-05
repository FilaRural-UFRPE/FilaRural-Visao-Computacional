import logging

import cv2
from ultralytics import YOLO as YOLO_MODEL

logger = logging.getLogger(__name__)


class Yolo:
    def __init__(self, model_path: str = "yolov8s.pt"):
        # O modelo deve ser carregado UMA vez (idealmente na inicialização da API,
        # não a cada requisição) e reutilizado entre chamadas.
        self.model = YOLO_MODEL(model_path)
        self.filepath = None
        self.image = None
        self.results = None

    def read(self, filepath: str) -> int:
        """
        Lê e processa a imagem no caminho informado, atualizando o estado da classe.

        Argumentos:
            filepath: localização da imagem

        Retorno:
            0 em caso de sucesso,
            1 em caso de erro
        """
        try:
            self.filepath = filepath
            self.image = cv2.imread(filepath)

            if self.image is None:
                logger.error("Não foi possível ler a imagem em %s (arquivo inválido ou corrompido)", filepath)
                return 1

            # "classes=0" restringe a detecção apenas à classe "pessoa" do COCO
            self.results = self.model(self.image, classes=0)
            return 0
        except Exception:
            logger.exception("Erro ao processar imagem em %s", filepath)
            return 1

    def save(self) -> int:
        """
        Salva a imagem processada com as caixas de identificação de pessoas.
        O arquivo é salvo no mesmo diretório da imagem original, com o prefixo "MODIFIED_".

        Retorno:
            0 em caso de sucesso,
            1 em caso de erro
        """
        if self.image is None or self.results is None:
            logger.error("save() chamado antes de read() ser executado com sucesso")
            return 1

        try:
            for result in self.results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 80), 2)

            output_path = f"MODIFIED_{self.filepath}"
            cv2.imwrite(output_path, self.image)
            logger.info("Imagem salva em %s", output_path)
            return 0
        except Exception:
            logger.exception("Erro ao salvar imagem processada")
            return 1

    def get_num_of_people(self) -> int:
        """
        Retorna o número de pessoas encontradas na imagem.
        """
        if self.results is None:
            return 0
        try:
            return sum(len(result.boxes) for result in self.results)
        except Exception:
            logger.exception("Erro ao contar pessoas nos resultados")
            return 0
