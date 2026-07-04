import sys

from yolo import Yolo

def read_line() -> tuple:
    """
    Lê a imagem da fila do RU, cuja localização é passada como argumento do programa 
    (python main.py [argumento]), e retorna uma tupla com dados da fila. O primeiro elemento dessa tupla
    é o número de pessoas na fila e o segundo é o tempo de espera. A função retorna None em caso de erro.
    """

    if len(sys.argv) == 1: # A localização da imagem não foi passada
        print("Imagem não informada")
        return None

    try:
        print("Lendo", sys.argv[1])

        yolo = Yolo()
        yolo.read(sys.argv[1])
        yolo.save()

        people_in_line = yolo.get_num_of_people()
        waiting_time = 0

        return (people_in_line, waiting_time)
    except:
        return None


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    print("Retorno da função read_line:", read_line())
