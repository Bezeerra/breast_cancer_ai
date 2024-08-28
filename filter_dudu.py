import cv2
import os


fator_escurecimento = 0.67


def filter_dudu(image_path: str, output_path: str):
    imagem = cv2.imread(image_path)
    # Converter para escala de cinza
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Aplicar equalização de histograma
    imagem_equalizada = cv2.equalizeHist(imagem_gray)

    # Escurecer a imagem equalizada
    imagem_escurecida = cv2.convertScaleAbs(imagem_equalizada, alpha=fator_escurecimento, beta=0)

    # Salvar a imagem processada no diretório de destino
    cv2.imwrite(output_path, imagem_escurecida)
