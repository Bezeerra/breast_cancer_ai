from pathlib import Path

import cv2
import os

from filter_dudu import filter_dudu


def separar_imagens(imagens, com_filtro, limiar=85):
    os.makedirs(com_filtro, exist_ok=True)

    for filename in os.listdir(imagens):
        if filename.endswith('.png'):
            img_path = os.path.join(imagens, filename)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Falha ao carregar a imagem: {img_path}")
                continue

            media = cv2.mean(img)[0]

            novo_caminho = os.path.join(com_filtro, filename)
            if media < limiar:
                cv2.imwrite(novo_caminho, img)
            else:
                filter_dudu(img_path, novo_caminho)


if __name__ == "__main__":
    imagens = os.path.join(Path(__file__).parent, "images_train/train_images")
    new_dataset_with_filter = os.path.join(Path(__file__).parent, "images_train/images_with_filter")
    new_dataset_without_filter = os.path.join(Path(__file__).parent, "images_train/images_without_filter")

    separar_imagens(imagens, new_dataset_with_filter)
