import os
from pathlib import Path

import cv2

from ultralytics import YOLO, solutions


def main():
    # Carrega o modelo YOLO
    model = YOLO(os.path.join(Path(__file__).parent, 'runs/detect/train4/weights/best.pt'))

    # Carrega a imagem
    image_path = os.path.join(Path(__file__).parent, 'images_train/images_with_filter_2/236@1531879119.png')
    im0 = cv2.imread(image_path)
    assert im0 is not None, "Erro ao ler o arquivo de imagem"

    # Inicializa o objeto de mapa de calor
    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA,
        view_img=True,
        shape="circle",
        names=model.names,
    )

    # Realiza o rastreamento na imagem
    tracks = model.track(im0, persist=True, show=False)

    # Gera o mapa de calor sobre a imagem
    im0 = heatmap_obj.generate_heatmap(im0, tracks)

    # Salva a imagem resultante
    output_path = "heatmap_output.png"
    cv2.imwrite(output_path, im0)

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
