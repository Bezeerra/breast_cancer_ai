import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def create_heatmap(image_path, model_path, save_heatmap_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)[0]  # Pegamos o primeiro resultado da lista

    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    for bbox in results.boxes:
        x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
        conf = bbox.conf

        # Adicionar ao heatmap baseado na confiança da predição
        heatmap[y_min:y_max, x_min:x_max] += conf.item()

    # Normalizar o heatmap para o intervalo [0, 1]
    heatmap = np.clip(heatmap, 0, 1)

    # Redimensionar para o tamanho original da imagem
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Aplicar colormap para visualizar o heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Sobrepor o heatmap na imagem original
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

    # Salvar o heatmap
    cv2.imwrite(save_heatmap_path, superimposed_img)

    # Mostrar o resultado
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


create_heatmap(
    os.path.join(Path(__file__).parent, 'images_train/images_with_filter_2/236@1531879119.png'),
    os.path.join(Path(__file__).parent, 'runs/detect/train4/weights/best.pt'),
    "heatmap.png"
)
