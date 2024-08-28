import random

import pandas as pd
import os
from shutil import copyfile, move

from PIL import Image

# Caminhos dos diretórios
csv_path = 'images_train/train.csv'
images_dir = 'dataset/images/train/'
labels_dir = 'dataset/labels/train/'

os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(f"{labels_dir}/val", exist_ok=True)
os.makedirs(f"{labels_dir}/train", exist_ok=True)
os.makedirs(f"{images_dir}/val", exist_ok=True)
os.makedirs(f"{images_dir}/train", exist_ok=True)


df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    patient_id = row['patient_id']
    image_id = row['image_id']
    cancer_class = int(row['cancer'])

    image_name = f"{patient_id}@{image_id}.png"
    annotation_file = os.path.join(labels_dir, f"{patient_id}@{image_id}.txt")

    image_path = os.path.join('images_train/images_with_filter', image_name)
    if not os.path.exists(image_path):
        print(f"Imagem {image_name} não encontrada")
        continue

    # Carregar a imagem e obter as dimensões
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    copyfile(image_path, os.path.join(images_dir, image_name))

    # Coordenadas do bounding box baseado nas dimensões da imagem
    x_center = image_width / 2.0
    y_center = image_height / 2.0
    width = image_width
    height = image_height

    # Normalizar as coordenadas e dimensões para o formato YOLOv8 (entre 0 e 1)
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Salva as anotações no formato YOLOv8
    with open(annotation_file, 'w') as f:
        f.write(f"{cancer_class} {x_center} {y_center} {width} {height}\n")


val_ratio = 0.2

image_files = os.listdir(images_dir)
random.shuffle(image_files)
num_val_images = int(len(image_files) * val_ratio)

# Mover imagens e anotações
for i in range(num_val_images):
    image_file = image_files[i]
    label_file = image_file.replace('.png', '.txt')
    move(os.path.join(images_dir, image_file), os.path.join('dataset/images/val/', image_file))
    move(os.path.join(labels_dir, label_file), os.path.join('dataset/labels/val/', label_file))
