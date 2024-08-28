# Detecção de Câncer de Mama em Mamografias

Este projeto tem como objetivo desenvolver um modelo para detecção de câncer de mama utilizando mamografias. Os dados utilizados neste trabalho foram retirados da competição [RSNA Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) no Kaggle.

## Integrantes do Grupo

- Matheus Bezerra
- Eduardo Amorim
- Matheus Gomes
- Pedro Sousa
- Thiago

## Descrição do Projeto

O câncer de mama é uma das principais causas de morte entre mulheres em todo o mundo, e a detecção precoce é crucial para aumentar as chances de sobrevivência. O objetivo deste trabalho é construir um modelo de aprendizado de máquina capaz de identificar sinais de câncer de mama em imagens de mamografias. O modelo será treinado e avaliado utilizando o conjunto de dados fornecido pela competição RSNA Breast Cancer Detection.

## Scripts necessários

O arquivo separete_images.py é utilizado para identificar quais mamografias já possuem filtros aplicados e quais ainda não foram filtradas. Para as imagens que ainda não possuem filtro, o script aplica automaticamente o filtro definido no arquivo filter_dudu.py.

Após a execução desse script, é necessário rodar o create_dataset.py. Este script é responsável por preparar o dataset no formato adequado para o YOLOv8, o modelo pré-treinado que estamos utilizando.

## Instruções para Execução

1. Clone o repositório:
   ```bash
   git clone https://github.com/Bezeerra/breast_cancer_ai
    ``` 

2. Instale os pacotes
    ``` bash
    pip install -r requirements.txt
    ```

2. Crie o modelo 
    ``` bash
    python3 main.py
    ```
