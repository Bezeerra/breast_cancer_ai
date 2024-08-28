from ultralytics import YOLO
import os


def clean_cuda_cache():
    import torch

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()


def run_model():
    clean_cuda_cache()
    model_path = "yolov8n.pt"
    data_path = "config.yaml"
    model = YOLO(model_path)

    results = model.train(
        data=data_path,
        epochs=200,
        imgsz=412
    )

    model.save("yolov8n_trained_412.pt")
    return results

if __name__ == "__main__":
    run_model()
