from ultralytics import YOLO
import os


def clean_cuda_cache():
    import torch

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()


def run_model():
    model_path = "yolov8m.pt"
    data_path = "config.yaml"
    model = YOLO(model_path)

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640
    )

    model.save("yolov8m_trained.pt")
    return results

if __name__ == "__main__":
    clean_cuda_cache()
    run_model()
