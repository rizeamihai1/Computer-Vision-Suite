from ultralytics import YOLO
import torch

def start_training():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Antrenarea va rula pe: {torch.cuda.get_device_name(0) if device==0 else 'CPU'}")

    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        name='detectie_scooby'
    )

if __name__ == '__main__':
    start_training()