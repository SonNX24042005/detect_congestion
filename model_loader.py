from ultralytics import YOLO


def load_model(model_path='yolo11n.pt'):
    """
    Tải model YOLO
    Args:
        model_path: Đường dẫn đến file model (mặc định: yolo11n.pt)
    Returns:
        Model YOLO đã được tải
    """
    print(f"Đang tải model: {model_path}")
    model = YOLO(model_path)
    print("Đã tải model thành công!")
    return model
