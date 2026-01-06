import cv2
from config import VEHICLE_CLASSES, COLORS
from draw_utils import draw_bounding_box


def detect_vehicles(model, frame, confidence=0.5, device="cuda"):
    """
    Nhận diện phương tiện trong frame
    Args:
        model: Model YOLO
        frame: Ảnh/frame cần nhận diện
        confidence: Ngưỡng confidence tối thiểu
        device: Device để chạy inference (cuda/cpu)
    Returns:
        Frame đã được vẽ bounding box và dict thống kê số lượng
    """
    # Chạy inference trên GPU/CPU
    results = model(frame, verbose=False, device=device)[0]
    
    # Thống kê số lượng từng loại phương tiện
    vehicle_count = {name: 0 for name in VEHICLE_CLASSES.values()}
    
    # Xử lý kết quả
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Kiểm tra xem có phải phương tiện giao thông không
        if cls_id in VEHICLE_CLASSES and conf >= confidence:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Lấy tên và màu
            label = VEHICLE_CLASSES[cls_id]
            color = COLORS[cls_id]
            
            # Vẽ bounding box
            draw_bounding_box(frame, x1, y1, x2, y2, label, conf, color)
            
            # Cập nhật thống kê
            vehicle_count[label] += 1
    
    return frame, vehicle_count

