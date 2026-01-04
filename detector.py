import cv2
from config import VEHICLE_CLASSES, COLORS


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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label với confidence
            text = f"{label}: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Vẽ nền cho text
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Cập nhật thống kê
            vehicle_count[label] += 1
    
    return frame, vehicle_count
