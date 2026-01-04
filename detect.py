import cv2
import argparse
from ultralytics import YOLO
from config import CONFIG, VEHICLE_CLASSES, COLORS
from draw_utils import draw_statistics
from processors import process_image, process_video


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


def main():
    parser = argparse.ArgumentParser(description='Nhận diện phương tiện giao thông với YOLO')
    
    # Sử dụng giá trị từ CONFIG làm mặc định
    parser.add_argument('--source', type=str, default=CONFIG["source"],
                       help='Nguồn video (camera ID, đường dẫn video, hoặc đường dẫn ảnh)')
    parser.add_argument('--model', type=str, default=CONFIG["model"],
                       help='Đường dẫn model YOLO (mặc định: yolo11n.pt)')
    parser.add_argument('--output', type=str, default=CONFIG["output"],
                       help='Đường dẫn lưu kết quả')
    parser.add_argument('--confidence', type=float, default=CONFIG["confidence"],
                       help='Ngưỡng confidence (mặc định: 0.5)')
    parser.add_argument('--image', action='store_true', default=CONFIG["is_image"],
                       help='Xử lý như ảnh thay vì video')
    parser.add_argument('--device', type=str, default=CONFIG["device"],
                       help='Device để chạy model (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Hiển thị thông tin device
    print(f"Sử dụng device: {args.device}")
    
    # Tải model
    model = load_model(args.model)
    
    # Xử lý theo loại nguồn
    if args.image:
        process_image(model, args.source, args.output, args.confidence, args.device, detect_vehicles)
    else:
        process_video(model, args.source, args.output, args.confidence, args.device, detect_vehicles)


if __name__ == "__main__":
    main()
