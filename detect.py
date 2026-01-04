import cv2
import argparse
from ultralytics import YOLO
CONFIG = {
    # Nguồn video/ảnh: 
    # - "0" hoặc "1" cho webcam
    # - Đường dẫn video: "D:/videos/traffic.mp4"
    # - Đường dẫn ảnh: "D:/images/street.jpg"
    "source": r"/home/samer/Desktop/DT/Learn/20252/detecting_congestion/video/AN___562_Th_Khe_20251231_112424.mp4",
    
    # Đường dẫn model YOLO (sẽ tự động tải nếu chưa có)
    "model": "yolo11n.pt",
    
    # Đường dẫn lưu kết quả (để None nếu không muốn lưu)
    # Ví dụ: "D:/output/result.mp4" hoặc "D:/output/result.jpg"
    "output": r"/home/samer/Desktop/DT/Learn/20252/detecting_congestion/result/AN___562_Th_Khe_20251231_112424.mp4",
    
    # Ngưỡng confidence (0.0 - 1.0)
    "confidence": 0.5,
    
    # True nếu xử lý ảnh, False nếu xử lý video/camera
    "is_image": False,
    
    # Device để chạy model:
    # - "cuda" hoặc "0" cho GPU NVIDIA
    # - "cuda:0", "cuda:1" cho GPU cụ thể
    # - "cpu" cho CPU
    "device": "cuda",
}


# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {
    0: 'Nguoi',
    1: 'Xe dap', 
    2: 'Xe o to',
    3: 'Xe may',
    5: 'Xe bus',
    7: 'Xe tai'
}

# Màu sắc cho từng loại phương tiện (BGR format)
COLORS = {
    0: (255, 0, 255),    # Người - Magenta
    1: (0, 255, 255),    # Xe đạp - Vàng
    2: (0, 255, 0),      # Xe ô tô - Xanh lá
    3: (255, 165, 0),    # Xe máy - Cam
    5: (255, 0, 0),      # Xe bus - Xanh dương
    7: (0, 0, 255)       # Xe tải - Đỏ
}


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


def draw_statistics(frame, vehicle_count):
    """
    Vẽ bảng thống kê lên frame
    Args:
        frame: Frame cần vẽ
        vehicle_count: Dict thống kê số lượng
    Returns:
        Frame đã được vẽ thống kê
    """
    # Vẽ nền cho bảng thống kê
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (200, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Vẽ tiêu đề
    cv2.putText(frame, "THONG KE:", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Vẽ số lượng từng loại
    y_offset = 60
    for label, count in vehicle_count.items():
        if count > 0:
            text = f"{label}: {count}"
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    # Tổng số phương tiện
    total = sum(vehicle_count.values())
    cv2.putText(frame, f"Tong: {total}", (20, y_offset + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame


def process_image(model, image_path, output_path=None, confidence=0.5, device="cuda"):
    """
    Xử lý ảnh đơn
    Args:
        model: Model YOLO
        image_path: Đường dẫn ảnh đầu vào
        output_path: Đường dẫn lưu ảnh kết quả
        confidence: Ngưỡng confidence
        device: Device để chạy inference (cuda/cpu)
    """
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Nhận diện
    frame, vehicle_count = detect_vehicles(model, frame, confidence, device)
    frame = draw_statistics(frame, vehicle_count)
    
    # In thống kê
    print("\n=== KẾT QUẢ NHẬN DIỆN ===")
    for label, count in vehicle_count.items():
        if count > 0:
            print(f"  {label}: {count}")
    print(f"  Tổng: {sum(vehicle_count.values())}")
    
    # Lưu hoặc hiển thị kết quả
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Đã lưu kết quả: {output_path}")
    
    # cv2.imshow("Nhan dien phuong tien", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def process_video(model, video_source, output_path=None, confidence=0.5, device="cuda"):
    """
    Xử lý video hoặc camera
    Args:
        model: Model YOLO
        video_source: Đường dẫn video hoặc camera ID (0, 1, ...)
        output_path: Đường dẫn lưu video kết quả
        confidence: Ngưỡng confidence
        device: Device để chạy inference (cuda/cpu)
    """
    # Mở video/camera
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Không thể mở nguồn video: {video_source}")
        return
    
    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"Độ phân giải: {width}x{height}, FPS: {fps}")
    
    # Lấy tổng số frame của video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Tổng số frame: {total_frames}")
    
    # Tạo video writer nếu cần lưu
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Bắt đầu xử lý video...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Nhận diện
        frame, vehicle_count = detect_vehicles(model, frame, confidence, device)
        frame = draw_statistics(frame, vehicle_count)
        
        # Hiển thị FPS
        cv2.putText(frame, f"Frame: {frame_count}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hiển thị
        # cv2.imshow("Nhan dien phuong tien - Nhan 'q' de thoat", frame)
        
        # Lưu frame nếu cần
        if writer:
            writer.write(frame)
        
        frame_count += 1
        
        # Hiển thị tiến trình
        if total_frames > 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rTiến trình: {frame_count}/{total_frames} ({progress:.1f}%)", end="", flush=True)
        
        # Xử lý phím
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('s'):
        #     screenshot_path = f"screenshot_{frame_count}.jpg"
        #     cv2.imwrite(screenshot_path, frame)
        #     print(f"Đã lưu: {screenshot_path}")
    
    # Giải phóng tài nguyên
    print()  # Xuống dòng sau progress bar
    cap.release()
    if writer:
        writer.release()
        print(f"Đã lưu video: {output_path}")
    print(f"Hoàn thành! Đã xử lý {frame_count} frames.")
    # cv2.destroyAllWindows()


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
        process_image(model, args.source, args.output, args.confidence, args.device)
    else:
        process_video(model, args.source, args.output, args.confidence, args.device)


if __name__ == "__main__":
    main()
