import cv2
from draw_utils import draw_statistics


def process_image(model, image_path, output_path=None, confidence=0.5, device="cuda", detect_vehicles_func=None):
    """
    Xử lý ảnh đơn
    Args:
        model: Model YOLO
        image_path: Đường dẫn ảnh đầu vào
        output_path: Đường dẫn lưu ảnh kết quả
        confidence: Ngưỡng confidence
        device: Device để chạy inference (cuda/cpu)
        detect_vehicles_func: Hàm detect_vehicles
    """
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Nhận diện
    frame, vehicle_count = detect_vehicles_func(model, frame, confidence, device)
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


def process_video(model, video_source, output_path=None, confidence=0.5, device="cuda", detect_vehicles_func=None):
    """
    Xử lý video hoặc camera
    Args:
        model: Model YOLO
        video_source: Đường dẫn video hoặc camera ID (0, 1, ...)
        output_path: Đường dẫn lưu video kết quả
        confidence: Ngưỡng confidence
        device: Device để chạy inference (cuda/cpu)
        detect_vehicles_func: Hàm detect_vehicles
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
        frame, vehicle_count = detect_vehicles_func(model, frame, confidence, device)
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
