import argparse
from config import CONFIG
from processors import process_image, process_video
from detector import detect_vehicles
from model_loader import load_model


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
