# Cấu hình cho ứng dụng nhận diện phương tiện giao thông

CONFIG = {
    # Nguồn video/ảnh: 
    # - "0" hoặc "1" cho webcam
    # - Đường dẫn video: "D:/videos/traffic.mp4"
    # - Đường dẫn ảnh: "D:/images/street.jpg"
    "source": r"/home/samer/Desktop/DT/Learn/20252/detecting_congestion/video/AN___562_Th_Khe_20251231_112424.mp4",
    
    # Đường dẫn model YOLO (sẽ tự động tải nếu chưa có)
    "model": "yolo11n.pt",
    
    # Đường dẫn lưu kết quả (để None nếu không muốn lưu)
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
