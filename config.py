# Cấu hình cho ứng dụng nhận diện phương tiện giao thông

CONFIG = {
    # Nguồn video/ảnh: 
    # - "0" hoặc "1" cho webcam
    # - Đường dẫn video: "D:/videos/traffic.mp4"
    # - Đường dẫn ảnh: "D:/images/street.jpg"
    "source": r"video/AN___562_Th_Khe_20251231_112424.mp4",
    
    # Đường dẫn model YOLO (sẽ tự động tải nếu chưa có)
    "model": "model/yolo12n.pt",
    
    # Đường dẫn lưu kết quả (để None nếu không muốn lưu)
    "output": r"result/AN___562_Th_Khe_20251231_1124242.mp4",
    
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


# ==================== CẤU HÌNH PHÁT HIỆN ÙN TẮC ====================
CONGESTION_CONFIG = {
    # Ngưỡng số lượng phương tiện để coi là ùn tắc
    "vehicle_count_threshold": 15,
    
    # Ngưỡng tỷ lệ chiếm đường (%) để coi là ùn tắc
    "occupancy_threshold": 25.0,
    
    # Ngưỡng mật độ phương tiện (phương tiện/10000 pixel²)
    "density_threshold": 0.5,
    
    # Ngưỡng tốc độ trung bình (pixel/frame) - dưới ngưỡng này coi là chậm
    "speed_threshold": 5.0,
    
    # Số frame để tính trung bình (smoothing)
    "smoothing_window": 10,
    
    # Trọng số cho các tiêu chí (tổng = 1.0)
    "weights": {
        "vehicle_count": 0.25,
        "occupancy": 0.35,
        "density": 0.20,
        "speed": 0.20
    },
    
    # Ngưỡng điểm tổng hợp để xác định mức độ ùn tắc
    "congestion_levels": {
        "low": 0.3,      # < 30%: Thông thoáng
        "medium": 0.5,   # 30-50%: Đông đúc
        "high": 0.7,     # 50-70%: Ùn tắc nhẹ
        "severe": 0.85   # > 85%: Ùn tắc nghiêm trọng
    },
    
    # ==================== CẤU HÌNH TỐI ƯU TỐC ĐỘ ====================
    # Bật/tắt optical flow (tắt sẽ nhanh hơn nhiều)
    "enable_optical_flow": False,
    
    # Chỉ phân tích ùn tắc mỗi N frame (1 = mọi frame, 3 = mỗi 3 frame)
    "analyze_every_n_frames": 3,
    
    # Resize frame để tính optical flow (nhỏ hơn = nhanh hơn)
    "optical_flow_scale": 0.25,
    
    # Chỉ chạy YOLO detection mỗi N frame
    "detect_every_n_frames": 2,
    
    # ==================== CẤU HÌNH NÉN VIDEO ĐẦU RA ====================
    # Tỷ lệ resize video đầu ra (1.0 = giữ nguyên, 0.5 = giảm 50%)
    "output_scale": 0.75,
    
    # FPS video đầu ra (0 = giữ nguyên FPS gốc)
    "output_fps": 0,
    
    # Codec video (XVID hoặc H264 cho file nhỏ hơn)
    "video_codec": "XVID"
}
