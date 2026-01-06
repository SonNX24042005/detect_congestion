import cv2
import numpy as np
from collections import deque
from draw_utils import draw_statistics


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


class CongestionDetector:
    """Lớp phát hiện ùn tắc giao thông"""
    
    def __init__(self, config=None):
        self.config = config or CONGESTION_CONFIG
        self.prev_frame_gray = None
        self.prev_boxes = []
        self.history = {
            "vehicle_count": deque(maxlen=self.config["smoothing_window"]),
            "occupancy": deque(maxlen=self.config["smoothing_window"]),
            "density": deque(maxlen=self.config["smoothing_window"]),
            "speed": deque(maxlen=self.config["smoothing_window"]),
            "congestion_score": deque(maxlen=self.config["smoothing_window"])
        }
        
    def calculate_occupancy(self, boxes, frame_area):
        """
        Tính tỷ lệ chiếm đường (diện tích bbox / diện tích frame)
        """
        if frame_area == 0:
            return 0.0
        
        total_box_area = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            total_box_area += box_area
        
        occupancy = (total_box_area / frame_area) * 100
        return min(occupancy, 100.0)  # Giới hạn tối đa 100%
    
    def calculate_density(self, vehicle_count, frame_area):
        """
        Tính mật độ phương tiện (số phương tiện / diện tích)
        """
        if frame_area == 0:
            return 0.0
        # Mật độ trên 10000 pixel²
        density = (vehicle_count / frame_area) * 10000
        return density
    
    def estimate_speed_optical_flow(self, frame_gray, boxes):
        """
        Ước lượng tốc độ di chuyển bằng Optical Flow (đã tối ưu)
        """
        # Nếu tắt optical flow, trả về giá trị mặc định
        if not self.config.get("enable_optical_flow", False):
            return 2.5  # Giá trị trung bình mặc định
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = frame_gray.copy()
            self.prev_boxes = boxes
            return 0.0
        
        if len(boxes) == 0:
            self.prev_frame_gray = frame_gray.copy()
            self.prev_boxes = boxes
            return 0.0
        
        speeds = []
        
        try:
            # Resize frame nhỏ hơn để tính optical flow nhanh hơn
            scale = self.config.get("optical_flow_scale", 0.25)
            small_prev = cv2.resize(self.prev_frame_gray, None, fx=scale, fy=scale)
            small_curr = cv2.resize(frame_gray, None, fx=scale, fy=scale)
            
            # Sử dụng Farneback optical flow với params tối ưu cho tốc độ
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            
            # Tính tốc độ trung bình trong các vùng bounding box (scaled)
            for box in boxes:
                x1, y1, x2, y2 = [int(coord * scale) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(flow.shape[1], x2)
                y2 = min(flow.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    flow_region = flow[y1:y2, x1:x2]
                    magnitude = np.sqrt(flow_region[..., 0]**2 + flow_region[..., 1]**2)
                    avg_speed = np.mean(magnitude) / scale  # Scale back
                    speeds.append(avg_speed)
        except Exception:
            pass
        
        self.prev_frame_gray = frame_gray.copy()
        self.prev_boxes = boxes
        
        return np.mean(speeds) if speeds else 0.0
    
    def calculate_congestion_score(self, vehicle_count, occupancy, density, speed):
        """
        Tính điểm ùn tắc tổng hợp từ các tiêu chí
        """
        weights = self.config["weights"]
        
        # Chuẩn hóa các chỉ số về thang 0-1
        # Vehicle count score
        count_score = min(vehicle_count / self.config["vehicle_count_threshold"], 1.0)
        
        # Occupancy score
        occ_score = min(occupancy / self.config["occupancy_threshold"], 1.0)
        
        # Density score
        density_score = min(density / self.config["density_threshold"], 1.0)
        
        # Speed score (đảo ngược - tốc độ thấp = ùn tắc cao)
        if self.config["speed_threshold"] > 0:
            speed_score = max(0, 1.0 - (speed / self.config["speed_threshold"]))
        else:
            speed_score = 0.0
        
        # Tính điểm tổng hợp có trọng số
        total_score = (
            weights["vehicle_count"] * count_score +
            weights["occupancy"] * occ_score +
            weights["density"] * density_score +
            weights["speed"] * speed_score
        )
        
        return min(total_score, 1.0)
    
    def get_congestion_level(self, score):
        """
        Xác định mức độ ùn tắc từ điểm tổng hợp
        """
        levels = self.config["congestion_levels"]
        
        if score < levels["low"]:
            return "THONG THOANG", (0, 255, 0)  # Xanh lá
        elif score < levels["medium"]:
            return "DONG DUC", (0, 255, 255)  # Vàng
        elif score < levels["high"]:
            return "UN TAC NHE", (0, 165, 255)  # Cam
        elif score < levels["severe"]:
            return "UN TAC", (0, 0, 255)  # Đỏ
        else:
            return "UN TAC NGHIEM TRONG", (0, 0, 139)  # Đỏ đậm
    
    def analyze(self, frame, boxes, vehicle_count_dict):
        """
        Phân tích ùn tắc cho một frame
        Returns: dict chứa các chỉ số và kết quả phân tích
        """
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Chuyển frame sang grayscale cho optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tính các chỉ số
        total_vehicles = sum(vehicle_count_dict.values())
        occupancy = self.calculate_occupancy(boxes, frame_area)
        density = self.calculate_density(total_vehicles, frame_area)
        speed = self.estimate_speed_optical_flow(frame_gray, boxes)
        
        # Lưu vào history để smoothing
        self.history["vehicle_count"].append(total_vehicles)
        self.history["occupancy"].append(occupancy)
        self.history["density"].append(density)
        self.history["speed"].append(speed)
        
        # Tính giá trị trung bình (smoothed)
        avg_count = np.mean(self.history["vehicle_count"])
        avg_occupancy = np.mean(self.history["occupancy"])
        avg_density = np.mean(self.history["density"])
        avg_speed = np.mean(self.history["speed"])
        
        # Tính điểm ùn tắc
        congestion_score = self.calculate_congestion_score(
            avg_count, avg_occupancy, avg_density, avg_speed
        )
        self.history["congestion_score"].append(congestion_score)
        
        # Smoothed congestion score
        final_score = np.mean(self.history["congestion_score"])
        
        # Xác định mức độ ùn tắc
        level, color = self.get_congestion_level(final_score)
        
        return {
            "vehicle_count": total_vehicles,
            "occupancy": occupancy,
            "density": density,
            "speed": speed,
            "avg_vehicle_count": avg_count,
            "avg_occupancy": avg_occupancy,
            "avg_density": avg_density,
            "avg_speed": avg_speed,
            "congestion_score": final_score,
            "congestion_level": level,
            "congestion_color": color
        }


def draw_congestion_info(frame, congestion_data):
    """
    Vẽ thông tin ùn tắc lên frame
    """
    height, width = frame.shape[:2]
    
    # Vẽ panel thông tin ở góc trên bên trái
    panel_width = 320
    panel_height = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Tiêu đề
    cv2.putText(frame, "PHAN TICH UN TAC", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Các chỉ số
    y_offset = 60
    line_height = 25
    
    info_lines = [
        f"So phuong tien: {congestion_data['vehicle_count']}",
        f"Ty le chiem duong: {congestion_data['occupancy']:.1f}%",
        f"Mat do: {congestion_data['density']:.3f}",
        f"Toc do TB: {congestion_data['avg_speed']:.1f} px/f",
        f"Diem un tac: {congestion_data['congestion_score']*100:.1f}%"
    ]
    
    for line in info_lines:
        cv2.putText(frame, line, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
    
    # Vẽ mức độ ùn tắc (to và nổi bật)
    level = congestion_data['congestion_level']
    color = congestion_data['congestion_color']
    
    # Vẽ banner trạng thái ở góc trên bên phải
    banner_width = 280
    banner_height = 50
    banner_x = width - banner_width - 10
    
    cv2.rectangle(frame, (banner_x, 10), (width - 10, 10 + banner_height), color, -1)
    cv2.rectangle(frame, (banner_x, 10), (width - 10, 10 + banner_height), (255, 255, 255), 2)
    
    # Text trạng thái
    text_size = cv2.getTextSize(level, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = banner_x + (banner_width - text_size[0]) // 2
    text_y = 10 + (banner_height + text_size[1]) // 2
    cv2.putText(frame, level, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Vẽ thanh progress bar cho mức độ ùn tắc
    bar_x = 20
    bar_y = y_offset + 10
    bar_width = panel_width - 40
    bar_height = 20
    
    # Nền thanh
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    
    # Phần đã fill theo score
    fill_width = int(bar_width * congestion_data['congestion_score'])
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    
    # Viền
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    
    return frame


def process_image(model, image_path, output_path=None, confidence=0.5, device="cuda", detect_vehicles_func=None):
    """
    Xử lý ảnh đơn với phát hiện ùn tắc
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
    
    # Khởi tạo congestion detector
    congestion_detector = CongestionDetector()
    
    # Nhận diện
    frame_result, vehicle_count, boxes = detect_vehicles_with_boxes(model, frame, confidence, device, detect_vehicles_func)
    frame_result = draw_statistics(frame_result, vehicle_count)
    
    # Phân tích ùn tắc
    congestion_data = congestion_detector.analyze(frame_result, boxes, vehicle_count)
    frame_result = draw_congestion_info(frame_result, congestion_data)
    
    # In thống kê
    print("\n=== KẾT QUẢ NHẬN DIỆN ===")
    for label, count in vehicle_count.items():
        if count > 0:
            print(f"  {label}: {count}")
    print(f"  Tổng: {sum(vehicle_count.values())}")
    
    print("\n=== PHÂN TÍCH ÙN TẮC ===")
    print(f"  Tỷ lệ chiếm đường: {congestion_data['occupancy']:.1f}%")
    print(f"  Mật độ: {congestion_data['density']:.3f}")
    print(f"  Điểm ùn tắc: {congestion_data['congestion_score']*100:.1f}%")
    print(f"  Trạng thái: {congestion_data['congestion_level']}")
    
    # Lưu hoặc hiển thị kết quả
    if output_path:
        cv2.imwrite(output_path, frame_result)
        print(f"Đã lưu kết quả: {output_path}")


def detect_vehicles_with_boxes(model, frame, confidence, device, detect_vehicles_func):
    """
    Wrapper để lấy cả bounding boxes từ detect_vehicles
    """
    from config import VEHICLE_CLASSES, COLORS
    
    # Chạy inference
    results = model(frame, verbose=False, device=device)[0]
    
    # Thống kê và lấy boxes
    vehicle_count = {name: 0 for name in VEHICLE_CLASSES.values()}
    boxes = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls_id in VEHICLE_CLASSES and conf >= confidence:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            
            label = VEHICLE_CLASSES[cls_id]
            color = COLORS[cls_id]
            vehicle_count[label] += 1
            
            # Vẽ bounding box trực tiếp (không gọi lại model)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, vehicle_count, boxes


def process_video(model, video_source, output_path=None, confidence=0.5, device="cuda", detect_vehicles_func=None):
    """
    Xử lý video hoặc camera với phát hiện ùn tắc (ĐÃ TỐI ƯU TỐC ĐỘ)
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
    
    # Lấy cấu hình nén video
    output_scale = CONGESTION_CONFIG.get("output_scale", 1.0)
    output_fps = CONGESTION_CONFIG.get("output_fps", 0)
    video_codec = CONGESTION_CONFIG.get("video_codec", "XVID")
    
    # Tính kích thước output
    out_width = int(width * output_scale)
    out_height = int(height * output_scale)
    out_fps = output_fps if output_fps > 0 else fps
    
    # Đảm bảo kích thước chẵn (yêu cầu của một số codec)
    out_width = out_width - (out_width % 2)
    out_height = out_height - (out_height % 2)
    
    print(f"Độ phân giải gốc: {width}x{height}, FPS: {fps}")
    print(f"Độ phân giải đầu ra: {out_width}x{out_height}, FPS: {out_fps}, Codec: {video_codec}")
    
    # Lấy tổng số frame của video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Tổng số frame: {total_frames}")
    
    # Tạo video writer nếu cần lưu
    writer = None
    if output_path:
        # Sử dụng codec nén tốt hơn
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))
    
    # Khởi tạo congestion detector
    congestion_detector = CongestionDetector()
    
    # Lấy cấu hình tối ưu
    detect_interval = CONGESTION_CONFIG.get("detect_every_n_frames", 2)
    analyze_interval = CONGESTION_CONFIG.get("analyze_every_n_frames", 3)
    
    print(f"Tối ưu: Detection mỗi {detect_interval} frame, Phân tích mỗi {analyze_interval} frame")
    
    # Thống kê ùn tắc cho toàn video
    congestion_stats = {
        "total_frames": 0,
        "congestion_frames": 0,
        "max_congestion_score": 0,
        "avg_congestion_score": 0,
        "congestion_scores": []
    }
    
    print("Bắt đầu xử lý video với phát hiện ùn tắc...")
    frame_count = 0
    
    # Cache kết quả detection và congestion
    cached_vehicle_count = {name: 0 for name in ['Nguoi', 'Xe dap', 'Xe o to', 'Xe may', 'Xe bus', 'Xe tai']}
    cached_boxes = []
    cached_congestion_data = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ chạy detection mỗi N frame
        if frame_count % detect_interval == 0:
            frame_result, cached_vehicle_count, cached_boxes = detect_vehicles_with_boxes(
                model, frame, confidence, device, detect_vehicles_func
            )
        else:
            # Sử dụng frame gốc và vẽ lại boxes từ cache
            frame_result = frame.copy()
            from config import VEHICLE_CLASSES, COLORS
            for i, box in enumerate(cached_boxes):
                x1, y1, x2, y2 = box
                # Vẽ box đơn giản từ cache
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        frame_result = draw_statistics(frame_result, cached_vehicle_count)
        
        # Chỉ phân tích ùn tắc mỗi N frame
        if frame_count % analyze_interval == 0:
            cached_congestion_data = congestion_detector.analyze(frame_result, cached_boxes, cached_vehicle_count)
        
        # Vẽ thông tin ùn tắc (dùng cache nếu có)
        if cached_congestion_data:
            frame_result = draw_congestion_info(frame_result, cached_congestion_data)
            
            # Cập nhật thống kê
            congestion_stats["total_frames"] += 1
            congestion_stats["congestion_scores"].append(cached_congestion_data["congestion_score"])
            if cached_congestion_data["congestion_score"] > congestion_stats["max_congestion_score"]:
                congestion_stats["max_congestion_score"] = cached_congestion_data["congestion_score"]
            if cached_congestion_data["congestion_score"] > CONGESTION_CONFIG["congestion_levels"]["medium"]:
                congestion_stats["congestion_frames"] += 1
        
        # Hiển thị frame number
        cv2.putText(frame_result, f"Frame: {frame_count}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Resize frame nếu cần trước khi lưu
        if writer:
            if output_scale != 1.0:
                frame_output = cv2.resize(frame_result, (out_width, out_height), interpolation=cv2.INTER_AREA)
            else:
                frame_output = frame_result
            writer.write(frame_output)
        
        frame_count += 1
        
        # Hiển thị tiến trình với trạng thái ùn tắc
        if total_frames > 0 and frame_count % 10 == 0:  # Cập nhật console mỗi 10 frame
            progress = (frame_count / total_frames) * 100
            status = cached_congestion_data['congestion_level'] if cached_congestion_data else "..."
            print(f"\rTiến trình: {frame_count}/{total_frames} ({progress:.1f}%) | Trạng thái: {status}    ", end="", flush=True)
    
    # Tính thống kê cuối cùng
    if congestion_stats["congestion_scores"]:
        congestion_stats["avg_congestion_score"] = np.mean(congestion_stats["congestion_scores"])
    
    # Giải phóng tài nguyên
    print()
    cap.release()
    if writer:
        writer.release()
        print(f"Đã lưu video: {output_path}")
    
    # In báo cáo ùn tắc
    print(f"\n{'='*50}")
    print("BÁO CÁO PHÂN TÍCH ÙN TẮC")
    print(f"{'='*50}")
    print(f"Tổng số frame đã xử lý: {congestion_stats['total_frames']}")
    print(f"Số frame có ùn tắc: {congestion_stats['congestion_frames']}")
    if congestion_stats['total_frames'] > 0:
        congestion_percent = (congestion_stats['congestion_frames'] / congestion_stats['total_frames']) * 100
        print(f"Tỷ lệ thời gian ùn tắc: {congestion_percent:.1f}%")
    print(f"Điểm ùn tắc cao nhất: {congestion_stats['max_congestion_score']*100:.1f}%")
    print(f"Điểm ùn tắc trung bình: {congestion_stats['avg_congestion_score']*100:.1f}%")
    
    # Đánh giá tổng thể
    avg_score = congestion_stats['avg_congestion_score']
    if avg_score < 0.3:
        overall = "GIAO THÔNG THÔNG THOÁNG"
    elif avg_score < 0.5:
        overall = "GIAO THÔNG ĐÔNG ĐÚC"
    elif avg_score < 0.7:
        overall = "CÓ ÙN TẮC NHẸ"
    else:
        overall = "ÙN TẮC NGHIÊM TRỌNG"
    print(f"Đánh giá tổng thể: {overall}")
    print(f"{'='*50}")
    
    print(f"Hoàn thành! Đã xử lý {frame_count} frames.")
