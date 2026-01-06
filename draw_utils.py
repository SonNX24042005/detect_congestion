import cv2


def draw_statistics(frame, vehicle_count):
    """
    Vẽ bảng thống kê lên frame (đã tắt)
    Args:
        frame: Frame cần vẽ
        vehicle_count: Dict thống kê số lượng
    Returns:
        Frame gốc không vẽ thống kê
    """
    # Đã tắt chức năng vẽ bảng thống kê
    return frame


def draw_bounding_box(frame, x1, y1, x2, y2, label, conf, color):
    """
    Vẽ bounding box và label lên frame
    Args:
        frame: Frame cần vẽ
        x1, y1, x2, y2: Tọa độ bounding box
        label: Nhãn phương tiện
        conf: Độ tin cậy
        color: Màu sắc (BGR)
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}: {conf:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                 (x1 + text_size[0], y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


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
