import cv2


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
