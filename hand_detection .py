import mediapipe as mp
import cv2
from Quartz import CGEventCreateScrollWheelEvent, kCGScrollEventUnitPixel, kCGHIDEventTap, CGEventCreateMouseEvent, kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventMouseMoved
from Quartz.CoreGraphics import CGEventPost, kCGHIDEventTap, kCGMouseButtonLeft
from AppKit import NSCursor
# 初始化手部特征点检测模块
hand_landmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions

# 创建手部特征点检测器
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)
landmarker = hand_landmarker.create_from_options(options)

# 视频流处理
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("无法打开摄像头")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
            
        # 将帧转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部特征点
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        
        # 绘制手部特征点和连接线
        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # 根据左右手设置不同颜色
                hand_type = detection_result.handedness[idx][0].category_name
                point_color = (0, 255, 0) if hand_type == 'Left' else (0, 0, 255)  # 左:绿色, 右:红色
                line_color = (255, 0, 0) if hand_type == 'Left' else (255, 255, 0)  # 左:蓝色, 右:黄色
                
                # 绘制关键点
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, point_color, -1)
                
                # 绘制连接线
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    x0 = int(hand_landmarks[start_idx].x * frame.shape[1])
                    y0 = int(hand_landmarks[start_idx].y * frame.shape[0])
                    x1 = int(hand_landmarks[end_idx].x * frame.shape[1])
                    y1 = int(hand_landmarks[end_idx].y * frame.shape[0])
                    cv2.line(frame, (x0, y0), (x1, y1), line_color, 2)
                
                                # 计算拇指根(1)到小指根(17)的距离
                thumb_base = hand_landmarks[1]
                pinky_base = hand_landmarks[17]
                
                thumb_base_x = int(thumb_base.x * frame.shape[1])
                thumb_base_y = int(thumb_base.y * frame.shape[0])
                pinky_base_x = int(pinky_base.x * frame.shape[1])
                pinky_base_y = int(pinky_base.y * frame.shape[0])
                
                base_distance = ((thumb_base_x - pinky_base_x)**2 + (thumb_base_y - pinky_base_y)**2)**0.5
                
        
        # 检测食指和中指指尖并拢模拟点击
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) < 2:
            for hand_landmarks in detection_result.hand_landmarks:
                index_tip = hand_landmarks[8]  # 食指尖
                middle_tip = hand_landmarks[12]  # 中指尖
                
                # 转换为像素坐标
                index_x = int(index_tip.x * frame.shape[1])
                index_y = int(index_tip.y * frame.shape[0])
                middle_x = int(middle_tip.x * frame.shape[1])
                middle_y = int(middle_tip.y * frame.shape[0])
                
                # 计算距离
                distance = ((index_x - middle_x)**2 + (index_y - middle_y)**2)**0.5
                
                
                # 根据距离动态控制鼠标按下状态
                if distance < base_distance * 0.3:
                    # cv2.putText(frame, '按下', (index_x, index_y-20), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # 创建并发送鼠标按下事件
                    if 'mouse_down' not in locals():
                        mouse_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (index_x, index_y), kCGMouseButtonLeft)
                        CGEventPost(kCGHIDEventTap, mouse_down)
                        NSCursor.hide()  # 隐藏鼠标
                    # 移动鼠标到指尖位置
                    mouse_move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (index_x, index_y), kCGMouseButtonLeft)
                    CGEventPost(kCGHIDEventTap, mouse_move)
                else:
                    # 创建并发送鼠标释放事件
                    if 'mouse_down' in locals():
                        mouse_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (index_x, index_y), kCGMouseButtonLeft)
                        CGEventPost(kCGHIDEventTap, mouse_up)
                        NSCursor.unhide()  # 恢复鼠标显示
                        del mouse_down
        
        # 当检测到两只手时且两手食指和中指并拢时计算掌心距离
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
            # 检查两手食指和中指是否并拢
            fingers_together = True
            for hand_landmarks in detection_result.hand_landmarks:
                index_tip = hand_landmarks[8]  # 食指尖
                middle_tip = hand_landmarks[12]  # 中指尖
                
                # 转换为像素坐标
                index_x = int(index_tip.x * frame.shape[1])
                index_y = int(index_tip.y * frame.shape[0])
                middle_x = int(middle_tip.x * frame.shape[1])
                middle_y = int(middle_tip.y * frame.shape[0])
                
                # 计算距离
                distance = ((index_x - middle_x)**2 + (index_y - middle_y)**2)**0.5
                
                # 如果距离大于阈值，则手指未并拢
                if distance > base_distance * 0.3:
                    fingers_together = False
                    break
            
            # 只有两手食指和中指都并拢时才计算掌心距离
            if fingers_together:
                # 计算两只手的掌心点(0号点)距离
                palm1 = detection_result.hand_landmarks[0][0]
                palm2 = detection_result.hand_landmarks[1][0]
                
                palm1_x = int(palm1.x * frame.shape[1])
                palm1_y = int(palm1.y * frame.shape[0])
                palm2_x = int(palm2.x * frame.shape[1])
                palm2_y = int(palm2.y * frame.shape[0])
                
                palm_distance = ((palm1_x - palm2_x)**2 + (palm1_y - palm2_y)**2)**0.5
                
                # 显示掌心距离
                # cv2.putText(frame, f'Palm Distance: {palm_distance:.2f}', 
                #            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                #            0.7, (255, 255, 255), 2)
                
                # 绘制掌心连线
                # cv2.line(frame, (palm1_x, palm1_y), (palm2_x, palm2_y), (255, 255, 255), 2)
                
                # 根据掌心距离变化控制鼠标滚轮
                if 'prev_palm_distance' in locals():
                    # 计算距离变化量
                    delta = palm_distance - prev_palm_distance
                    
                    # 添加平滑滤波
                    if 'scroll_buffer' not in locals():
                        scroll_buffer = []
                    scroll_buffer.append(delta)
                    if len(scroll_buffer) > 5:  # 保持最近3帧数据
                        scroll_buffer.pop(0)
                    smoothed_delta = sum(scroll_buffer) / len(scroll_buffer)
                    
                    # 设置滚动速度和阈值
                    if abs(smoothed_delta) > 5:  # 最小触发阈值
                        scroll_speed = min(max(int(smoothed_delta * 0.2), -10), 10)  # 限制滚动速度范围
                        scroll_event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, scroll_speed)
                        CGEventPost(kCGHIDEventTap, scroll_event)
                
                prev_palm_distance = palm_distance
            
        # 显示结果
        cv2.imshow("tres.js手势控制", frame)
        
        # 按q键退出
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # ESC键也可以退出
            break
            
except Exception as e:
    print(f"发生错误: {e}")
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()