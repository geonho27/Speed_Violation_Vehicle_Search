import cv2
from ultralytics import YOLO
import numpy as np
import os

# 고속도로 영상 파일 경로
video_path = 'data2/고속도로영상test.mp4'

# 비디오 파일 확인
if not os.path.exists(video_path):
    print(f"The video file does not exist: {video_path}")
    exit()

# YOLO 모델 로드
model = YOLO("yolo11n.pt")  # YOLO 모델 파일 경로
model.cpu()  # GPU 사용 비활성화

# 속도 임계값 설정
speed_threshold_kph = 50  # 50 km/h 이상인 객체를 강조 표시

# 비디오 파일 처리
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Could not open the video file: {video_path}")
    exit()

# 비디오의 프레임 속도(fps) 및 해상도 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1 픽셀 당 실제 거리 (미터/픽셀 비율)
meters_per_pixel = 0.05  # 도로 상황에 맞게 조정 필요

# 비디오 출력 파일 설정
output_path = 'data2/고속도로_분석결과.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 추적할 차량 객체 리스트
tracked_objects = {}

# 비디오 프레임 단위 처리
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)  # YOLO 모델 실행
    detections = []

    # YOLO 결과에서 바운딩 박스 가져오기
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = model.names[int(detection.cls)]
            confidence = float(detection.conf)

            if class_name in ["car", "truck", "bus"]:
                detections.append((x1, y1, x2, y2, confidence))

    # 추적 및 속도 계산
    new_tracked_objects = {}
    for detection in detections:
        x1, y1, x2, y2, confidence = detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 이전 프레임에서의 객체 확인
        closest_id = None
        min_distance = float('inf')
        for obj_id, obj_data in tracked_objects.items():
            prev_x, prev_y = obj_data['center']
            distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            if distance < min_distance and distance < 50:  # 거리 임계값
                min_distance = distance
                closest_id = obj_id

        # 새 객체 또는 기존 객체 업데이트
        if closest_id is not None:
            speed_pixels_per_frame = min_distance
            speed_mps = speed_pixels_per_frame * fps * meters_per_pixel
            speed_kph = speed_mps * 3.6
            new_tracked_objects[closest_id] = {
                'center': (center_x, center_y),
                'speed_kph': speed_kph,
                'box': (x1, y1, x2, y2)
            }
        else:
            new_id = len(tracked_objects) + len(new_tracked_objects) + 1
            new_tracked_objects[new_id] = {
                'center': (center_x, center_y),
                'speed_kph': 0,
                'box': (x1, y1, x2, y2)
            }

    # 업데이트된 객체로 교체
    tracked_objects = new_tracked_objects

    # 화면에 박스와 속도 표시
    for obj_id, obj_data in tracked_objects.items():
        x1, y1, x2, y2 = obj_data['box']
        speed_kph = obj_data['speed_kph']

        color = (0, 255, 0) if speed_kph <= speed_threshold_kph else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {obj_id}, Speed: {speed_kph:.2f} km/h", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과를 비디오 파일에 저장
    out.write(frame)

    # 'q' 키를 눌러서 재생 종료
    cv2.imshow("YOLO Results with Speed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at: {output_path}")
