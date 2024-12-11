import cv2
from ultralytics import YOLO
import numpy as np
import os

# YOLO 모델 로드
model = YOLO("yolo11n.pt")  # YOLO 모델 파일 경로
model.cpu()  # GPU 사용 비활성화

# 비디오 파일 경로
video_file_path = 'data2/고속도로영상합본.mp4'

# 속도 임계값 설정
speed_threshold_kph = 100  # 50 km/h 이상인 객체만 빨간색 박스로 표시
slowdown_factor = 0.5  # 저장 비디오 속도를 원래의 50%로 설정

# 비디오 파일 처리
cap = cv2.VideoCapture(video_file_path)
if not cap.isOpened():
    print(f"Could not open the video file: {video_file_path}")
    exit()

# 비디오의 속성 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 저장 비디오 속도 계산
save_fps = fps * slowdown_factor

# 저장 디렉토리 생성
output_dir = "data/speed_violation_segments"
os.makedirs(output_dir, exist_ok=True)

# 초기화 변수
ret, prev_frame = cap.read()
if not ret:
    print("Error reading the first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
save_video = False
video_counter = 0
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow를 사용한 객체 속도 계산
    results = model(frame)
    detections = []
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = model.names[int(detection.cls)]
            if class_name in ["car", "truck", "bus"]:
                detections.append([x1, y1, x2, y2])

    points = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for x1, y1, x2, y2 in detections], dtype=np.float32)

    if points is not None and len(points) > 0:
        new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)

        # 속도 계산 및 경고 표시
        for i, (new_point, s) in enumerate(zip(new_points, status)):
            if s:
                x, y = new_point.ravel()
                x1, y1, x2, y2 = detections[i]
                move_distance = np.linalg.norm(new_point - points[i])
                speed_in_mps = move_distance / (1 / fps)
                speed_in_kph = speed_in_mps * 3.6

                if speed_in_kph > speed_threshold_kph:
                    # 빨간색 박스 및 텍스트 추가
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Speed: {speed_in_kph:.2f} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 저장 시작
                    if not save_video:
                        video_counter += 1
                        save_path = os.path.join(output_dir, f"violation_{video_counter}.mp4")
                        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), save_fps, (frame_width, frame_height))
                        save_video = True

                    # 현재 프레임 저장
                    if out:
                        out.write(frame)
                else:
                    # 초과 속도가 끝날 때 저장 중지
                    if save_video:
                        out.release()
                        save_video = False
            else:
                # Optical Flow 실패 시 저장 종료
                if save_video:
                    out.release()
                    save_video = False

    # 현재 프레임을 이전 프레임으로 설정
    prev_gray = gray

    # 실행 영상 표시
    cv2.imshow("YOLO Results with Optical Flow", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 저장 종료
if save_video and out:
    out.release()

# 비디오 파일 닫기
cap.release()
cv2.destroyAllWindows()

print(f"Saved violation video segments in: {output_dir}")
