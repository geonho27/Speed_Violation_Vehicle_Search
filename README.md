# Speed Violation Vehicle Search
<h2>과속 차량 자동 검출 시스템</h2>

🚗 고속도로 차량 속도 분석 프로젝트

프로젝트 개요

본 프로젝트는 YOLO11 모델을 활용하여 동영상에서 과속 차량을 자동으로 감지하고, 차량의 속도를 추정하는 프로그램입니다. 이 프로그램은 교통 안전을 강화하고, 과속 차량에 대한 효율적인 모니터링 시스템을 제공하기 위해 설계되었습니다.

# 주요 기능

* 차량 감지: YOLO11 모델을 사용하여 동영상 내의 차량을 정확하게 식별.

* 속도 추정: 차량의 위치 변화에 기반하여 초당 속도를 계산.

* 결과 시각화: 과속 차량을 강조 표시하고, 동영상 출력 파일로 저장.

# 동작 원리

* YOLO11 모델

YOLO (You Only Look Once) 모델은 실시간 객체 감지 기술 중 하나로, 입력 이미지에서 여러 객체를 한 번의 인퍼런스로 빠르게 탐지합니다. 본 프로젝트에서는 yolo11n.pt 모델 파일을 사용하여 차량 객체를 인식합니다.

* 속도 추정 알고리즘

속도 추정은 특정 프레임 간 차량의 픽셀 단위 이동 거리와 실제 환경에서의 거리 비율을 기반으로 계산됩니다:
픽셀 거리 변환(Pixel-to-World Transformation)을 사용했습니다.

시간과 거리: 차량이 ROI를 통과하는 데 걸린 시간과 이동 거리를 바탕으로 속도를 계산.

![고속도로 상황일떄](https://github.com/user-attachments/assets/89dd9cd0-7cbb-4340-bf07-b1a007505f33)
