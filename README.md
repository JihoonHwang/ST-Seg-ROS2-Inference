# Semantic Segmentation Package

## 환경 요구사항

- **Python**: 3.10
- **Numpy**: 1.24.4
- **CV2**: 4.11.0
- **ROS2**: Humble (Base)
- **PyTorch**: 2.1 (2.3)
- **MMCV**: 2.2.0
- **MMSegmentation**: 1.2.2

## 패키지 구조

```
semantic_segmentation/
├── config/                    # 설정 파일들
│   ├── config.yaml           # 기본 설정
│   ├── config_MIC.yaml       # MIC 버전 설정
│   ├── config_STseg_front.yaml   # 전방 카메라 설정
│   ├── config_STseg_back.yaml    # 후방 카메라 설정
│   ├── config_STseg_left.yaml    # 좌측 카메라 설정
│   └── config_STseg_right.yaml   # 우측 카메라 설정
├── launch/
│   └── semantic_segmentation.launch.py
└── semantic_segmentation/     # 소스 코드
    ├── semantic_segmentation.py
    ├── semantic_segmentation_MIC.py
    ├── semantic_segmentation_STseg_front.py
    ├── semantic_segmentation_STseg_back.py
    ├── semantic_segmentation_STseg_left.py
    └── semantic_segmentation_STseg_right.py
```

## 설정 (Configuration)

각 카메라별 설정은 `semantic_segmentation/config/config_STseg_{camera}.yaml` 파일에서 관리

### 설정 파라미터

- **rgb_topic**: 입력 RGB 이미지 토픽
- **seg_topic**: 출력 분할 결과 토픽  
- **model_config**: 모델 설정 파일 경로
- **model_weight**: 모델 가중치 파일 경로

### 기본 설정값

```yaml
# 예시 - 전방 카메라
rgb_topic: /zed_front/zed_node/left/image_rect_color/compressed
seg_topic: /seg_front/left/compressed
model_config: "~/path/to/model/config"
model_weight: "~/path/to/model/weight"
```

## 토픽

### 입력 토픽
- `/zed_front/zed_node/left/image_rect_color/compressed` (기본)
- 각 카메라별로 설정 파일에서 변경 가능

### 출력 토픽  
- `/seg_front/left/compressed` (기본)
- 각 카메라별로 설정 파일에서 변경 가능


## 사용법

### 단일 노드 실행

```bash
# 기본 버전
ros2 launch semantic_segmentation semantic_segmentation.launch.py

# 특정 버전 실행
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front"
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="MIC"
```

### 다중 노드 실행

```bash
# 전방 + 후방 카메라
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front,STseg_back"

# 좌측 + 우측 카메라  
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_left,STseg_right"

# 모든 STseg 버전
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front,STseg_back,STseg_left,STseg_right"
```
### ALIAS

```bash
alias seg_front='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front'
alias seg_back='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_back'
alias seg_fb='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front, STseg_back"'
```

### Weight 

https://drive.google.com/file/d/1sM2AsjgaRy2mj13oAh1IWgMykL_rncjk/view?usp=sharing

###설치이슈

빌드안될때, setuptools 버젼 낮추기
opencv - numpy 버젼 맞추기

