# Semantic Segmentation Package

## Environment Settings

- **Python**: 3.10
- **Numpy**: 1.24.4
- **CV2**: 4.11.0
- **ROS2**: Humble (Base)
- **PyTorch**: 2.1 (2.3)
- **MMCV**: 2.2.0
- **MMSegmentation**: 1.2.2

## Package Structure

```
semantic_segmentation/
├── config/                    # 설정 파일들
│   ├── config.yaml           # 기본 설정
│   ├── config_STseg_front.yaml   # 전방 카메라 설정
│   ├── config_STseg_back.yaml    # 후방 카메라 설정
├── launch/
│   └── semantic_segmentation.launch.py
└── semantic_segmentation/     # 소스 코드
    ├── semantic_segmentation.py
    ├── semantic_segmentation_STseg_front.py
    ├── semantic_segmentation_STseg_back.py
```

## Configuration

manage configuration in `semantic_segmentation/config/config_STseg_{camera}.yaml` 

### Configuration parameters

- **rgb_topic**: input rgb image topic
- **seg_topic**: output segmentation image topic 
- **model_config**: model config file path
- **model_weight**: model weight file path

### Default Configuration

```yaml
# Example - Front Camera
rgb_topic: /zed_front/zed_node/left/image_rect_color/compressed
seg_topic: /seg_front/left/compressed
model_config: "~/path/to/model/config"
model_weight: "~/path/to/model/weight"
```

## Topics

### Subscribe Topic
- `/zed_front/zed_node/left/image_rect_color/compressed` (default)
- It can be modified for each camera in the configuration file.

### Publish Topic
- `/seg_front/left/compressed` (default)
- It can be modified for each camera in the configuration file.


## Usage

### Single Node Launch

```bash
# default
ros2 launch semantic_segmentation semantic_segmentation.launch.py

# ST-Seg version
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front"
```

### Multi Node Launch

```bash
# Front + Rear Camera
ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front,STseg_back"

### ALIAS

```bash
alias seg_front='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front'
alias seg_back='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_back'
alias seg_fb='ros2 launch semantic_segmentation semantic_segmentation.launch.py versions:="STseg_front, STseg_back"'
```

### Weight 

SegFormer Weights:


ST-Seg Weights:

https://drive.google.com/file/d/1sM2AsjgaRy2mj13oAh1IWgMykL_rncjk/view?usp=sharing

### Possible build error

"If the build fails, try downgrading the setuptools version."
"Make sure the OpenCV and NumPy versions are compatible."

