# ST-Seg ROS2 Inference Package

**Official ROS2 Inference Package of ST-Seg**

J. -H. Hwang, D. Kim, H. -S. Yoon, D. -W. Kim and S. -W. Seo, "How to Relieve Distribution Shifts in Semantic Segmentation for Off-Road Environments," in IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 4500-4507, May 2025

## Environment Settings

- **Python**: 3.10
- **Numpy**: 1.24.4
- **CV2**: 4.11.0
- **ROS2**: Humble (Base)
- **PyTorch**: 2.1 (2.3)
- **MMCV**: 2.2.0
- **MMSegmentation**: 1.2.2

**"For installing mmcv and mmseg, refer to https://github.com/open-mmlab/mmsegmentation."**

**"If you want to do a source install, the ~/library/mmcv and ~/library/mmsegmentation directories are already included, so you can proceed with the source installation."**
```
cd ~/semantic_segmentation/library/mmcv
pip install -r requirements/optional.txt
MMCV_WITHS_OPS=1 pip install -e .

cd ~/semantic_segmentation/library/mmsegmentation
pip install -v -e .
```

## Package Structure

```
semantic_segmentation/
├── config/                    # Config files
│   ├── config.yaml           # default files
│   ├── config_STseg_front.yaml   # Front camera config
│   ├── config_STseg_back.yaml    # Rear camera config
├── launch/
│   └── semantic_segmentation.launch.py
└── semantic_segmentation/     # Source files
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

- SegFormer(Baseline) Weights:
https://drive.google.com/file/d/1w602JVDqZzbVz8nKypTgp4kHAB-Ko12n/view?usp=drive_link

- ST-Seg Weights:
https://drive.google.com/file/d/1sM2AsjgaRy2mj13oAh1IWgMykL_rncjk/view?usp=sharing

### Possible build error

"If the build fails, try downgrading the setuptools version."
"Make sure the OpenCV and NumPy versions are compatible."

## CITATION

BIBTEX

@ARTICLE{10925898,
  author={Hwang, Ji-Hoon and Kim, Daeyoung and Yoon, Hyung-Suk and Kim, Dong-Wook and Seo, Seung-Woo},
  journal={IEEE Robotics and Automation Letters}, 
  title={How to Relieve Distribution Shifts in Semantic Segmentation for Off-Road Environments}, 
  year={2025},
  volume={10},
  number={5},
  pages={4500-4507},
  keywords={Training;Semantic segmentation;Robot sensing systems;Feature extraction;Navigation;Robots;Semantics;Standards;Robustness;Roads;Deep learning for visual perception;computer vision for transportation;object detection;segmentation and categorization},
  doi={10.1109/LRA.2025.3551536}}




