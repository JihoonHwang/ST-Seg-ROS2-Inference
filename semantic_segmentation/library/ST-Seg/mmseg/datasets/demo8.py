from .builder import DATASETS
from .custom import CustomDataset
import numpy as np


@DATASETS.register_module()
class Demo8(CustomDataset):
    """ADD dataset.
# 0 -- background:          sky, mountain, void
# 1    ground               asphalt_concrete, soil, dirt_road, dry_leaf, gravel, puddle
# 2 -- soft_veg             grass, high_grass, 
# 3 -- hard_veg             bush, stem_branch, tree_trunk, tree_forest
# 4 -- obstacle             rock, stone, log_root person, vehicle, dynamic_object, building_wall, 
#                           pole, guardrail_fence, etc, 
    """



    CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "obstacle", "vehicle", "pole", "building")

    PALETTE = [
        [0, 0, 0],        # undefined - 검정
        [128, 64, 128],  # road - 어두운 회색
        [107, 142, 35],  # grass - 올리브 그린
        [152, 251, 152], # vegetation - 연한 그린
        [255, 182, 193], # animal - 연한 핑크
        [0, 0, 142],     # vehicle - 짙은 블루
        [220, 220, 0],   # pole - 밝은 옐로우
        [70, 130, 180],  # building - 스틸 블루
    ]

    def __init__(self, 
                 crop_pseudo_margins=None,
                 **kwargs):
        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')
        super(Demo8, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_demo8.png',
            **kwargs)
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [720, 1280]
        
        self.CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "obstacle", "vehicle", "pole", "building")
        self.PALETTE = [
        [0, 0, 0],        # undefined - 검정
        [128, 64, 128],  # road - 어두운 회색
        [107, 142, 35],  # grass - 올리브 그린
        [152, 251, 152], # vegetation - 연한 그린
        [255, 182, 193], # animal - 연한 핑크
        [0, 0, 142],     # vehicle - 짙은 블루
        [220, 220, 0],   # pole - 밝은 옐로우
        [70, 130, 180],  # building - 스틸 블루
    ]
    def pre_pipeline(self, results):
        super(Demo8, self).pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')