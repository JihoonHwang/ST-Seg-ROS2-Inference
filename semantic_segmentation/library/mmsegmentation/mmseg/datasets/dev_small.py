from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Dev_Small(CustomDataset):
    """ADD dataset.
# 0 -- background:          sky, mountain, void
# 1    ground               asphalt_concrete, soil, dirt_road, dry_leaf, gravel, puddle
# 2 -- soft_veg             grass, high_grass, 
# 3 -- hard_veg             bush, stem_branch, tree_trunk, tree_forest
# 4 -- obstacle             rock, stone, log_root person, vehicle, dynamic_object, building_wall, 
#                           pole, guardrail_fence, etc, 
    """



    CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "obstacle")

    PALETTE = [[0, 0, 0], [155, 155, 155], [173, 255, 47], [0, 139, 139], [138,43,226]]

    def __init__(self, **kwargs):
        super(Dev_Small, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_small.png',
            **kwargs)
        self.CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "obstacle")
        self.PALETTE = [[0, 0, 0], [155, 155, 155], [173, 255, 47], [0, 139, 139], [138,43,226]]
