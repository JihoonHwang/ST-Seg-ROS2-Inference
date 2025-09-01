from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Dev_Large(CustomDataset):
    """ADD dataset.
# 0 -- background:          sky, mountain, void
# 1    smooth_ground        asphalt_concrete, 
# 2 -- rough_ground         soil, dirt_road, dry_leaf, 
# 3 -- bumpy ground         gravel,
# 4 -- soft_veg             grass, high_grass, 
# 5 -- hard_veg             bush, stem_branch, tree_trunk, tree_forest
# 6 -- puddle:              puddle,
# 7 -- obstacle             rock, stone, log_root person, vehicle, dynamic_object, building_wall, 
#                           pole, guardrail_fence, etc
    """



    CLASSES = ("Background", "smooth_ground", "rough_ground", "bumpy_ground" ,"soft_veg", "hard_veg", "puddle", "obstacle")

    PALETTE = [[0, 0, 0], [255, 20, 147], [155, 155, 155], [255, 215, 0], [173, 255, 47], [0,139, 139], [0,0,100],[138,43,226]]

    def __init__(self, **kwargs):
        super(Dev_Large, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_large.png',
            **kwargs)
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "bumpy_ground" ,"soft_veg", "hard_veg", "puddle", "obstacle")
        self.PALETTE = [[0, 0, 0], [255, 20, 147], [155, 155, 155], [255, 215, 0], [173, 255, 47], [0,139, 139], [0,0,100],[138,43,226]]
