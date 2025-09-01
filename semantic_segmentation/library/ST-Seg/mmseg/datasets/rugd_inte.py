from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Inte(CustomDataset):
    """RELLIS dataset.
# 0 -- Background:          sky
# 1 -- smooth ground:       concrete,  asphalt
# 2 -- rough ground         gravel, dirt, sand, mulch, rock-bed
# 3 -- soft_veg:            grass 
# 4 -- hard_veg:            bush
# 5 -- puddle:              water
# 6 -- dynamic:             person, vehicle, bicycle
# 7 -- obstacle:            tree, pole, container/generic-object, building, log, rock, sign, fence, 
#                           picnic-table, bridge
    """



    CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")

    PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]

    def __init__(self, **kwargs):
        super(RUGDDataset_Inte, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_inte.png',
            **kwargs)
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")
        self.PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]
