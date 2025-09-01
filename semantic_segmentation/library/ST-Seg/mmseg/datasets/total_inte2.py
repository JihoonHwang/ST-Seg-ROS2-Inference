from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Dataset_Inte2(CustomDataset):
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



    CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "nature_things", "obstacle")

    PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]

    def __init__(self, **kwargs):
        super(Dataset_Inte2, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_inte2.png',
            **kwargs)
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "nature_things", "obstacle")
        self.PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]
