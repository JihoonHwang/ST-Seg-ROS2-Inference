from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Dataset_Dev(CustomDataset):
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



    CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "puddle", "obstacle")

    PALETTE = [[0, 0, 0], [64, 64, 64], [0, 255, 0], [0, 50, 0], [0, 0, 255], [255,0,127]]

    def __init__(self, **kwargs):
        super(Dataset_Dev, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_dev.png',
            **kwargs)
        self.CLASSES = ("Background", "ground", "soft_veg", "hard_veg", "puddle", "obstacle")
        self.PALETTE = [[0, 0, 0], [64, 64, 64], [0, 255, 0], [0, 50, 0], [0, 0, 255], [255,0,127]]
