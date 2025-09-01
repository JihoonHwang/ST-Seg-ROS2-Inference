from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Inte(CustomDataset):
    """RELLIS dataset.
# 0 -- Background : void, sky
# 1 -- smooth : concrete, asphalt
# 2 -- rough : dirt, mud
# 3 -- soft_veg : grass
# 4 -- hard_veg : bush
# 5 -- puddle : water, puddle
# 6 -- dynamic_obstacle :  person, vehicle
# 7 -- obstacle : pole, object, log, fence, barrier, rubble, building, tree
    """



    CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")

    PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]

    def __init__(self, **kwargs):
        super(RELLISDataset_Inte, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_inte.png',
            **kwargs)
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")
        self.PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]
