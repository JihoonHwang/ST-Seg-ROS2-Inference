from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Dataset_Dee(CustomDataset):
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



    CLASSES = ("ground", "vegetation", "background", "obstacle")

    PALETTE = [[64, 64, 64], [0, 255, 0], [0, 0, 0],[255,0,127]]

    def __init__(self, **kwargs):
        super(Dataset_Dee, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_dee2.png',
            **kwargs)
        self.CLASSES = ("ground", "vegetation", "background", "obstacle")
        self.PALETTE = [[64, 64, 64], [0, 255, 0], [0, 0, 0],[255,0,127]]