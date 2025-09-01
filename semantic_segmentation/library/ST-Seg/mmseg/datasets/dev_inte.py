from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DEVDataset_Inte(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")

    PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]


    def __init__(self, **kwargs):
        super(DEVDataset_Inte, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_inte.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "soft_veg", "hard_veg", "puddle", "dynamic_objects", "obstacle")
        self.PALETTE = [[0, 0, 0], [64, 64, 64], [255, 128, 0], [0, 255, 0], [0, 100, 0], [0,0,255], [138,43,226],[255,0,127]]
        # assert osp.exists(self.img_dir)
