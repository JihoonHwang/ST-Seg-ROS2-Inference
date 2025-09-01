from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Group6(CustomDataset):
    """RUGD dataset.
    """



    CLASSES =  ("background", "ground", "soft_veg", "hard_veg","puddle", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[ 255, 0, 127] ]


    def __init__(self, **kwargs):
        super(RUGDDataset_Group6, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group6.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES =  ("background", "ground", "soft_veg", "hard_veg","puddle", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[ 255, 0, 127] ]
        # assert osp.exists(self.img_dir)
