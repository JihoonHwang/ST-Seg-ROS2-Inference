from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Group8(CustomDataset):
    """RELLIS dataset.

    """



    CLASSES = ("background", "L1", "L2", "L3", "soft_veg","hard_veg","non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

    def __init__(self, **kwargs):
        super(RELLISDataset_Group8, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group8.png',
            **kwargs)
        self.CLASSES = ("background", "L1", "L2", "L3","soft-veg", "hard-veg", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]