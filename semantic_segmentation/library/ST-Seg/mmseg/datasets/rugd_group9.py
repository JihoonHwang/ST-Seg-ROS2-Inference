from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Group9(CustomDataset):
    """RUGD dataset.

    """



    CLASSES = ("background", "smooth-road", "rough-road", "drivable-veg", "non_drivable-veg", "tree", "non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,100,0], [0,20,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]


    def __init__(self, **kwargs):
        super(RUGDDataset_Group9, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group9.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth", "rough", "bumpy","drivable-veg", "non_drivable-veg", "tree", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,100,0], [0,20,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
        # assert osp.exists(self.img_dir)
