from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Group7(CustomDataset):
    """RUGD dataset.

    """



    CLASSES = ("background", "smooth-road", "rough-road", "drivable-veg", "non_drivable-veg", "non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 0, 255, 0 ], [0,102,0],
            [ 255, 0, 0 ],[  0, 0, 128] ]


    def __init__(self, **kwargs):
        super(RUGDDataset_Group7, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group7.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth", "rough", "drivable-veg", "non_drivable-veg", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 0, 255, 0 ], [0,102,0],
            [ 255, 0, 0 ],[  0, 0, 128] ]
        # assert osp.exists(self.img_dir)
