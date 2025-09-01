from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RUGDDataset_Group10(CustomDataset):
    """RUGD dataset.

    """



    CLASSES = ("background", "smooth-road", "rough-road","bumpy", "drivable-veg", "cost-veg","non_drivable-veg", "tree", "non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ],[173,255,47], [0,100,0], [0,20,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]


    def __init__(self, **kwargs):
        super(RUGDDataset_Group10, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_group10.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth", "rough", "bumpy","drivable-veg","cost-veg", "non_drivable-veg", "tree", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ],[173,255,47], [0,100,0], [0,20,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
        # assert osp.exists(self.img_dir)
