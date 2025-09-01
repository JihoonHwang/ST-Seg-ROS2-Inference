from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HanhwaDataset_Group8(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("background", "smooth-road", "rough-road", "bumpy-road", "drivable-veg", "non_drivable-veg", "non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]


    def __init__(self, **kwargs):
        super(HanhwaDataset_Group8, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group8.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth-road", "rough-road", "bumpy-road", "drivable-veg", "non_drivable-veg", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
        # assert osp.exists(self.img_dir)
