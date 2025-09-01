from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HanhwaDataset_Group6(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("Background", "Road", "Drivable_veg", "Non_veg", "Sky", "Obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 80, 0 ],
            [ 0, 0, 255 ],[  255, 0, 127] ]


    def __init__(self, **kwargs):
        super(HanhwaDataset_Group6, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group6.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("Background", "Road", "Drivable_veg", "Non_veg", "Sky", "Obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 80, 0 ],
            [ 0, 0, 255 ],[  255, 0, 127] ]
        # assert osp.exists(self.img_dir)
