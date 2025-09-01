from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADDDataset_Group10(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("background", "smooth-road", "rough-road", "bumpy-road", "drivable-veg", "non_drivable-veg", "tree", "non-Nav", "obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [173,255,47],[0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]


    def __init__(self, **kwargs):
        super(ADDDataset_Group10, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group10.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "smooth-road", "rough-road", "bumpy-road", "drivable-veg", "cost_veg","non_drivable-veg", "tree", "non-Nav", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [173,255,47],[0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
        # assert osp.exists(self.img_dir)
