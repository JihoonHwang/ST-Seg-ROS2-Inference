from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADDDataset_Group7(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("background", "ground", "soft_veg", "hard_veg","puddle", "dynamic","obstacle")

    PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255], [138,43,226],[ 255, 0, 127] ]


    def __init__(self, **kwargs):
        super(ADDDataset_Group7, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group7.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("background", "ground", "soft_veg", "hard_veg", "puddle", "dynamic", "obstacle")
        self.PALETTE = [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[138,43,226],[ 255, 0, 127] ]
        # assert osp.exists(self.img_dir)
