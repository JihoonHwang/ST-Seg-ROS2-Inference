from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADDDataset_Group2(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("non-drivable", "drivable")

    PALETTE = [[  255, 0, 127] ,[153,204,255]]


    def __init__(self, **kwargs):
        super(ADDDataset_Group2, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_group2.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("non-drivable", "drivable")
        self.PALETTE = [[  255, 0, 127] ,[153,204,255]]
        # assert osp.exists(self.img_dir)
