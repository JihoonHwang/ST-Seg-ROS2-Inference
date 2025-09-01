from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HanhwaDataset(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("void","asphalt","concrete","dirt_road","gravel","lane","lane2","fence","building","pole","person",
               "soldier","vehicle","bicycle","sky","water","grass","bush","tree","mountain","etc")

    PALETTE = [ [0,0,0],[50,50,50],[70,70,70],[100,100,100],[200,200,200],[210,220,170],[210,240,180],[190,200,150],
               [153,153,102],[150,150,150],[150,40,40],[150,40,140],[40,40,180],[20,20,140],[0,204,255],
               [0,104,255],[0,150,0],[0,100,0],[100,150,0],[50,105,0],[50,30,0]]


    def __init__(self, **kwargs):
        super(HanhwaDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_orig.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("void","asphalt","concrete","dirt_road","gravel","lane","lane2","fence","building","pole","person",
               "soldier","vehicle","bicycle","sky","water","grass","bush","tree","mountain","etc")
        self.PALETTE = [[0,0,0], [50,50,50],[70,70,70],[100,100,100],[200,200,200],[210,220,170],[210,240,180],[190,200,150],
               [153,153,102],[150,150,150],[150,40,40],[150,40,140],[40,40,180],[20,20,140],[0,204,255],
               [0,104,255],[0,150,0],[0,100,0],[100,150,0],[50,105,0],[50,30,0]]
        # assert osp.exists(self.img_dir)
