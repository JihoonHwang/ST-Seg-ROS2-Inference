from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TAS500Dataset(CustomDataset):
    """TAS500 dataset.


    """



    CLASSES = ("asphalt","gravel","soil","sand","bush","forest","low_grass","high_grass","misc_veg",
               "tree_crown","tree_trunk","building","fence","wall","car","bus","sky","misc_obj","pole",
               "traffic_sign","person","animal","ego","undefined")

    PALETTE = [[192,192,192],[105,105,105],[160,82,45],[244,164,96],[60,179,113],[34,139,34],[154,205,50],
               [0,128,0],[0,100,0],[0,250,154],[139,69,19],[1,51,73],[190,153,153],[0,132,111],
               [0,0,142],[0,60,100],[135,206,250],[128,0,128],[153,153,153],[255,255,0],[220,20,60,],
               [255,182,193],[220,220,220],[0,0,0]]


    def __init__(self, **kwargs):
        super(TAS500Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
        self.CLASSES = ("asphalt","gravel","soil","sand","bush","forest","low_grass","high_grass",
                        "misc_veg","tree_crown","tree_trunk","building","fence","wall","car",
                        "bus","sky","misc_obj","pole","traffic_sign","person","animal","ego","undefined")
        self.PALETTE = [[192,192,192],[105,105,105],[160,82,45],[244,164,96],[60,179,113],[34,139,34],[154,205,50],
               [0,128,0],[0,100,0],[0,250,154],[139,69,19],[1,51,73],[190,153,153],[0,132,111],
               [0,0,142],[0,60,100],[135,206,250],[128,0,128],[153,153,153],[255,255,0],[220,20,60,],
               [255,182,193],[220,220,220],[0,0,0]]