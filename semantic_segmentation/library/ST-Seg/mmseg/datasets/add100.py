from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADDDataset(CustomDataset):
    """Hanhwa dataset.

    """
    # [210,210,210] - masked class
    # 



    CLASSES = ("void","asphalt_concrete", "soil", "dirt_road", "dry_leaf", "gravel", "person", "vehicle", "dynamic_object",
               "grass", "high_grass", "bush", "stem_branch", "tree_trunk", "tree_forest", "rock", "stone", "log_root",
               "puddle", "sky", "mountain", "building_wall", "pole", "guardrail_fence", "etc")
    PALETTE = [[0,0,0],[255, 20, 147], [ 255, 228, 196 ],[ 155, 155, 155 ],[ 255, 215, 0 ], [ 255, 0, 0 ],
               [255, 127, 80],[ 138, 43, 226 ], [ 0, 206, 209 ], [ 189, 183, 107 ], [ 85, 107, 47 ],
               [0, 139, 139], [173, 255, 47], [91, 53, 21], [34, 139, 34], [221, 160, 221],
               [255, 105, 180], [184, 105, 35], [253, 245, 230], [0, 255, 255], [107, 142, 35],
               [255, 255, 0], [0, 0, 255], [150, 141, 207], [212, 154, 103]]


    def __init__(self, **kwargs):
        super(ADDDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_orig.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        self.CLASSES = ("void","asphalt_concrete", "soil", "dirt_road", "dry_leaf", "gravel", "person", "vehicle", "dynamic_object",
               "grass", "high_grass", "bush", "stem_branch", "tree_trunk", "tree_forest", "rock", "stone", "log_root",
               "puddle", "sky", "mountain", "building_wall", "pole", "guardrail_fence", "etc")
        self.PALETTE = [[0,0,0],[255, 20, 147], [ 255, 228, 196 ],[ 155, 155, 155 ],[ 255, 215, 0 ], [ 255, 0, 0 ],
               [255, 127, 80],[ 138, 43, 226 ], [ 0, 206, 209 ], [ 189, 183, 107 ], [ 85, 107, 47 ],
               [0, 139, 139], [173, 255, 47], [91, 53, 21], [34, 139, 34], [221, 160, 221],
               [255, 105, 180], [184, 105, 35], [253, 245, 230], [0, 255, 255], [107, 142, 35],
               [255, 255, 0], [0, 0, 255], [150, 141, 207], [212, 154, 103]]
        # assert osp.exists(self.img_dir)
