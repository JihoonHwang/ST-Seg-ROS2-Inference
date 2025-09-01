# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

def rellis_group_classes():
    """rellis group class names for external use."""
    return [ "background", "L1", "L2", "L3", "non-Nav", "obstacle"]

def rellis_group8_classes():
    return ["background", "L1", "L2", "L3", "soft_veg", "hard_veg", "non-Nav", "obstacle"]
def rugd_group6_classes():
    """rugd group class names for external use."""
    return [ "background", "ground", "soft_veg", "hard_veg","puddle", "obstacle"]
def group5_classes():
    return ["background", "ground", "soft_veg", "hard_veg", "obstacle"]

def rugd_group7_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough", "non-veg", "dri-veg", "Non-Nav", "obstacle"]

def rugd_group8_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "non-veg", "dri-veg", "Non-Nav", "obstacle"]

def rugd_group9_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "dri-veg", "non-veg", "tree", "Non-Nav", "obstacle"]

def rugd_group10_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "dri-veg", "cost-veg","non-veg", "tree", "Non-Nav", "obstacle"]

def rugd_group4_classes():
    """rugd group class names for external use."""
    return [ "background", "L1", "Non-Nav", "obstacle"]

def rugd_hanhwa_group9_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "dri-veg", "non-veg", "tree", "Non-Nav", "obstacle"]

def add_group9_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "dri-veg", "non-veg", "tree", "Non-Nav", "obstacle"]

def add_group2_classes():
    """rugd group class names for external use."""
    return [ "non-drivable", "drivable"]
def add_group6_classes():
    return ["background", "ground", "soft_veg", "hard_veg", "puddle", "obstacle"]
def add_texture_classes():
    return ["asphalt_concrete", "soil", "dirt_road", "dry_leaf", "gravel", "person", "vehicle", "dynamic_object",
               "grass", "high_grass", "bush", "stem_branch", "tree_trunk", "tree_forest", "rock", "stone", "log_root",
               "puddle", "sky", "mountain", "building_wall", "pole", "guardrail_fence", "etc"]
def add_object_classes():
    return ["asphalt_concrete", "soil", "dirt_road", "dry_leaf", "gravel", "person", "vehicle", "dynamic_object",
               "grass", "high_grass", "bush", "stem_branch", "tree_trunk", "tree_forest", "rock", "stone", "log_root",
               "puddle", "sky", "mountain", "building_wall", "pole", "guardrail_fence", "etc"]

def add_group10_classes():
    """rugd group class names for external use."""
    return [ "background", "smooth", "rough","bumpy", "dri-veg", "cost-veg","non-veg", "tree", "Non-Nav", "obstacle"]

def rellis_classes():
    """rellis class names for external use."""
    return [
        "void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
        "object", "asphalt", "building", "log", "person", "fence", "bush", 
        "concrete", "barrier", "puddle", "mud", "rubble"
    ]

def rugd_classes():
    """rugd class names for external use."""
    return [
        "dirt", "sand", "grass", "tree", "pole", "water", "sky", 
        "vehicle", "container/generic-object", "asphalt", "gravel", 
        "building", "mulch", "rock-bed", "log", "bicycle", "person", 
        "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table"
    ]
    
def hanhwa_classes():
    return ["void","asphalt","concrete","dirt_road","gravel","lane","lane2","fence","building","pole","person",
            "soldier","vehicle","bicycle","sky","water","grass","bush","tree","mountain","etc"]
    
def hanhwa_group6_classes():
    return ["background", "ground", "soft_veg", "hard_veg","puddle", "obstacle"]

def hanhwa_group8_classes():
    return ["background", "smooth-road", "rough-road", "bumpy-road","drivable-veg", "non_drivable-veg", "non-Nav", "obstacle"]
    
def hanhwa_group8_classes():
    return ["background", "smooth-road", "rough-road", "bumpy-road","drivable-veg", "non_drivable-veg", "tree", "non-Nav", "obstacle"]

def tas500_classes():
    return ["asphalt","gravel","soil","sand","bush","forest","low_grass","high_grass","misc_veg",
               "tree_crown","tree_trunk","building","fence","wall","car","bus","sky","misc_obj","pole",
               "traffic_sign","person","animal","ego","undefined"]
    
def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def ade_classes():
    """ADE20K class names for external use."""
    return [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    ]


def voc_classes():
    """Pascal VOC class names for external use."""
    return [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]

def rellis_palette():
    """rellis palette for external use."""
    return [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]



def rugd_group6_palette():
    """rugd group palette for external use."""
    return  [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[ 255, 0, 127] ]
    
def group5_palette():
    """rugd group palette for external use."""
    return  [[ 0, 0, 0 ], [ 128,128,128 ],[ 0, 255, 0 ],[ 0, 80, 0 ], [ 255, 0, 127] ]

def rugd_group7_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 0, 255, 0 ], [0,102,0],
            [ 255, 0, 0 ],[  0, 0, 128] ]
    
def rugd_group8_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

def rugd_group9_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

def rugd_group10_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [173,255,47],[0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

def rugd_group4_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 0,128,0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]
    
def rugd_hanhwa_group9_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
    
    
    ############ ADD EVAULATION #################
    
def add_texture_palette():
    return [[210,210,210],[255, 20, 147], [ 255, 228, 196 ],[ 155, 155, 155 ],[ 255, 215, 0 ], [ 255, 0, 0 ],
               [210,210,210],[210,210,210], [210,210,210], [ 189, 183, 107 ], [ 85, 107, 47 ],
               [0, 139, 139], [210,210,210],[210,210,210], [34, 139, 34], [210,210,210],
               [255, 105, 180], [184, 105, 35], [253, 245, 230], [210,210,210], [210,210,210],
               [210,210,210], [210,210,210], [210,210,210], [210,210,210]]
    
def add_object_palette():
    return [[255,255,255],[255,255,255], [255,255,255],[255,255,255],[255,255,255], [255,255,255],
               [255, 127, 80],[ 138, 43, 226 ], [ 0, 206, 209 ], [255,255,255], [255,255,255],
               [255,255,255], [173, 255, 47], [91, 53, 21], [255,255,255], [221, 160, 221],
               [255,255,255], [184,105,35], [255,255,255], [255,255,255], [255,255,255],
               [255, 255, 0], [0, 0, 255], [150, 141, 207], [255,255,255]]
def add_drivable_palette():
    return [[255,0,127], [153,204,255], [153,204,255], [153,204,255], [153,204,255], [153,204,255], [  255, 0, 127], [  255, 0, 127], [  255, 0, 127],
            [153,204,255], [153,204,255], [  255, 0, 127],[  255, 0, 127],[  255, 0, 127],[  255, 0, 127],[  255, 0, 127],
            [  255, 0, 127],[  255, 0, 127],[  255, 0, 127],[  255, 0, 127],[  255, 0, 127],[  255, 0, 127], [  255, 0, 127],
            [  255, 0, 127], [  255, 0, 127]]
    ############################################
def add_group9_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
def add_group2_palette():
    """rugd group palette for external use."""
    return [[  255, 0, 127], [153,204,255] ]

def add_group10_palette():
    """rugd group palette for external use."""
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [173,255,47],[0,150,0], [0,50,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
def add_group6_palette():
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[  255, 0, 127] ]
    
def rellis_group_palette():
    """rellis group palette for external use."""
    return [[ 0, 0, 0 ], [ 0,128,0 ],[ 255, 255, 0 ],[ 255, 128, 0 ],
            [ 255, 0, 0 ],[  0, 0, 128] ]

def rellis_group8_palette():
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

def rugd_palette():
    """rugd palette for external use."""
    return [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
            [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
            [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
            [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]
    
def hanhwa_palette():
    return [ [0,0,0],[50,50,50],[70,70,70],[100,100,100],[200,200,200],[210,220,170],[210,240,180],[190,200,150],
               [153,153,102],[150,150,150],[150,40,40],[150,40,140],[40,40,180],[20,20,140],[0,204,255],
               [0,104,255],[0,150,0],[0,100,0],[100,150,0],[50,105,0],[50,30,0]]
    
def hanhwa_group6_palette():
    return  [[ 0, 0, 0 ], [ 64,64,64 ],[ 0, 255, 0 ],[ 0, 50, 0 ], [0,0,255],[ 255, 0, 127] ]
    
def hanhwa_group8_palette():
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,80,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]

def hanhwa_group9_palette():
    return [[ 0, 0, 0 ], [ 64,64,64 ],[ 255, 128, 0 ],[153,204,255],[ 0, 255, 0 ], [0,100,0], [0,20,0],
            [ 0, 0, 255 ],[  255, 0, 127] ]
     
def tas500_palette():
    return [[192,192,192],[105,105,105],[160,82,45],[244,164,96],[60,179,113],[34,139,34],[154,205,50],
               [0,128,0],[0,100,0],[0,250,154],[139,69,19],[1,51,73],[190,153,153],[0,132,111],
               [0,0,142],[0,60,100],[135,206,250],[128,0,128],[153,153,153],[255,255,0],[220,20,60,],
               [255,182,193],[220,220,220],[0,0,0]]

def cocostuff_classes():
    """CocoStuff class names for external use."""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
        'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
        'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
        'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood'
    ]


def loveda_classes():
    """LoveDA class names for external use."""
    return [
        'background', 'building', 'road', 'water', 'barren', 'forest',
        'agricultural'
    ]

def demo_classes():
    return [
        "road", "sidewalk", "terrain", "grass", "vegetation", "water", 
               "sky", "building", "wall", "pole", "vehicle", "human", "animal",
               "obstacle", "traffic_light", "traffic_cone", "snow", "undefined"
    ]

def demo_palette():
    return [
        [128, 64, 128],  # road - 어두운 회색
        [244, 35, 232],  # sidewalk - 밝은 핑크
        [70, 70, 70],    # terrain - 중간 회색
        [0, 255, 0],     # obstacle - 초록
        [152, 251, 152], # vegetation - 연한 그린
        [0, 170, 255],   # water - 밝은 블루
        [135, 206, 235], # sky - 하늘색
        [70, 130, 180],  # building - 스틸 블루
        [153, 153, 153], # wall - 밝은 회색
        [220, 220, 0],   # pole - 밝은 옐로우
        [0, 0, 142],     # vehicle - 짙은 블루
        [220, 20, 60],   # human - 빨강
        [255, 182, 193], # animal - 연한 핑크
        [107, 142, 35],  #  - 올리브 그린
        [255, 69, 0],    # trafic_light - 오렌지 레드
        [255, 240, 0],   # traffic_cone - 옐로우
        [255, 250, 250], # snow - 눈처럼 흰색
        [0, 0, 0]        # undefined - 검정
    ]
    
def demo8_classes():
    return [
        "Background", "ground", "soft_veg", "hard_veg", "obstacle", "vehicle", "pole", "building"
    ]
    
def demo8_palette():
    return [
        [0, 0, 0],        # undefined - 검정
        [128, 64, 128],  # road - 어두운 회색
        [107, 142, 35],  # grass - 올리브 그린
        [152, 251, 152], # vegetation - 연한 그린
        [255, 182, 193], # animal - 연한 핑크
        [0, 0, 142],     # vehicle - 짙은 블루
        [220, 220, 0],   # pole - 밝은 옐로우
        [70, 130, 180],  # building - 스틸 블루
    ]
    
def demo9_classes():
    return [
        "Background", "ground", "soft_veg", "hard_veg", "obstacle", "vehicle", "person", "building", "pole"
    ]
    
def demo9_palette():
    return [
        [0, 0, 0],        # undefined - 검정
        [128, 64, 128],  # road - 어두운 회색
        [107, 142, 35],  # grass - 올리브 그린
        [152, 251, 152], # vegetation - 연한 그린
        [128, 0, 128], # animal - 연한 핑크
        [0, 0, 142],     # vehicle - 짙은 블루
        [220, 220, 0],   # pole - 밝은 옐로우
        [70, 130, 180],  # building - 스틸 블루
        [220, 20, 60],
    ]


def potsdam_classes():
    """Potsdam class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]


def vaihingen_classes():
    """Vaihingen class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]

def group8_classes():
    return ["Background", "smooth_ground", "rough_ground", "bumpy_ground" ,"soft_veg", "hard_veg", "puddle", "obstacle"]
    
    
def group8_palette():
    return [[0, 0, 0], [155, 155, 155], [155, 155, 155], [155, 155, 155], [173, 255, 47], [0,139, 139], [155, 155, 155],[138,43,226]]

def isaid_classes():
    """iSAID class names for external use."""
    return [
        'background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court',
        'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
        'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout',
        'Soccer_ball_field', 'plane', 'Harbor'
    ]


def stare_classes():
    """stare class names for external use."""
    return ['background', 'vessel']


def occludedface_classes():
    """occludedface class names for external use."""
    return ['background', 'face']


def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]


def ade_palette():
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


def voc_palette():
    """Pascal VOC palette for external use."""
    return [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def cocostuff_palette():
    """CocoStuff palette for external use."""
    return [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0],
            [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0],
            [192, 128, 32], [128, 96, 128], [0, 0, 128], [64, 0, 32],
            [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
            [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64],
            [192, 0, 32], [128, 96, 0], [128, 0, 192], [0, 128, 32],
            [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
            [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64],
            [128, 128, 32], [192, 32, 128], [0, 64, 192], [0, 0, 32],
            [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128],
            [128, 192, 192], [0, 0, 160], [192, 160, 128], [128, 192, 0],
            [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
            [64, 160, 0], [0, 64, 0], [192, 128, 224], [64, 32, 0],
            [0, 192, 128], [64, 128, 224], [192, 160, 0], [0, 192, 0],
            [192, 128, 96], [192, 96, 128], [0, 64, 128], [64, 0, 96],
            [64, 224, 128], [128, 64, 0], [192, 0, 224], [64, 96, 128],
            [128, 192, 128], [64, 0, 224], [192, 224, 128], [128, 192, 64],
            [192, 0, 96], [192, 96, 0], [128, 64, 192], [0, 128, 96],
            [0, 224, 0], [64, 64, 64], [128, 128, 224], [0, 96, 0],
            [64, 192, 192], [0, 128, 224], [128, 224, 0], [64, 192, 64],
            [128, 128, 96], [128, 32, 128], [64, 0, 192], [0, 64, 96],
            [0, 160, 128], [192, 0, 64], [128, 64, 224], [0, 32, 128],
            [192, 128, 192], [0, 64, 224], [128, 160, 128], [192, 128, 0],
            [128, 64, 32], [128, 32, 64], [192, 0, 128], [64, 192, 32],
            [0, 160, 64], [64, 0, 0], [192, 192, 160], [0, 32, 64],
            [64, 128, 128], [64, 192, 160], [128, 160, 64], [64, 128, 0],
            [192, 192, 32], [128, 96, 192], [64, 0, 128], [64, 64, 32],
            [0, 224, 192], [192, 0, 0], [192, 64, 160], [0, 96, 192],
            [192, 128, 128], [64, 64, 160], [128, 224, 192], [192, 128, 64],
            [192, 64, 32], [128, 96, 64], [192, 0, 192], [0, 192, 32],
            [64, 224, 64], [64, 0, 64], [128, 192, 160], [64, 96, 64],
            [64, 128, 192], [0, 192, 160], [192, 224, 64], [64, 128, 64],
            [128, 192, 32], [192, 32, 192], [64, 64, 192], [0, 64, 32],
            [64, 160, 192], [192, 64, 64], [128, 64, 160], [64, 32, 192],
            [192, 192, 192], [0, 64, 160], [192, 160, 192], [192, 192, 0],
            [128, 64, 96], [192, 32, 64], [192, 64, 128], [64, 192, 96],
            [64, 160, 64], [64, 64, 0]]


def loveda_palette():
    """LoveDA palette for external use."""
    return [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def potsdam_palette():
    """Potsdam palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def vaihingen_palette():
    """Vaihingen palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def isaid_palette():
    """iSAID palette for external use."""
    return [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
            [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127,
                                                       127], [0, 0, 127],
            [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191],
            [0, 127, 255], [0, 100, 155]]


def stare_palette():
    """STARE palette for external use."""
    return [[120, 120, 120], [6, 230, 230]]


def occludedface_palette():
    """occludedface palette for external use."""
    return [[0, 0, 0], [128, 0, 0]]


dataset_aliases = {
    'cityscapes': ['cityscapes'],
    'ade': ['ade', 'ade20k'],
    'voc': ['voc', 'pascal_voc', 'voc12', 'voc12aug'],
    'rugd': ['rugd'],
    'rugd_group6': ['rugd_group6'],
    'rugd_group7': ['rugd_group7'],
    'group5' : ['group5'],
    'group8' : ['group8'],
    'rugd_group8': ['rugd_group8'],
    'rugd_group9': ['rugd_group9'],
    'rellis_group': ['rellis_group'],
    'rellis_group8' : ['rellis_group8'],
    'rugd_group4': ['rugd_group4'],
    'rugd_group2' : ['rugd_group2'],
    'rugd_hanhwa_group9': ['rugd_hanhwa_group9'],
    'add_group9' : ['add_group9'],
    'add_group10' : ['add_group10'],
    'add_group6' : ['add_group6'],
    'add_group2' : ['add_group2'],
    'add_texture' : ['add_texture'],
    'add_object' : ['add_object'],
    'add_drivable': ['add_drivable'],
    'hanhwa' : ['hanhwa'],
    'hanhwa_group6': ['hanhwa_group6'],
    'hanhwa_group8': ['hanhwa_group8'],
    'hanhwa_group9': ['hanhwa_group9'],
    'tas500' : ['tas500'],
    'demo' : ['demo'],
    'demo8' : ['demo8'],
    'demo9' : ['demo9'],
    'loveda': ['loveda'],
    'potsdam': ['potsdam'],
    'vaihingen': ['vaihingen'],
    'cocostuff': [
        'cocostuff', 'cocostuff10k', 'cocostuff164k', 'coco-stuff',
        'coco-stuff10k', 'coco-stuff164k', 'coco_stuff', 'coco_stuff10k',
        'coco_stuff164k'
    ],
    'isaid': ['isaid', 'iSAID'],
    'stare': ['stare', 'STARE'],
    'occludedface': ['occludedface']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
