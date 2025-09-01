import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image


data_dir = "./data/tas500/"
annotation_folder = "train_labels_ids/"
annot_folder = 'val_labels_ids/'
save_folder = "gt/"
additional_save_folder = './data/rugd_2x/gt/tas/'

IDs =    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20,21,22, 255]
# 24
Groups = [1, 1, 1, 4, 3, 3, 2, 2, 
          0, 3, 3, 4, 4, 4, 4, 
          4, 0, 4, 4, 4, 4, 4, 0,
          0]

ID_seq = {}
ID_group = {}
for n, label in enumerate(IDs):
    ID_seq[label] = n
    ID_group[label] = Groups[n]


# 0 -- Background :     misc.veg, sky, ego_vehicle, undefined
# 1 -- ground :         gravel, soil, asphalt
# 2 -- soft_veg :       low_grass, high_grass
# 3 -- hard_veg :       bush, forest, tree_crown, tree_trunk, 
# 4 -- obstacle :       building, fence, wall, misc_object, pole, traffic_sign, 
#                       car, bus, person, animal, sand, 


CLASSES = ("asphalt", "gravel", "soil", "sand", "bush", "forest", "low_grass", "high_grass", 
            "misc_veg", "tree_crown", "tree_trunk", "building", "fence", "wall", "car", 
            "bus", "sky", "misc.object", "pole", "traffic_sign", "person", "animal", "ego_vehicle",
            "undefined")

PALETTE = [[192,192,192], [105,105,105], [160, 82, 45 ], [244,164, 96], [60,179,113], 
            [34,139, 34], [154,205, 50 ], [0,128,  0], [0,100,  0], [0,250,154], 
            [139, 69, 19 ], [ 1, 51, 73 ], [190,153,153], [0,132,111], [ 0,  0,142], 
            [0, 60,100], [135,206,250], [128,  0,128], [153,153,153], [ 255,255,  0 ],
            [220, 20, 60],[255,182,193], [220,220,220], [0,0,0]]


def raw_to_seq(seg):
    h, w = seg.shape
    out1 = np.zeros((h, w))
    out2 = np.zeros((h, w))
    for i in IDs:
        out1[seg==i] = ID_seq[i]
        out2[seg==i] = ID_group[i]

    return out1, out2




with open(osp.join(data_dir, 'train.txt'), 'r') as r:
    i = 0
    for l in r:
        print("train: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(data_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)

        #mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
        mmcv.imwrite(out2, data_dir + save_folder + l.strip() + "_small.png")
        mmcv.imwrite(out2, additional_save_folder + l.strip() + "_small.png")

        i += 1


with open(osp.join(data_dir, 'val.txt'), 'r') as r:
    i = 0
    for l in r:
        print("val: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(data_dir + annot_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)
        
        #mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
        mmcv.imwrite(out2, data_dir + save_folder + l.strip() + "_small.png")
        mmcv.imwrite(out2, additional_save_folder + l.strip() + "_small.png")

        i += 1



# with open(osp.join(rellis_dir, 'test.txt'), 'r') as r:
#     i = 0
#     for l in r:
#         print("test: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(rellis_dir + annotation_folder + l.strip() + '.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         out1, out2 = raw_to_seq(gt_semantic_seg)

#         #mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
#         mmcv.imwrite(out2, rellis_dir + save_folder + l.strip() + "_inte.png")

#         i += 1




print("successful")