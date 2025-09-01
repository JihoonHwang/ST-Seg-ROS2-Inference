import argparse
import os.path as osp
import numpy as np
import mmcv
import os
# import cv2
from PIL import Image


data_dir = "./data/dev2/"
annotation_folder = "color_gt/"
new_annotation_folder = "gt/"
additional_folder = './data/rugd_2x/gt/'

CLASSES = ("void","asphalt_concrete", "soil", "dirt_road", "dry_leaf", "gravel", "person", "vehicle", "dynamic_object",
           "grass", "high_grass", "bush", "stem_branch", "tree_trunk", "tree_forest", "rock", "stone", "log_root",
           "puddle", "sky", "mountain", "building_wall", "pole", "guardrail_fence", "etc")

PALETTE = [[0, 0, 0], [255, 20, 147], [ 255, 228, 196 ],[ 155, 155, 155 ],[ 255, 215, 0 ], [ 255, 0, 0 ],
            [255, 127, 80],[ 138, 43, 226 ], [ 0, 206, 209 ], [ 189, 183, 107 ], [ 85, 107, 47 ],
            [0, 139, 139], [173, 255, 47], [91, 53, 21], [34, 139, 34], [221, 160, 221],
            [255, 105, 180], [184, 105, 35], [253, 245, 230], [0, 255, 255], [107, 142, 35],
            [255, 255, 0], [0, 0, 255], [150, 141, 207], [212, 154, 103]]


Groups = [0, 1, 1, 1, 1, 1, 4, 4, 4, 
          2, 2, 3, 3, 3, 3, 4, 4, 4, 
          1, 0, 0, 4, 4, 4, 4]

# 0 -- background:          sky, mountain, void
# 1    ground               asphalt_concrete, soil, dirt_road, dry_leaf, gravel, puddle
# 2 -- soft_veg             grass, high_grass, 
# 3 -- hard_veg             bush, stem_branch, tree_trunk, tree_forest
# 4 -- obstacle             rock, stone, log_root person, vehicle, dynamic_object, building_wall, 
#                           pole, guardrail_fence, etc, 
 
color_id = {tuple(c):i for i, c in enumerate(PALETTE)}
color_id[tuple([0, 0, 0])] = 255

def rgb2mask(img):
    # assert len(img) == 3
    h, w, c = img.shape
    out = np.ones((h, w, c)) * 255
    for i in range(h):
        for j in range(w):
            if tuple(img[i, j]) in color_id:
                out[i][j] = color_id[tuple(img[i, j])]
            #else:
                # print("unknown color, exiting...")
                # print(tuple(img[i, j]))
                # print("i.j : ",i, j)
                # exit(0)
                #out[i][j] = color_id[tuple([0,0,0])]
    return out


def raw_to_seq(seg):
    h, w = seg.shape
    out = np.zeros((h, w))
    for i in range(len(Groups)):
        out[seg==i] = Groups[i]

    out[seg==255] = 0
    return out

# with open(osp.join(data_dir, 'test_dev2.txt'), 'r') as r:
#     i = 0
#     #print("total: {}".format(len(r)))
#     for l in r:
#         print("test: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(data_dir + annotation_folder + l.strip() + '.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
#         out = rgb2mask(gt_semantic_seg)
#         out = out[:, :, 0]
#         mmcv.imwrite(out, data_dir + new_annotation_folder + l.strip()+ "_orig.png")
#         #out = gt_semantic_seg
#         out2 = raw_to_seq(out)
#         if not osp.exists(data_dir + new_annotation_folder):
#             os.makedirs(data_dir + new_annotation_folder)
#         mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_inte2.png")
#         mmcv.imwrite(out2, additional_folder + l.strip() + "_inte2.png")

#         i += 1
        
        
with open(osp.join(data_dir, 'dev2_total.txt'), 'r') as r:
    i = 0
    #print("total: {}".format(len(r)))
    for l in r:
        print("train: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(data_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        out = rgb2mask(gt_semantic_seg)
        out = out[:, :, 0]
        # out = mmcv.imresize(out, (1280, 720), interpolation="nearest")
        # if not osp.exists(data_dir + new_annotation_folder):
        #     os.makedirs(data_dir + new_annotation_folder)
        # mmcv.imwrite(out, data_dir + new_annotation_folder + l.strip()+ "_orig.png")
        # mmcv.imwrite(out, additional_folder + l.strip() + "_orig.png")
        #out = gt_semantic_seg
        out2 = raw_to_seq(out)
        if not osp.exists(data_dir + new_annotation_folder):
            os.makedirs(data_dir + new_annotation_folder)
        out2 = mmcv.imresize(out2, (1280, 720), interpolation="nearest")
        mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_small.png")
        mmcv.imwrite(out2, additional_folder + l.strip() + "_small.png")

        i += 1


# with open(osp.join(data_dir, 'test.txt'), 'r') as r:
#     i = 0
#     for l in r:
#         print("val: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(data_dir + annotation_folder + l.strip() + '.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
#         out = rgb2mask(gt_semantic_seg)
#         out = out[:, :, 0]
#         #mmcv.imwrite(out, data_dir + new_annotation_folder + l.strip()+ "_orig.png")
#         out2 = raw_to_seq(out)
#         mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_group9.png")

#         i += 1








print("successful")