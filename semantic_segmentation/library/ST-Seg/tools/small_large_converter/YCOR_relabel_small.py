import argparse
import os.path as osp
import numpy as np
import mmcv
import os
# import cv2
from PIL import Image


data_dir = "./data/rugd_2x/"
annotation_folder = "gt/"
new_annotation_folder = "gt/"
additional_folder = './data/rugd_2x/gt/'

CLASSES = ("background", "high_vegetation", "traversable_grass", "smooth_trail",
           "obstacle", "sky", "rough_trail", "puddle", "non_traversable_low_vegetation")

PALETTE = [[255,255,255], [40,80,0], [128,255,0], [178,176,153], [255,0,0], [1,88,255], 
           [156,76,30], [255,0,128], [0,160,0] ]


Groups = [0, 3, 2, 1, 4, 0, 1, 1, 3]

# 0 -- background:          background, sky      
# 1 -- ground         smooth_trail    rough_trail puddle
# 2 -- soft_veg             traversable_grass
# 3 -- hard_veg             non_traversable_low_vegetation, high_vegetation   
# 4 -- obstacle             obstacle 
                        
 
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


# with open(osp.join(data_dir, 'train_YCOR.txt'), 'r') as r:
#     i = 0
#     #print("total: {}".format(len(r)))
#     for l in r:
#         print("train: {}".format(i))
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
#         if not osp.exists(data_dir + new_annotation_folder):
#             os.makedirs(data_dir + new_annotation_folder)
#         mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_dev.png")
#         #mmcv.imwrite(out2, additional_folder + l.strip() + "_inte2.png")

#         i += 1


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



with open(osp.join(data_dir, 'test_YCOR.txt'), 'r') as r:
    i = 0
    #print("total: {}".format(len(r)))
    for l in r:
        print("test: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(data_dir + annotation_folder + l.strip() + '.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        out = rgb2mask(gt_semantic_seg)
        out = out[:, :, 0]
        #mmcv.imwrite(out, data_dir + new_annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        if not osp.exists(data_dir + new_annotation_folder):
            os.makedirs(data_dir + new_annotation_folder)
        mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_small.png")
        #mmcv.imwrite(out2, additional_folder + l.strip() + "_inte2.png")

        i += 1




print("successful")