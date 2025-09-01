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

CLASSES = ("road", "grass", "vegetation", "tree", "sky", "obstacle")

PALETTE = [[170,170,170], [0,255,0], [102,102,51], [0,60,0], [0,120,255], [0,0,0]]


Groups = [1,2,3,3,0,4]

# 0 -- background:          sky  
# 1 -- ground         road
# 2 -- soft_veg             grass,
# 3 -- hard_veg             vegetation, tree               
# 4 -- obstacle             obstacle 
#                           
 
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


# with open(osp.join(data_dir, 'train_Deepscene.txt'), 'r') as r:
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
#         mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_large.png")
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



with open(osp.join(data_dir, 'test_Deepscene.txt'), 'r') as r:
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
        #mmcv.imwrite(out, data_dir + new_annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        if not osp.exists(data_dir + new_annotation_folder):
            os.makedirs(data_dir + new_annotation_folder)
        mmcv.imwrite(out2, data_dir + new_annotation_folder + l.strip() + "_small.png")
        #mmcv.imwrite(out2, additional_folder + l.strip() + "_inte2.png")

        i += 1




print("successful")