import argparse
import os.path as osp
import numpy as np
import mmcv
import cv2
from PIL import Image


rudg_dir = "./data/rugd_2x/"
annotation_folder = "label/"
additional_folder = "./data/rugd_2x/gt/"

CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
        "vehicle", "container/generic-object", "asphalt", "gravel", 
        "building", "mulch", "rock-bed", "log", "bicycle", "person", 
        "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")

PALETTE = [ [ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
            [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
            [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
            [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ]]

Groups = [1, 1, 2, 3, 4, 1, 0, 
          4, 4, 1, 1, 
          4, 1, 1, 4, 4, 4, 
          4, 3, 4, 4, 4, 1, 4]


# 0 -- Background:          sky
# 1 -- smooth ground:       concrete,  asphalt, dirt, sand, mulch, gravel, rock-bed, water
# 2 -- soft_veg:            grass 
# 3 -- hard_veg:            bush, tree
# 4 -- obstacle:            log, rock, person, vehicle, bicycle, pole, container/generic-object, building, sign, fence, 
#                           picnic-table, bridge
   

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
            else:
                print("unknown color, exiting...")
                exit(0)
    return out


def raw_to_seq(seg):
    h, w = seg.shape
    out = np.zeros((h, w))
    for i in range(len(Groups)):
        out[seg==i] = Groups[i]

    out[seg==255] = 0
    return out


with open(osp.join(rudg_dir, 'rugd_total.txt'), 'r') as r:
    i = 0
    for l in r:
        print("small: {}".format(i))
        # w.writelines(l[:-5] + "\n")
        # w.writelines(l.split(".")[0] + "\n")
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '_orig.png')
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        #print(gt_semantic_seg.shape)
        
        # gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        # #print(gt_semantic_seg.shape)
        # out = rgb2mask(gt_semantic_seg)
        # #print(out.shape)
        # out = out[:, :, 0]
        out = gt_semantic_seg
        #mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
        out2 = raw_to_seq(out)
        out2 = cv2.resize(out2, (1280,720), interpolation=cv2.INTER_NEAREST)
        mmcv.imwrite(out2, additional_folder + l.strip() + "_small.png")
        #mmcv.imwrite(out2, additional_folder + l.strip() + "_group6.png")

        i += 1


# with open(osp.join(rudg_dir, 'test_rugd.txt'), 'r') as r:
#     i = 0
#     for l in r:
#         print("test: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '_orig.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         #print(gt_semantic_seg.shape)
        
#         # gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
#         # #print(gt_semantic_seg.shape)
#         # out = rgb2mask(gt_semantic_seg)
#         # #print(out.shape)
#         # out = out[:, :, 0]
#         out = gt_semantic_seg
#         #mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
#         out2 = raw_to_seq(out)
#         out2 = cv2.resize(out2, (1280,720), interpolation=cv2.INTER_NEAREST)
#         mmcv.imwrite(out2, additional_folder + l.strip() + "_large.png")
#         #mmcv.imwrite(out2, additional_folder + l.strip() + "_group6.png")

        # i += 1



print("successful")