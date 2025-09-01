# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import (ConcatDataset,                             RepeatDataset)
from .drive import DRIVEDataset
from .face import FaceOccludedDataset
from .hrf import HRFDataset
from .imagenets import (ImageNetSDataset, LoadImageNetSAnnotations,
                        LoadImageNetSImageFromFile)
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .loveda import LoveDADataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset

from .rellis import RELLISDataset
from .rugd import RUGDDataset
from .rellis_group6 import RELLISDataset_Group6
from .rellis_group8 import RELLISDataset_Group8
from .rugd_group6 import RUGDDataset_Group6
from .rugd_group4 import RUGDDataset_Group4
from .rugd_group9 import RUGDDataset_Group9
from .rugd_group2 import RUGDDataset_Group2
from .rellis_group4 import RELLISDataset_Group4
from .rugd_group7_ours import RUGDDataset_Group7
from .rugd_group8_ours import RUGDDataset_Group8
from .hanhwa import HanhwaDataset
from .hanhwa_group6 import HanhwaDataset_Group6
from .hanhwa_group8 import HanhwaDataset_Group8
from .hanhwa_group9 import HanhwaDataset_Group9
from .tas500 import TAS500Dataset
from .rugd_hanhwa_group9 import RUGD_Hanhwa_Dataset_Group9
from .rugd_hanhwa_group6 import RUGD_Hanhwa_Dataset_Group6
from .add100_group9 import ADDDataset_Group9
from .rugd_add100_group9 import RUGD_ADD_Dataset_Group9
from .add100_group10 import ADDDataset_Group10
from .rugd_group10 import RUGDDataset_Group10
from .add100_group2 import ADDDataset_Group2
from .add100_group6 import ADDDataset_Group6
from .add100_group7 import ADDDataset_Group7
from .add100 import ADDDataset

from .rellis_inte import RELLISDataset_Inte
from .rugd_inte import RUGDDataset_Inte
from .dev_inte import DEVDataset_Inte

from .base_cls_dataset import BaseClsDataset
from .custom_cls import CustomClsDataset
from .imagenet import ImageNet
from .uda_dataset import UDADataset
from .dev_wild_inte import DEVWILDDataset_Inte
from .total_inte2 import Dataset_Inte2
from .total_wild_inte2 import WILDDataset_Inte2
from .total_dev import Dataset_Dev
from .total_wild_dev import WILDDataset_Dev
from .total_dee import Dataset_Dee
from .dev_large import Dev_Large
from .dev_small import Dev_Small
from .total_wild_small import WILDDataset_Small
from .total_wild_large import WILDDataset_Large
from .dev_aug_small import Dev_Aug_Small
from .dev_aug_large import Dev_Aug_Large
from .demo8 import Demo8

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'FaceOccludedDataset',
    'ImageNetSDataset', 'LoadImageNetSAnnotations',
    'LoadImageNetSImageFromFile', 'RUGDDataset', 'RELLISDataset', 'RELLISDataset_Group6', 
    'RUGDDataset_Group6', 'RUGDDataset_Group4', 'RELLISDataset_Group4', 'RUGDDataset_Group7',
    'RUGDDataset_Group8', 'RUGDDataset_Group9','HanhwaDataset',
    'HanhwaDataset_Group6', 'HanhwaDataset_Group8' , 'HanhwaDataset_Group9', 'RELLISDataset_Group8'
    'TAS500Dataset', 'RUGD_Hanhwa_Dataset_Group6'
    'RUGD_Hanhwa_Dataset_Group9',
    'ADDDataset_Group9', 'ADDDataset_Group10', 'ADDDataset_Group2','ADDDataset','ADDDataset_Group6',
    'RUGD_ADD_Dataset_Group9','RUGDDataset_Group10', 'ADDDataset_Group7',
    ##
    'RELLISDataset_Inte', 'RUGDDataset_Inte', 'DEVDataset_Inte',
    
    ##
    'Dataset_Dee',
    'BaseClsDataset', 'CustomClsDataset','ImageNet', 'UDADataset', 'DEVWILDDataset_Inte', 'Dataset_Inte2', 'WILDDataset_Inte2',
    'Dataset_Dev', 'WILDDataset_Dev',
    
    ##
    'Dev_large', 'Dev_Small', 'Dev_Aug_Small', 'Dev_Aug_Large',
    'WILDDataset_Small', 'WILDDataset_Large', 'Demo8'
    
]
