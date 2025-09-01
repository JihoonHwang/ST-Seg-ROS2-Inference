from .builder import DATASETS
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike
import warnings
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from mmseg.utils import get_root_logger
from .custom import CustomDataset
from mmcv import FileClient
from torch.utils.data import Dataset
from .pipelines import Compose
import mmcv
from mmcv.utils import print_log

def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path
    
def find_folders(root: str,
                file_client: FileClient) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        file_client.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int],
                is_valid_file: Callable, file_client: FileClient):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = file_client.join_path(root, folder_name)
        files = list(
            file_client.list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = file_client.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders
@DATASETS.register_module()
class WILDDataset_Large(CustomDataset):
    """Hanhwa dataset.

    """



    CLASSES = ("Background", "smooth_ground", "rough_ground", "bumpy_ground" ,"soft_veg", "hard_veg", "puddle", "obstacle")

    PALETTE = [[0, 0, 0], [255, 20, 147], [155, 155, 155], [255, 215, 0], [173, 255, 47], [0,139, 139], [0,0,100],[138,43,226]]


    
    
    def __init__(self, **kwargs):
        super(WILDDataset_Large, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_large.png',
            # seg_map_suffix='_labelid_novoid_255.png',
            **kwargs)
        extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif')
        self.CLASSES = ("Background", "smooth_ground", "rough_ground", "bumpy_ground" ,"soft_veg", "hard_veg", "puddle", "obstacle")
        self.PALETTE = [[0, 0, 0], [255, 20, 147], [155, 155, 155], [255, 215, 0], [173, 255, 47], [0,139, 139], [0,0,100],[138,43,226]]
        assert osp.exists(self.img_dir)
        # data_prefix = '/home/docker/workspace/mmseg/data/ImageNet'
        data_prefix = '/home/docker/workspace/mmseg/data/rugd_2x/test_v3'
        self.data_prefix=expanduser(data_prefix)
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.img_ann_file = None
        self.data_infos =self.load_wild_annotations()
       
       
        

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        file_client = FileClient.infer_client(self.file_client_args,
                                              self.data_prefix)
        classes, folder_to_idx = find_folders(self.data_prefix, file_client)
        samples, empty_classes = get_samples(
            self.data_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file,
            file_client=file_client,
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        # if self.CLASSES is not None:
        #     assert len(self.CLASSES) == len(classes), \
        #         f"The number of subfolders ({len(classes)}) doesn't match " \
        #         f'the number of specified classes ({len(self.CLASSES)}). ' \
        #         'Please check the data folder.'
        else:
            self.CLASSES = classes

        if empty_classes:
            warnings.warn(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}",
                UserWarning)

        self.folder_to_idx = folder_to_idx

        return samples
    
    def load_wild_annotations(self):
        """Load image paths and gt_labels."""
        ############## WILD ####################
        
        samples = self._find_samples()

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['wild_prefix'] = self.data_prefix
        if self.custom_classes:
            results['label_map'] = self.label_map
    
    
    # def __getitem__(self,idx):
    #     if self.test_mode:
    #         return self.prepare_test_img(idx)
    #     else:
    #         return self.
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        # print('idx',idx)
        # print('len',len(self.data_infos))
        if idx >= len(self.data_infos):
            wild_idx = idx % len(self.data_infos)
            # print('idx', idx)
            # print('wild', wild_idx)
        else:
            wild_idx = idx
        wild_info = self.data_infos[wild_idx]
        results = dict(img_info=img_info, ann_info=ann_info, wild_info=wild_info)
        self.pre_pipeline(results)
        out_dic = self.pipeline(results)
        return out_dic