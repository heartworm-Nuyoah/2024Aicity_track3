import copy
import os
import os.path as osp

import torch
import json
import mmcv
import random
from mmaction.datasets.base import BaseDataset
from tqdm import trange
import numpy as np
from .builder import DATASETS

@DATASETS.register_module()
class FadRawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='frame_{:05}.jpg', #'img_{:05}.jpg',
                 dict_util=None,
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 ):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.clip_len = pipeline[0]['clip_len']
        self.dict_util = json.load(open(dict_util,'r'))
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class,
            power,
            dynamic_length)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames

                    video_info['total_frames'] = int(line_split[idx])
                    # video_info['offset'] =
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    video_info['label'] = label
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)

        return video_infos

    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_video_clips = len(video_infos)
        # path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        video_infos_new = []
        for i in trange(num_video_clips):#():#
            path_value = video_infos[i]['video']
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix,video_infos[i]['label'], path_value+'.mp4')
                
            video_infos[i]['frame_dir'] = path_value

            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = self.dict_util[list(video_infos[i]['label'].keys())[0]]
            video_infos_new.append(video_infos[i])
        return video_infos_new

    # def load_json_annotations(self):
    #     """Load json annotation file to get video information."""
    #     video_infos = mmcv.load(self.ann_file)
    #     num_videos = len(video_infos)
    #     path_key = 'frame_dir'
    #     video_infos_new = []
    #     for i in trange(num_videos):#(100):#
    #         path_value = video_infos[i]['video'].replace('.r13', '')
    #         path_value = path_value.replace('_mot', '.avi')
    #         if self.data_prefix is not None:
    #             path_value = osp.join(self.data_prefix, path_value)
    #         video_infos[i][path_key] = path_value
    #         if 'frame_%05d.jpg' % video_infos[i]['fid'] not in os.listdir(path_value):
    #             continue
    #         if video_infos[i]['bbox_clip'][0] == video_infos[i]['bbox_clip'][2] or video_infos[i]['bbox_clip'][1] == video_infos[i]['bbox_clip'][3]:
    #             continue
    #         if self.multi_class:
    #             assert self.num_classes is not None
    #         else:
    #             assert len(video_infos[i]['label']) == 1
    #             video_infos[i]['label'] = self.dict_util[list(video_infos[i]['label'].keys())[0]]
    #         video_infos_new.append(video_infos[i])
    #     return video_infos_new


    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['flip'] = False
        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        if self.multi_class:
            onehot[self.dict_util[results['label']]] = 1.
        results['label'] = onehot
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['flip'] = False
        # prepare tensor in getitem
        onehot = torch.zeros(self.num_classes)
        if self.multi_class:
            onehot[self.dict_util[results['label']]] = 1.
        results['label'] = onehot
        return self.pipeline(results)
