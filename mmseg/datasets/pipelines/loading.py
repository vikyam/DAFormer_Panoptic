# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import mmcv
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image

from ..builder import PIPELINES
from .transforms import PanopticTargetGenerator
from .utils import DatasetDescriptor

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 2975,
                     'trainval': 3475,
                     'val': 500,
                     'test': 1525},
    num_classes=19,
    ignore_label=255,
)

# Add 1 void label.
_CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                            23, 24, 25, 26, 27, 28, 31, 32, 33, 0]

_CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint16)
        
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsPanopticImage(object):
    def __init__(self, ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,):
        self.num_classes = _CITYSCAPES_INFORMATION.num_classes
        self.ignore_label = _CITYSCAPES_INFORMATION.ignore_label
        self.label_dtype = np.float32
        self.thing_list = _CITYSCAPES_THING_LIST
        
    def __call__(self, results):

        if results['aux_prefix']:
            filename = osp.join(results['pan_prefix'], results['aux_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = osp.join(results['pan_prefix'],
                                results['ann_info']['seg_map'])
        # Load panoptic gt image
        pan_gt = Image.open(filename)
        pan_gt = np.array(pan_gt, dtype=np.uint32).astype('long')
        results['pan_gt'] = pan_gt
        results['seg_fields'].append('pan_gt')
        
        return results

@PIPELINES.register_module()
class LoadAnnotationsPanopticData(object):
    def __init__(self, ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,):
        self.num_classes = _CITYSCAPES_INFORMATION.num_classes
        self.ignore_label = _CITYSCAPES_INFORMATION.ignore_label
        self.label_dtype = np.float32
        self.thing_list = _CITYSCAPES_THING_LIST

        # Initialize the panoptic target generator
        self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _CITYSCAPES_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        
    def __call__(self, results):

        results['gt_pan_data'] = self.target_transform(results['pan_gt']) 
        results['gt_pan_data']['semantic'] = np.array(results['gt_pan_data']['semantic'], dtype=np.uint32).astype('long')
        results['seg_fields'].clear()
        results['seg_fields'].append('gt_pan_data') 
        return results
    
    @staticmethod
    def train_id_to_eval_id():
        return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID
    
    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])



        
        