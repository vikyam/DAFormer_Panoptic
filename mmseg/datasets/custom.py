# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import glob
import contextlib
import torch
import io
import numpy as np
from fvcore.common.file_io import PathManager
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from PIL import Image
import json

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 aux_dir=None,
                 dict_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.aux_dir = aux_dir
        self.dict_dir = dict_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir, self.aux_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, aux_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['pan_prefix'] = self.ann_dir 
        results['aux_prefix'] = self.aux_dir
        results['dict_dir'] = self.dict_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

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
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_seg_maps(self, index, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        seg_map = osp.join(self.ann_dir, self.img_infos[index]['ann']['seg_map'])
        if efficient_test:
            gt_seg_map = seg_map
        else:
            gt_seg_map = Image.open(seg_map)
            gt_seg_map = np.asarray(gt_seg_map, dtype=np.float32)
        return gt_seg_map

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results_sem,
                 results_center,
                 results_offset,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        
        
        num_img = len(results_sem)
        print(num_img)
        ret_metrics_areas = OrderedDict()
        ret_metrics = OrderedDict()
        ret_metrics_areas['total_area_intersect'] = 0
        ret_metrics_areas['total_area_union'] = 0
        ret_metrics_areas['total_area_label'] = 0
        for i in range(num_img):
            gt_seg_map = self.get_gt_seg_maps(i, efficient_test)
            if self.CLASSES is None:
                num_classes = len(
                    reduce(np.union1d, [np.unique(_) for _ in gt_seg_map]))
            else:
                num_classes = len(self.CLASSES)
            ret_metric = eval_metrics(
                results_sem[i], 
                results_center[i],
                results_offset[i],
                gt_seg_map,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
            ret_metrics_areas['total_area_intersect'] += ret_metric['total_area_intersect']
            ret_metrics_areas['total_area_union'] += ret_metric['total_area_union']
            ret_metrics_areas['total_area_label'] += ret_metric['total_area_label']
        
        ret_metrics['IoU'] =  ret_metrics_areas['total_area_intersect'] / ret_metrics_areas['total_area_union']
        ret_metrics['Acc'] = ret_metrics_areas['total_area_intersect'] / ret_metrics_areas['total_area_label']

        # # Instance metric evaluation
        # import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval_inst
        # # set some global states in cityscapes evaluation API, before evaluating
        # output_dir = 'work_dirs/local-exp8/220609_1547_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_afe82/preds/instance'
        # cityscapes_eval_inst.args.predictionPath = os.path.abspath(output_dir)
        # cityscapes_eval_inst.args.predictionWalk = None
        # cityscapes_eval_inst.args.JSONOutput = False
        # cityscapes_eval_inst.args.colorized = False

        # # These lines are adopted from
        # # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        # gt_dir = PathManager.get_local_path('/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/gtFine_panoptic')
        # cityscapes_eval_inst.args.gtInstancesFile = os.path.join(gt_dir, "cityscapes_panoptic_val.json")
        # groundTruthImgList = glob.glob(os.path.join(gt_dir, 'cityscapes_panoptic_val', "*", "*_gtFine_panoptic.png"))
        
        # groundTruthImgList = []
        # predictionImgList = []
        # for gt in groundTruthImgList:
        #     predictionImgList.append(cityscapes_eval_inst.getPrediction(gt, cityscapes_eval_inst.args))
        # results = cityscapes_eval_inst.evaluateImgLists(
        #     predictionImgList, groundTruthImgList, cityscapes_eval_inst.args
        # )["averages"]
        
        # # ret_metrics["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}

        # # Panoptic metric evaluation
        # import cityscapesscripts.evaluation.evalPanopticSemanticLabeling as cityscapes_eval_panoptic
        # gt_dir = '/srv/beegfs02/scratch/uda2022/data/panoptic_datasets_april_2022/data/cityscapes/'
        # output_dir = 'work_dirs/local-exp8/220609_1547_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_afe82/preds/panoptic'
        # gt_json_file = os.path.join(gt_dir, 'gtFine_panoptic', 'cityscapes_panoptic_val.json')
        # gt_folder = os.path.join(gt_dir, 'gtFine_panoptic', 'cityscapes_panoptic_val')
        # pred_json_file = os.path.join(output_dir, 'predictions.json')
        # pred_folder = os.path.join(output_dir)
        # resultsFile = os.path.join(output_dir, 'resultPanopticSemanticLabeling.json')

        # # with open(gt_json_file, "r") as f:
        # #     json_data = json.load(f)
        # # with open(pred_json_file, "r") as f:
        # #     json_data["annotations"] = json.load(f)
        # # with PathManager.open(pred_json_file, "w") as f:
        # #     f.write(json.dumps(json_data))

        # with contextlib.redirect_stdout(io.StringIO()):
        #     results_panoptic = cityscapes_eval_panoptic.evaluatePanoptic(gt_json_file, gt_folder, pred_json_file, pred_folder, resultsFile)
        
        # pq_met = np.zeros((num_classes, ), dtype=np.float)
        # sq_met = np.zeros((num_classes, ), dtype=np.float)
        # rq_met = np.zeros((num_classes, ), dtype=np.float)
        # i = 0
        # for res in results_panoptic['per_class']:
        #     pq_met[i] = results_panoptic['per_class'][res]['pq']
        #     sq_met[i] = results_panoptic['per_class'][res]['sq']
        #     rq_met[i] = results_panoptic['per_class'][res]['rq']
        #     i += 1

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100 * 19 / 16, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # ret_metrics_summary['Pq'] = np.round(results_panoptic['All']['pq'] * 100 * 19 / 16, 2)
        # ret_metrics_summary['Sq'] = np.round(results_panoptic['All']['sq'] * 100 * 19 / 16, 2)
        # ret_metrics_summary['Rq'] = np.round(results_panoptic['All']['rq'] * 100 * 19 / 16, 2)

        # ret_metrics_summary['_thing_pq'] = np.round(results_panoptic['Things']['pq'] * 100, 2)
        # ret_metrics_summary['_thing_sq'] = np.round(results_panoptic['Things']['sq'] * 100, 2)
        # ret_metrics_summary['_thing_rq'] = np.round(results_panoptic['Things']['rq'] * 100, 2)

        # ret_metrics_summary['_stuff_pq'] = np.round(results_panoptic['Stuff']['pq'] * 100, 2)
        # ret_metrics_summary['_stuff_sq'] = np.round(results_panoptic['Stuff']['sq'] * 100, 2)
        # ret_metrics_summary['_stuff_rq'] = np.round(results_panoptic['Stuff']['rq'] * 100, 2)

        # ret_metrics['pq'] = pq_met
        # ret_metrics['sq'] = sq_met
        # ret_metrics['rq'] = rq_met

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results_sem, str):
            for file_name in results_sem:
                os.remove(file_name)
        return eval_results
