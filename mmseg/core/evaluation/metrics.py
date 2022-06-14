# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmseg.datasets.pipelines.transforms import PanopticTargetGenerator
from mmseg.datasets.pipelines.loading import LoadAnnotationsPanopticData
from mmseg.utils.instance_post_processing import get_panoptic_segmentation
from mmseg.utils.semantic_post_processing import get_semantic_segmentation
# from mmseg.core.utils.save_annotations import save_annotation
# from mmseg.core.evaluation.instance import CityscapesInstanceEvaluator
# from mmseg.core.evaluation.panoptic import CityscapesPanopticEvaluator

_CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]
target_transform = PanopticTargetGenerator(255, LoadAnnotationsPanopticData.rgb2id, _CITYSCAPES_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=False,
                                                            small_instance_area=0,
                                                            small_instance_weight=1)

# instance_metric = CityscapesInstanceEvaluator(
#                 output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.INSTANCE_FOLDER),
#                 train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
#                 gt_dir=os.path.join(config.DATASET.ROOT, 'gtFine', config.DATASET.TEST_SPLIT)
#             )

# panoptic_metric = CityscapesPanopticEvaluator(
#                 output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.PANOPTIC_FOLDER),
#                 train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
#                 label_divisor=data_loader.dataset.label_divisor,
#                 void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
#                 gt_dir=config.DATASET.ROOT,
#                 split=config.DATASET.TEST_SPLIT,
#                 num_classes=data_loader.dataset.num_classes
#             )


def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))
    
    pred_label = torch.argmax(pred_label, 0)

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = label

    label = target_transform(label)
    label = label['semantic']

    train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    # t2e = train_id_to_eval_id

    # label = t2e(label)

    # conv_label = pred_label.copy()
    # for train_id, eval_id in enumerate(train_id_to_eval_id):
    #         conv_label[pred_label == train_id] = eval_id

    # pred_label = torch.tensor(conv_label)
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    # num_imgs = len(results)
    # assert len(gt_seg_maps) == num_imgs
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    # for i in range(num_imgs):
    area_intersect, area_union, area_pred_label, area_label = \
        intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index,
            label_map, reduce_zero_label)
    total_area_intersect = area_intersect
    total_area_union = area_union
    total_area_pred_label = area_pred_label
    total_area_label = area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result

def get_cityscapes_instance_format(panoptic, sem, ctr_hmp, label_divisor, score_type="semantic"):
    """
    Get Cityscapes instance segmentation format.
    Arguments:
        panoptic: A Numpy Ndarray of shape [H, W].
        sem: A Numpy Ndarray of shape [C, H, W] of raw semantic output.
        ctr_hmp: A Numpy Ndarray of shape [H, W] of raw center heatmap output.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        score_type: A string, how to calculates confidence scores for instance segmentation.
            - "semantic": average of semantic segmentation confidence within the instance mask.
            - "instance": confidence of heatmap at center point of the instance mask.
            - "both": multiply "semantic" and "instance".
    Returns:
        A List contains instance segmentation in Cityscapes format.
    """
    instances = []

    pan_labels = np.unique(panoptic)
    for pan_lab in pan_labels:
        if pan_lab % label_divisor == 0:
            # This is either stuff or ignored region.
            continue

        ins = OrderedDict()

        train_class_id = pan_lab // label_divisor
        ins['pred_class'] = train_class_id

        mask = panoptic == pan_lab
        ins['pred_mask'] = np.array(mask, dtype='uint8')

        sem_scores = sem[train_class_id, ...]
        ins_score = np.mean(sem_scores[mask])
        # mask center point
        mask_index = np.where(panoptic == pan_lab)
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        ctr_score = ctr_hmp[int(center_y), int(center_x)]

        if score_type == "semantic":
            ins['score'] = ins_score
        elif score_type == "instance":
            ins['score'] = ctr_score
        elif score_type == "both":
            ins['score'] = ins_score * ctr_score
        else:
            raise ValueError("Unknown confidence score type: {}".format(score_type))

        instances.append(ins)

    return instances

def eval_metrics(results_sem,
                 results_center,
                 results_offset,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Imet', 'Pmet']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    mask_dir = 'work_dirs/local-exp8/220530_1508_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_9bf47/mask_pred'

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results_sem, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict()
    for metric in metrics:
        if metric == 'mIoU':
            # iou = total_area_intersect / total_area_union
            # acc = total_area_intersect / total_area_label
            ret_metrics['total_area_intersect'] = total_area_intersect
            ret_metrics['total_area_union'] = total_area_union
            ret_metrics['total_area_label'] = total_area_label
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        # elif metric == 'Imet':
        #     semantic_pred = get_semantic_segmentation(results_sem.unsqueeze(0))
        #     foreground_pred = torch.zeros_like(semantic_pred)
        #     panoptic_pred, center_pred = get_panoptic_segmentation(semantic_pred,
        #                 results_center.unsqueeze(0),
        #                 results_offset.unsqueeze(0),
        #                 thing_list=_CITYSCAPES_THING_LIST,
        #                 label_divisor=1000,
        #                 stuff_area=2048,
        #                 void_label=(1000 * 255),
        #                 threshold=0.1,
        #                 nms_kernel=7,
        #                 top_k=200,
        #                 foreground_mask=foreground_pred)
        #     instances = get_cityscapes_instance_format(panoptic_pred,
        #                                                        results_sem.cpu().numpy(),
        #                                                        results_center.squeeze(0).cpu().numpy(),
        #                                                        label_divisor=1000,
        #                                                        score_type="semantic")
        #     num_instances = len(instances)
        #     for i in range(num_instances):
        #         pred_class = instances[i]['pred_class']
        #         score = instances[i]['score']
        #         mask = instances[i]['pred_mask'].astype("uint8")
        #         ret_metrics[pred_class] = score
        #         save_annotation(mask, mask_dir, image_filename + "_{}_{}".format(i, pred_class),
        #                         add_colormap=False, scale_values=True)


    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
