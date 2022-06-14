# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import json
from pandas import lreshape
import torch
from collections import OrderedDict
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmseg.utils.instance_post_processing import get_panoptic_segmentation
from mmseg.utils.semantic_post_processing import get_semantic_segmentation
from mmseg.core.utils.save_annotations import save_annotation

_CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name

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

def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results_sem = []
    results_center = []
    results_offset = []
    dataset = data_loader.dataset
    predictions_json = os.path.join(out_dir + '/panoptic', 'predictions.json')
    x = 0
    prog_bar = mmcv.ProgressBar(len(dataset))
    _CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    predictions = []
    for i, data in enumerate(data_loader):
        if i < 2000:
            with torch.no_grad():
                result, center, offset = model(return_loss=False, **data)

            if i < 0: #show or out_dir:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for img, img_meta in zip(imgs, img_metas):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir + 'semantic', img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
                    
                    # For instance segmentation result show
                    semantic_pred =  get_semantic_segmentation(torch.from_numpy(result[0]).unsqueeze(0))
                    foreground_pred = torch.zeros_like(semantic_pred)
                    for thing_class in _CITYSCAPES_THING_LIST:
                            foreground_pred[semantic_pred == thing_class] = 1
                    panoptic_pred, center_pred = get_panoptic_segmentation(torch.from_numpy(result[0]).unsqueeze(0),
                                torch.from_numpy(center[0]).unsqueeze(0),
                                torch.from_numpy(offset[0]).unsqueeze(0),
                                thing_list=_CITYSCAPES_THING_LIST,
                                label_divisor=1000,
                                stuff_area=2048,
                                void_label=(1000 * 255),
                                threshold=0.1,
                                nms_kernel=7,
                                top_k=200,
                                foreground_mask=foreground_pred)     
                    # instances = get_cityscapes_instance_format(panoptic_pred.squeeze(0).cpu().numpy(),
                    #                                                 result[0],
                    #                                                 torch.from_numpy(center[0]).squeeze(0).cpu().numpy(),
                    #                                                 label_divisor=1000,
                    #                                                 score_type="semantic")
                    # pred_txt = osp.join(out_dir + 'instance', img_meta['ori_filename'].split('.')[0] + "_pred.txt")
                    # num_instances = len(instances)
                    # if not os.path.isdir(osp.join(out_dir + 'instance',img_meta['ori_filename'].split('.')[0])):
                    #     os.makedirs(osp.join(out_dir + 'instance',img_meta['ori_filename'].split('.')[0]))
                    # with open(pred_txt, "w") as fout:
                    #     for i in range(num_instances):
                    #         pred_class = instances[i]['pred_class']
                    #         if _CITYSCAPES_TRAIN_ID_TO_EVAL_ID is not None:
                    #             pred_class = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID[pred_class]
                    #         score = instances[i]['score']
                    #         mask = instances[i]['pred_mask'].astype("uint8")
                    #         png_filename = osp.join(out_dir + 'instance', img_meta['ori_filename'].split('.')[0], img_meta['ori_filename'].split('.')[0].split('/')[1]  + "_{}_{}.png".format(i, pred_class))
                    #         save_annotation(mask, out_dir + 'instance/', img_meta['ori_filename'].split('.')[0] + '/' + img_meta['ori_filename'].split('.')[0].split('/')[-1] + "_{}_{}".format(i, pred_class),
                    #                         add_colormap=False, scale_values=True)
                    #         fout.write("{} {} {}\n".format(osp.join('mask', osp.basename(png_filename)), pred_class, score))
                    
                    # For panoptic segmentation result show
                    # Change void region.
                    if not os.path.isdir(osp.join(out_dir + 'panoptic',img_meta['ori_filename'].split('.')[0].split('/')[0])):
                        os.makedirs(osp.join(out_dir + 'panoptic',img_meta['ori_filename'].split('.')[0].split('/')[0]))
                    panoptic = panoptic_pred.squeeze(0).cpu().numpy()
                    panoptic[panoptic == 255] = 0
                    panoptic[panoptic == 255000] = 0
                    segments_info = []
                    for pan_lab in np.unique(panoptic):
                        pred_class = pan_lab // 1000
                        if _CITYSCAPES_TRAIN_ID_TO_EVAL_ID is not None:
                            pred_class = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID[pred_class]
                        
                        segments_info.append(
                            {
                                'id': int(pan_lab),
                                'category_id': int(pred_class),
                            }
                        )
                    save_annotation(id2rgb(panoptic), out_dir + 'panoptic/', img_meta['ori_filename'].split('.')[0], add_colormap=False)
                    predictions.append(
                        {
                            'image_id': '_'.join(img_meta['ori_filename'].split('.')[0].split('/')[-1].split('_')[:3]),
                            'file_name': img_meta['ori_filename'].split('.')[0].split('/')[-1] + '.png',
                            'segments_info': segments_info,
                        }
                    )

        if i < 2000:    
            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
                results_sem.extend(result)
                results_center.extend(center)
                results_offset.extend(offset)
            else:
                if efficient_test:
                    result = np2tmp(result, tmpdir='.efficient_test')
                results_sem.append(result)
                results_center.append(center)
                results_offset.append(offset)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    # with open(predictions_json, "w") as f:
    #         f.write(json.dumps(predictions))
    
    return results_sem, results_center, results_offset


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
