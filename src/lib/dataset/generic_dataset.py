from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict

import pycocotools.coco as coco
import torch
import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy

class ClassBalancedDataset(object):
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)

class GenericDataset(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    rest_focal_length = 1200
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]
    edges = [[0, 1], [0, 2], [1, 3], [2, 4],
             [4, 6], [3, 5], [5, 6],
             [5, 7], [7, 9], [6, 8], [8, 10],
             [6, 12], [5, 11], [11, 12],
             [12, 14], [14, 16], [11, 13], [13, 15]]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    ignore_val = 1
    nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4],
                          4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}

    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        super(GenericDataset, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)

        if ann_path is not None and img_dir is not None:
            print('==> initializing {} data from {}, \n images from {} ...'.format(
                split, ann_path, img_dir))
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()

            if opt.tracking:
                if not ('videos' in self.coco.dataset):
                    self.fake_video_data()
                print('Creating video index!')
                self.video_to_images = defaultdict(list)
                for image in self.coco.dataset['images']:
                    self.video_to_images[image['video_id']].append(image)

            self.img_dir = img_dir

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        coco = self.coco
        img_id = self.images[idx]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        return [ann['category_id'] for ann in ann_info]

    def __getitem__(self, index):
        opt = self.opt
        img, anns, img_info, img_path = self._load_data(index) # img:tuple(1080,1920,3) anns:list

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
            else np.array([img.shape[1], img.shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        if self.split == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            if np.random.random() < opt.flip:
                flipped = 1
                img = img[:, ::-1, :]
                anns = self._flip_anns(anns, width)

        trans_input = get_affine_transform(
            c, s, rot, [opt.input_w, opt.input_h])
        trans_output = get_affine_transform(
            c, s, rot, [opt.output_w, opt.output_h])
        inp = self._get_input(img, trans_input) # (3,608,992) /255, minus mean
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        pre_cts, track_ids = None, None
        if opt.tracking:
            pre_image, pre_anns, frame_dist = self._load_pre_data(
                img_info['video_id'], img_info['frame_id'],
                img_info['sensor_id'] if 'sensor_id' in img_info else 1)
            if self.opt.mask_input:
                img_path = img_path.split('img1')
                mask_path = img_path[0]+'mask'+img_path[1]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask[np.newaxis,:,:]
            if flipped:
                pre_image = pre_image[:, ::-1, :].copy()
                pre_anns = self._flip_anns(pre_anns, width)
                if self.opt.mask_input:
                    mask = mask[:, ::-1, :].copy()
            if opt.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input
                trans_output_pre = trans_output
            else:
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c, s, width, height, disturb=True)
                s_pre = s * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.input_w, opt.input_h])
                trans_output_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.output_w, opt.output_h])
            pre_img = self._get_input(pre_image, trans_input_pre)
            pre_hm, pre_cts, track_ids = self._get_pre_dets(
                pre_anns, trans_input_pre, trans_output_pre)
            ret['pre_img'] = pre_img
            if opt.pre_hm:
                if self.opt.mask_input:
                    ret['pre_hm'] = mask
                else:
                    ret['pre_hm'] = pre_hm  # (1,1024,1024)

        ### init samples
        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)
            if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            self._add_instance(
                ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
                calib, pre_cts, track_ids)

        if self.opt.debug > 0:
            gt_det = self._format_gt_det(gt_det)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
                    'img_path': img_path, 'calib': calib,
                    'flipped': flipped}
            ret['meta'] = meta

        # src_img = (ret['image'] * 255).astype(np.uint8).transpose(1, 2, 0)
        # src_img = cv2.resize(src_img, (512,512))
        # heat_img = (ret['hm'] * 255).astype(np.uint8).transpose(1, 2, 0)
        # debug_heat = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        # add_img = cv2.addWeighted(src_img, 0.3, debug_heat, 0.7, 0)
        # cv2.imwrite('../debug/data.jpg', add_img)
        # print(img_path)
        return ret

    def get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

        return img, anns, img_info, img_path

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]
        # If training, random sample nearby frames as the "previoud" frame
        # If testing, get the exact prevous frame
        if 'train' in self.split:
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
                       (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        else:
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if (img_info['frame_id'] - frame_id) == -1 and \
                       (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
            if len(img_ids) == 0:
                img_ids = [(img_info['id'], img_info['frame_id']) \
                           for img_info in img_infos \
                           if (img_info['frame_id'] - frame_id) == 0 and \
                           (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        rand_id = np.random.choice(len(img_ids))
        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
        return img, anns, frame_dist

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        trans = trans_input
        reutrn_hm = self.opt.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -99 or \
                    ('iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                if self.opt.big_radius:
                    radius = max(radius, min(math.ceil(h), math.ceil(w)) / 2.0)
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, track_ids

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            if type(s) == float:
                s = [s, s]
                c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            else:
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
                keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
                    self.num_joints, 3)
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = \
                        keypoints[e[1]].copy(), keypoints[e[0]].copy()
                anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

            if 'rot' in self.opt.heads and 'alpha' in anns[k]:
                anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                    else - np.pi - anns[k]['alpha']

            if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
                anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

            if self.opt.velocity and 'velocity' in anns[k]:
                anns[k]['velocity'] = [-10000, -10000, -10000]

        return anns

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep': 1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []

        if 'hm_hp' in self.opt.heads:
            num_joints = self.num_joints
            ret['hm_hp'] = np.zeros(
                (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
            ret['hm_hp_mask'] = np.zeros(
                (max_objs * num_joints), dtype=np.float32)
            ret['hp_offset'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret['hp_offset_mask'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if 'rot' in self.opt.heads:
            ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
            ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
            ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({'rot': []})

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        if ('hm_hp' in ret) and cls_id <= 1:
            self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                         [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

    def _add_instance(
            self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
            aug_s, calib, pre_cts=None, track_ids=None):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        if self.opt.big_radius:
            radius = max(radius, min(math.ceil(h), math.ceil(w)) / 2.0)
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
            np.array([ct[0] - w / 2, ct[1] - h / 2,
                      ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['clses'].append(cls_id - 1)
        gt_det['cts'].append(ct)

        if 'tracking' in self.opt.heads:
            if ann['track_id'] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann['track_id'])]
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = pre_ct - ct_int
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                gt_det['tracking'].append(np.zeros(2, np.float32))

        if 'ltrb' in self.opt.heads:
            ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
                             bbox[2] - ct_int[0], bbox[3] - ct_int[1]
            ret['ltrb_mask'][k] = 1

        if 'ltrb_amodal' in self.opt.heads:
            ret['ltrb_amodal'][k] = \
                bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
                bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
            ret['ltrb_amodal_mask'][k] = 1
            gt_det['ltrb_amodal'].append(bbox_amodal)

        if 'nuscenes_att' in self.opt.heads:
            if ('attributes' in ann) and ann['attributes'] > 0:
                att = int(ann['attributes'] - 1)
                ret['nuscenes_att'][k][att] = 1
                ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
            gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

        if 'velocity' in self.opt.heads:
            if ('velocity' in ann) and min(ann['velocity']) > -1000:
                ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
                ret['velocity_mask'][k] = 1
            gt_det['velocity'].append(ret['velocity'][k])

        if 'hps' in self.opt.heads:
            self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

        if 'rot' in self.opt.heads:
            self._add_rot(ret, ann, k, gt_det)

        if 'dep' in self.opt.heads:
            if 'depth' in ann:
                ret['dep_mask'][k] = 1
                ret['dep'][k] = ann['depth'] * aug_s
                gt_det['dep'].append(ret['dep'][k])
            else:
                gt_det['dep'].append(2)

        if 'dim' in self.opt.heads:
            if 'dim' in ann:
                ret['dim_mask'][k] = 1
                ret['dim'][k] = ann['dim']
                gt_det['dim'].append(ret['dim'][k])
            else:
                gt_det['dim'].append([1, 1, 1])

        if 'amodel_offset' in self.opt.heads:
            if 'amodel_center' in ann:
                amodel_center = affine_transform(ann['amodel_center'], trans_output)
                ret['amodel_offset_mask'][k] = 1
                ret['amodel_offset'][k] = amodel_center - ct_int
                gt_det['amodel_offset'].append(ret['amodel_offset'][k])
            else:
                gt_det['amodel_offset'].append([0, 0])

    def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
        num_joints = self.num_joints
        pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
            if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)
        if self.opt.simple_radius > 0:
            hp_radius = int(simple_radius(h, w, min_overlap=self.opt.simple_radius))
        else:
            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = max(0, int(hp_radius))

        for j in range(num_joints):
            pts[j, :2] = affine_transform(pts[j, :2], trans_output)
            if pts[j, 2] > 0:
                if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
                        pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
                    ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
                    pt_int = pts[j, :2].astype(np.int32)
                    ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
                    ret['hp_ind'][k * num_joints + j] = \
                        pt_int[1] * self.opt.output_w + pt_int[0]
                    ret['hp_offset_mask'][k * num_joints + j] = 1
                    ret['hm_hp_mask'][k * num_joints + j] = 1
                    ret['joint'][k * num_joints + j] = j
                    draw_umich_gaussian(
                        ret['hm_hp'][j], pt_int, hp_radius)
                    if pts[j, 2] == 1:
                        ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
                        ret['hp_offset_mask'][k * num_joints + j] = 0
                        ret['hm_hp_mask'][k * num_joints + j] = 0
                else:
                    pts[j, :2] *= 0
            else:
                pts[j, :2] *= 0
                self._ignore_region(
                    ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1,
                    int(bbox[0]): int(bbox[2]) + 1])
        gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

    def _add_rot(self, ret, ann, k, gt_det):
        if 'alpha' in ann:
            ret['rot_mask'][k] = 1
            alpha = ann['alpha']
            if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                ret['rotbin'][k, 0] = 1
                ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                ret['rotbin'][k, 1] = 1
                ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
            gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
        else:
            gt_det['rot'].append(self._alpha_to_8(0))

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def _format_gt_det(self, gt_det):
        if (len(gt_det['scores']) == 0):
            gt_det = {'bboxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                      'scores': np.array([1], dtype=np.float32),
                      'clses': np.array([0], dtype=np.float32),
                      'cts': np.array([[0, 0]], dtype=np.float32),
                      'pre_cts': np.array([[0, 0]], dtype=np.float32),
                      'tracking': np.array([[0, 0]], dtype=np.float32),
                      'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                      'hps': np.zeros((1, 17, 2), dtype=np.float32), }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset['videos'] = []
        for i in range(len(self.coco.dataset['images'])):
            img_id = self.coco.dataset['images'][i]['id']
            self.coco.dataset['images'][i]['video_id'] = img_id
            self.coco.dataset['images'][i]['frame_id'] = 1
            self.coco.dataset['videos'].append({'id': img_id})

        if not ('annotations' in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset['annotations'])):
            self.coco.dataset['annotations'][i]['track_id'] = i + 1
