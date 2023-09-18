from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..generic_dataset import GenericDataset
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import os
import numpy as np
import json


class CustomDataset(GenericDataset):
    num_categories = 2
    default_resolution = [-1, -1]
    class_name = ['plane', 'ship']
    max_objs = 400
    cat_ids = {1: 1}
    _valid_ids = [1, 2]

    def __init__(self, opt, split):
        assert (opt.custom_dataset_img_path != '') and \
               (opt.custom_dataset_ann_path != '') and \
               (opt.num_classes != -1) and \
               (opt.input_h != -1) and (opt.input_w != -1), \
            'The following arguments must be specified for custom datasets: ' + \
            'custom_dataset_img_path, custom_dataset_ann_path, num_classes, ' + \
            'input_h, input_w.'
        img_dir = os.path.join(opt.custom_dataset_img_path, split)
        ann_path = os.path.join(opt.custom_dataset_ann_path, split+'.json')
        self.opt = opt
        self.split = split
        self.num_categories = opt.num_classes
        self.class_name = ['plane', 'ship']
        self.default_resolution = [opt.input_h, opt.input_w]
        self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

        self.images = None
        # load image list and coco
        super().__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded Custom dataset {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    cat_id = item['class'] - 1
                    category_id = self._valid_ids[cat_id]
                    bbox = item['bbox']
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "iscrowd": 0,
                        "score": float("{:.2f}".format(item['score']))
                    }
                    detections.append(detection)
        return detections

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_coco.json'.format(save_dir), 'w'))
        # results_dir = os.path.join(save_dir, 'results_jilin')
        # if not os.path.exists(results_dir):
        #     os.mkdir(results_dir)
        # for video in self.coco.dataset['videos']:
        #     video_id = video['id']
        #     file_name = video['file_name']
        #     out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
        #     images = self.video_to_images[video_id]
        #     results_txt = []
        #     for image_info in images:
        #         if not (image_info['id'] in results):
        #             continue
        #         result = results[image_info['id']]
        #         # frame_id = image_info['frame_id']
        #         for key, value in result.items():
        #             for j in value:
        #                 x = str(int(round(j["bbox"][0])))
        #                 y = str(int(round(j["bbox"][1])))
        #                 w = str(int(round(j["bbox"][2]-j["bbox"][0])))
        #                 h = str(int(round(j["bbox"][3]-j["bbox"][1])))
        #                 results_txt.append(','.join([str(key), str(j["tracking_id"]), x,y,w,h,'1',str(j["class"]),'1\n']))
        #     with open(out_path, 'w') as f:
        #         f.writelines(results_txt)


    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results_coco.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # img_dir = os.path.join(self.opt.custom_dataset_img_path, self.split)
        # # save_dir is the dir of opt.exp_id
        # self.save_results(results, save_dir)
        # os.system('python tools/eval_motchallenge.py ' + \
        #           '{} '.format(img_dir) + \
        #           '{}/first_rst_{}/ '.format(save_dir, self.split) + \
        #            ' --eval_official')
