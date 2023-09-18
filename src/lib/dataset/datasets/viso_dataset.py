from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..generic_dataset import GenericDataset
from collections import defaultdict
import os
import numpy as np


class VisoDataset(GenericDataset):
    default_resolution = [1024,1024]

    # num_categories = 3
    # CLASSES = ('airplane', 'ship', 'train')
    # class_name = ['airplane', 'ship', 'train']

    # num_categories = 4
    # CLASSES = ('airplane','car','ship','train')
    # class_name = ['airplane','car','ship','train']

    num_categories = 1
    CLASSES = ('car')
    class_name = ['car']
    max_objs = 500
    cat_ids = {1: 1}

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
        self.class_name = ['airplane', 'car', 'ship', 'train']
        if opt.num_classes == 3:
            self.class_name = ['airplane', 'ship', 'train']
        elif opt.num_classes == 1:
            self.class_name = ['car']
        self.split = split
        self.num_categories = opt.num_classes

        self.default_resolution = [opt.input_h, opt.input_w]
        self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

        self.images = None
        # load image list and coco
        super().__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded Custom dataset {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results_jilin')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for video in self.coco.dataset['videos']:
            video_id = video['id']
            file_name = video['file_name']
            out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
            images = self.video_to_images[video_id]
            results_txt = []
            for image_info in images:
                if not (image_info['id'] in results):
                    continue
                result = results[image_info['id']]
                # frame_id = image_info['frame_id']
                for key, value in result.items():
                    for j in value:
                        x = str(int(round(j["bbox"][0])))
                        y = str(int(round(j["bbox"][1])))
                        w = str(int(round(j["bbox"][2]-j["bbox"][0])))
                        h = str(int(round(j["bbox"][3]-j["bbox"][1])))
                        results_txt.append(','.join([str(key), str(j["tracking_id"]), x,y,w,h,'1',str(j["class"]),'1\n']))
            with open(out_path, 'w') as f:
                f.writelines(results_txt)


    def run_eval(self, results, save_dir):
        pass
        # img_dir = os.path.join(self.opt.custom_dataset_img_path, self.split)
        # # save_dir is the dir of opt.exp_id
        # self.save_results(results, save_dir)
        # os.system('python tools/eval_motchallenge.py ' + \
        #           '{} '.format(img_dir) + \
        #           '{}/first_rst_{}/ '.format(save_dir, self.split) + \
        #            ' --eval_official')
