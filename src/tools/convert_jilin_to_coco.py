from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import shutil
import cv2

HAVE_DET_FILE = False
HALF_VIDEO = False
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False


def rm_zero_data(root_path):
    if not os.path.exists(os.path.join(root_path, "gt")):
        return
    txt_lists = os.listdir(os.path.join(root_path, "gt"))
    for txt in txt_lists:
        txt_path = os.path.join(root_path, "gt", txt)
        dataid = txt.split('.')[0]
        if os.path.getsize(txt_path) == 0:
            os.remove(os.path.join(txt_path))
            if os.path.exists(os.path.join(root_path, "video", dataid)):
                shutil.rmtree(os.path.join(root_path, "video", dataid))
        else:
            if not os.path.exists(os.path.join(root_path, "video", dataid, "gt")):
                os.mkdir(os.path.join(root_path, "video", dataid, "gt"))
            shutil.move(txt_path, os.path.join(root_path, "video", dataid, "gt/gt.txt"))

def convert_airmot(data_p = '/workspace/JLTrack/', SPLITS=None):
    if SPLITS is None:
        SPLITS = ['train', 'test']
    OUT_PATH = data_p + 'annotations/'
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = data_p + (split if not HALF_VIDEO else 'train')
        out_path = OUT_PATH + '{}.json'.format(split)
        # out = {'images': [], 'annotations': [],
        #        'categories': [{'id': 1, 'name': 'plane'}, {'id': 2, 'name': 'ship'}],
        #        'videos': []}
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'car'}],
               'videos': []}
        seqs = sorted(os.listdir(data_path))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            img_path = seq_path + 'img/'
            ann_path = seq_path + 'gt/gt.txt'
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                    [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
            for i in range(num_images):
                if (i < image_range[0] or i > image_range[1]):
                    continue
                image_info = {'file_name': '{}/img/{:06d}.jpg'.format(seq, i + 1),
                              'id': image_cnt + i + 1,
                              'frame_id': i + 1 - image_range[0],
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            # try:
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                    continue
                track_id = int(anns[i][1])
                cat_id = int(anns[i][6])
                ann_cnt += 1
                ann = {'id': ann_cnt,
                       'category_id': cat_id,
                       'image_id': image_cnt + frame_id,
                       'track_id': track_id,
                       'bbox': anns[i][2:6].tolist(),
                       'area': int(anns[i][4] * anns[i][5]),
                       'conf': 1.0,
                       "iscrowd": 0, "ignore": 0}
                out['annotations'].append(ann)
            # except:
            #     anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=' ')
            #     for i in range(anns.shape[0]):
            #         frame_id = int(anns[i][0])
            #         if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
            #             continue
            #         track_id = int(anns[i][1])
            #         cat_id = 1 #int(anns[i][6])
            #         ann_cnt += 1
            #         box = np.array([anns[i][2], anns[i][3], anns[i][4]-anns[i][2], anns[i][5]-anns[i][3]]).tolist()
            #         ann = {'id': ann_cnt,
            #                'category_id': cat_id,
            #                'image_id': image_cnt + frame_id,
            #                'track_id': track_id,
            #                'bbox': box,
            #                'area': int(box[2] * box[3]),
            #                'conf': 1.0,
            #                "iscrowd": 0, "ignore": 0}
            #         out['annotations'].append(ann)
            print(' {} ann images'.format(int(anns[:, 0].max())))

            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))




def convert_viso(root_path='/workspace/VISO/', SPLITS=None):
    # only track car: id = 1
    # track_flag = [1,2,3,4]
    # dict_gt2jsonid = {1: 2, 2: 1, 3: 3, 4: 4}
    # cat = [{'id': 1, 'name': 'airplane'}, {'id': 2, 'name': 'car'}, {'id': 3, 'name': 'ship'}, {'id': 4, 'name': 'train'}]

    track_flag = [2, 3, 4]
    dict_gt2jsonid = {2: 1, 3: 2, 4: 3}
    cat = [{'id': 1, 'name': 'airplane'}, {'id': 2, 'name': 'ship'}, {'id': 3, 'name': 'train'}]

    if SPLITS is None:
        SPLITS = ['train', 'val']
    OUT_PATH = root_path + 'annotations/'
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = root_path + (split if not HALF_VIDEO else 'train')
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': cat,
               'videos': []}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue

            seq_path = '{}/{}/'.format(data_path, seq)
            img_path = seq_path + 'img1/'
            ann_path = seq_path + 'gt.txt'
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                    [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            youxiao_ann = 0
            for cls in track_flag:
                youxiao_ann += sum(anns[:,7] == cls)
            if youxiao_ann == 0:
                continue
            print(' {} ann images'.format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                    continue
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                if cat_id not in track_flag:
                    continue
                ann_cnt += 1
                ann = {'id': ann_cnt,
                       'category_id': dict_gt2jsonid[cat_id],
                       'image_id': image_cnt + frame_id,
                       'track_id': track_id,
                       'bbox': anns[i][2:6].tolist(),
                       'area': int(anns[i][4] * anns[i][5]),
                       'conf': 1.0,
                       "iscrowd": 0, "ignore": 0}
                out['annotations'].append(ann)


            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            for i in range(num_images):
                if (i < image_range[0] or i > image_range[1]):
                    continue
                image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                              'id': image_cnt + i + 1,
                              'frame_id': i + 1 - image_range[0],
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            image_cnt += num_images
            print('{}: {} images'.format(seq, num_images))


        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'), indent=4)


if __name__ == '__main__':
    # rm_zero_data(os.path.join(DATA_PATH, "test"))
    convert_airmot(data_p ='/workspace/viso1st/', SPLITS=['test'])


