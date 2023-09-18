from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
print(sys.path)
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import warnings
warnings.filterwarnings("ignore")

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def model_init(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.debug = max(opt.debug, 1)
    detector = Detector(opt)
    return detector


def demo(opt, detector, imgpath=None):
    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        out_name = opt.demo[opt.demo.rfind('/') + 1:]
    else:
        is_video = False
        imgs_path = opt.demo if imgpath == None else imgpath
        # Demo on images sequences
        if os.path.isdir(imgs_path):
            image_names = []
            ls = os.listdir(imgs_path)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(imgs_path, file_name))
            out_name = imgs_path.split('/')[-2]
        else:
            image_names = [opt.demo]
            out_name = imgs_path.split('/')[-2]

    # Initialize output video
    out = None

    print('out_name', out_name)
    if opt.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        save_video_dir = '../exp/tracking/{}/results/'.format(opt.exp_id)
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        out = cv2.VideoWriter(save_video_dir+'{}'.format(
            opt.exp_id + '_' + out_name+'.mp4'), fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}
    continue_flag = True

    while continue_flag:
        if is_video:
            print("begin read video....")
            ret, img = cam.read()
            if (not ret) and cnt == 0:
                print("read video False!!!")
            if img is None:
                save_and_exit(opt, out, results, out_name)
                break
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
                save_and_exit(opt, out, results, out_name)
                break
        cnt += 1
        if img is None:
            continue
        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        # cv2.imshow('input', img)

        # track or detect the image.
        meta = {}
        ret = detector.run(img, meta=meta)

        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        results[cnt] = ret['results']

        # save debug image to video
        if opt.save_video:
            out.write(ret['generic'])
        if opt.mysave_imgs:
            cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:
            save_and_exit(opt, out, results, out_name)
            return
    save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if not opt.demo_videos and opt.save_results and (results is not None):
        save_dir = '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.demo_videos or opt.debug>=1:
        results_txt=[]
        if opt.demo_videos:
            m1 = 'first' if opt.demo.find('First')>=0 else 'second'
            m2 = 'train' if opt.demo.find('train')>=0 else 'test'
            save_txt_dir = '../exp/tracking/{}/{}_rst_{}/'.format(opt.exp_id, m1, m2)
            if not os.path.exists(save_txt_dir):
                os.makedirs(save_txt_dir)
            save_txt_path = save_txt_dir + '{}.txt'.format(out_name)
        else:
            save_txt_path = '../exp/tracking/{}/results/'.format(opt.exp_id) + '{}.txt'.format(out_name)
        for key, value in results.items():
            for j in value:
                if j["active"] == 0:
                    continue
                x = str(int(round(j["bbox"][0])))
                y = str(int(round(j["bbox"][1])))
                w = str(int(round(j["bbox"][2]-j["bbox"][0])))
                h = str(int(round(j["bbox"][3]-j["bbox"][1])))
                results_txt.append(','.join([str(key), str(j["tracking_id"]), x,y,w,h,str(j["score"]),str(j["class"]),'1\n']))
        with open(save_txt_path, 'w') as f:
            f.writelines(results_txt)
    if opt.save_video and out is not None:
        out.release()



def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().init()
    detector = model_init(opt)
    if opt.demo_videos:
        video_list = os.listdir(opt.demo)
        for video in video_list:
            detector.reset()
            video_path = os.path.join(opt.demo, video, 'img')
            demo(opt, detector, video_path)
    else:
        demo(opt, detector)

