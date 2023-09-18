import json
import os
import pickle
import numpy as np

save_txt_dict = {1:1, 2:2}

def convert_eval_format(results):
    detections = []
    image_id = 1
    # is_next_img = False
    cur_img_id = 1
    for r in results:
        if r == "":
            continue
        r = r.split(',')
        if int(r[0]) == (cur_img_id):
            is_next_img = False
        else:
            is_next_img = True
            cur_img_id = int(r[0])

        if is_next_img:
            image_id+=1

        bbox = np.zeros((4))
        bbox[0] = float(r[2])
        bbox[1] = float(r[3])
        bbox[2] = float(r[4])
        bbox[3] = float(r[5])
        score = float(r[6])
        category_id = int(r[7])
        bbox_out = list(map(_to_float, bbox[0:4]))

        detection = {
            "image_id": int(image_id),
            "category_id": save_txt_dict[int(category_id)],
            "bbox": bbox_out,
            "score": float("{:.2f}".format(score))
        }
        # if len(bbox) > 5:
        #     extreme_points = list(map(self._to_float, bbox[5:13]))
        #     detection["extreme_points"] = extreme_points
        detections.append(detection)
        # image_id += 1

    return detections
def _to_float( x):
    return float("{:.2f}".format(x))

def gettxtall(path_root,frame_ids):
    results = []
    for id in frame_ids:
        f = open(os.path.join(path_root,id),'r')
        result = f.read().split('\n')
        results +=result
    return results


if __name__ == '__main__':
    txt_root = '/workspace/CenterTrack/exp/tracking/JL_ch_easyAtt/second_rst_test-pre0p6'
    frame_ids = os.listdir(txt_root)
    frame_ids.sort()

    txt_results = gettxtall(txt_root,frame_ids)



    # f = open(test_result,'rb')
    # results = pickle.load(f)


    json.dump(convert_eval_format(txt_results),
              open('{}/results_{}.json'.format('/workspace/CenterTrack/exp/tracking/JL_ch_easyAtt/second_rst_test-pre0p6', 'coco'), 'w'))
