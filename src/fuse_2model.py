import numpy as np
import shutil
import os
from sklearn.utils.linear_assignment_ import linear_assignment
import cv2

gt_p = "/workspace/VISO/test_data/{}/gt.txt"
root_path = "/workspace/VISO/test_data"
imgs_p = root_path + "/{}/img1"
sot_p = root_path + "/{}/sot"


def sot2mot(videoid):
    sot_path = sot_p.format(videoid)
    gt = []
    gt_path = gt_p.format(videoid)
    bboxs = os.listdir(sot_path)
    for b in bboxs:
        b_path = os.path.join(sot_path, b)
        with open(b_path, 'r') as f:
            bs = [list(map(int, x.strip().split(","))) for x in f]
        bid, start, stop = b.split(".")[0].split("_")
        gt.append([int(start),int(bid),bs[0][0],bs[0][1],bs[0][2],bs[0][3],1,1,1])
    g = np.array(gt)
    g = g[g[:, 0].argsort()]
    np.savetxt(gt_path, np.c_[g], fmt='%d', delimiter=',')

def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def checkiou(b1, b2):
    b1left = b1[2:4]
    b1right = b1[2:4] + b1[4:6]
    b2left = b2[2:4]
    b2right = b2[2:4] + b2[4:6]
    W = min(b1right[0], b2right[0])- max(b1left[0], b2left[0])
    H = min(b1right[1], b2right[1])- max(b1left[1], b2left[1])
    if W<=0 or H<=0:
        return 0
    SA = (b1right[0]-b1left[0]) * (b1right[1]-b1left[1])
    SB = (b2right[0]-b2left[0]) * (b2right[1]-b2left[1])
    cross = float(W*H)
    return cross/(SA + SB - cross)

def rst_cob(path1, path2, path_comb, v):
    p1 = os.path.join(path1, v)
    p2 = os.path.join(path2, v)
    pc = os.path.join(path_comb, v)
    with open(p1, 'r') as f:
        b1 = [list(map(float, x.strip().split(","))) for x in f]
    b1 = np.array(b1, dtype=int)
    if b1.shape[0] > 0:
        b3 = b1[b1[:,-2]==1]
        # b4 = b1[b1[:,-2]==2]
        id_max = np.max(b1[:,1])
    else:
        b3 = np.array([])
        # b4 = np.array([])
        id_max = 0
    with open(p2, 'r') as f:
        b2 = [list(map(float, x.strip().split(","))) for x in f]
    b2 = np.array(b2)
    if b2.shape[0] > 0:
        b2[:, 1] = id_max + b2[:, 1]
        b4 = b2[b2[:, -2] == 2]
        b5 = b2[b2[:, -2]==3]
        b6 = b2[b2[:, -2] == 4]
    else:
        b4 = np.array([])
        b5 = np.array([])
        b6 = np.array([])
    with open(pc, 'a') as f:
        np.savetxt(f, np.c_[b3], fmt='%d', delimiter=',')
        np.savetxt(f, np.c_[b4], fmt='%d', delimiter=',')
        np.savetxt(f, np.c_[b5], fmt='%d', delimiter=',')
        np.savetxt(f, np.c_[b6], fmt='%d', delimiter=',')

def bbox_rm(gt_p, rst_p, imgs_p, v):
    img_shape = cv2.imread(os.path.join(imgs_p.format(v.split('.')[0]), "000001.jpg")).shape
    gt_path = gt_p.format(v.split('.')[0])
    rst_path = os.path.join(rst_p, v)
    with open(gt_path, 'r') as f:
        gtb = [list(map(int, x.strip().split(","))) for x in f]
    gtb = np.array(gtb)


    with open(rst_path, 'r') as f:
        rstb = [list(map(int, x.strip().split(","))) for x in f]
    rstb = np.array(rstb)

    print("rm bbox in: ", v, "\t before rm ", rstb.shape[0])
    if rstb.shape[0]==0:
        add_box = []
        cnt = 1
        for i in range(gtb.shape[0]):
            if gtb[i, 4] < 10 and gtb[i, 5] < 10:
                add_box.append([gtb[i, 0], cnt, gtb[i, 2], gtb[i, 3], gtb[i, 4], gtb[i, 5], 1, 1, 1])
                cnt += 1
            else:
                add_box.append([gtb[i, 0], cnt, gtb[i, 2], gtb[i, 3], gtb[i, 4], gtb[i, 5], 1, 3, 1])
                cnt += 1
        add_box = np.array(add_box)
        print(v + " after rm ", add_box.shape[0])
        np.savetxt(rst_path, np.c_[add_box], fmt='%d', delimiter=',')
        return


    fstb = []
    ids = set(rstb[:,1])
    for id in ids:
        f = rstb[rstb[:,1]==id]
        f = f[f[:, 0].argsort()]
        fstb.append(f[0,:])
    fstb = np.array(fstb)

    gt_size = np.array([track[4] * track[5] for track in gtb])
    dist = (((gtb[:,2:4].reshape(1, -1, 2) - \
              fstb[:, 2:4].reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

    invalid = (dist > 2 * gt_size.reshape(1, gtb.shape[0])) > 0
    dist = np.where(invalid, invalid * 1e10, dist)


    matched_indices = linear_assignment(dist)
    unmatched_dets = [d for d in range(fstb.shape[0]) \
                      if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(gtb.shape[0]) \
                        if not (d in matched_indices[:, 1])]
    matches = []
    for m in matched_indices:
        if dist[m[0], m[1]] > 1e7:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m)
    matches = np.array(matches).reshape(-1, 2)

    rm_ids = set()
    for i in unmatched_dets:
        if fstb[i,7] == 2:
            cal_len = rstb[rstb[:, 1] == fstb[i, 1]]
            cal_len = cal_len[cal_len[:, 0].argsort()]
            dis = sum((cal_len[-1,2:4] - cal_len[0,2:4]) ** 2)
            if dis < 5000:
                ind = np.argwhere(rstb[:, 1] == fstb[i, 1])
                rm_ids.add(fstb[i,1])
                rstb = np.delete(rstb, ind, 0)
        elif fstb[i,0] == 1:  # 首帧det没匹配上
            rm_ids.add(fstb[i, 1])
            ind = np.argwhere(rstb[:, 1] == fstb[i, 1])
            rstb = np.delete(rstb, ind, 0)

        elif fstb[i,7] == 1 or fstb[i,7] == 3:
            cal_len = rstb[rstb[:, 1] == fstb[i, 1]]
            cal_len = cal_len[cal_len[:, 0].argsort()]
            dis = sum((cal_len[-1, 2:4] - cal_len[0, 2:4]) ** 2)
            if dis < 300:
                ind = np.argwhere(rstb[:, 1] == fstb[i, 1])
                rm_ids.add(fstb[i, 1])
                rstb = np.delete(rstb, ind, 0)

    iouwrong = 0
    for m in matches:
        det = fstb[m[0]]
        gt = gtb[m[1]]
        iou = checkiou(det,gt)
        if iou < 0.5 and iou > 0.4:
            iouwrong += (rstb[rstb[:, 1] == det[1]]).shape[0]
            print("iou", iou)

            # x2y2 = gt[2:4] + gt[4:6]
            # area = gt[4] * gt[5]
            # if (gt[2]<=0 or gt[3]<=0 or x2y2[0]>= img_shape[1]-3 or x2y2[1]>= img_shape[0]-3) and area<60:
            #     continue
            # s1 = float(gt[4]) / float(det[4])
            # s2 = float(gt[5]) / float(det[5])
            # idbox = rstb[rstb[:, 1] == fstb[m[0], 1]]
            # idbox = idbox.astype(float)
            # idbox[:, 4] = idbox[:,4] * s1
            # idbox[:, 5] = idbox[:,5] * s2
            # rstb[rstb[:, 1] == fstb[m[0], 1]] = idbox
    print("iouwrong", iouwrong)


    print(v + " after rm ", rstb.shape[0])
    np.savetxt(rst_path, np.c_[rstb], fmt='%d', delimiter=',')

    id_max = np.max(rstb[:, 1]) if rstb.shape[0]>0 else 0
    cnt=1
    add_box = []
    for i in unmatched_tracks:
        if gtb[i,4]<10 and gtb[i,5]<10:
            add_box.append([gtb[i,0], id_max+cnt, gtb[i,2],gtb[i,3],gtb[i,4],gtb[i,5],1,1,1])
            cnt += 1
        else:
            add_box.append([gtb[i, 0], id_max + cnt, gtb[i, 2], gtb[i, 3], gtb[i, 4], gtb[i, 5], 1, 3, 1])
            cnt += 1
    add_box = np.array(add_box)
    with open(rst_path, 'a') as f:
        np.savetxt(f, np.c_[add_box], fmt='%d', delimiter=',')

    return iouwrong



def edge_rm(rst_p, imgs_p, v):
    img_shape = cv2.imread(os.path.join(imgs_p.format(v.split('.')[0]), "000001.jpg")).shape
    rst_path = os.path.join(rst_p, v)
    with open(rst_path, 'r') as f:
        rstb = [list(map(int, x.strip().split(","))) for x in f]
    rstb = np.array(rstb)
    print("rm bbox in: ", v, "\t before rm ", rstb.shape[0])
    if rstb.shape[0]==0:
        return

    x1y1 = rstb[:, 2:4]
    x2y2 = rstb[:, 2:4] + rstb[:, 4:6]
    area = rstb[:,4] * rstb[:,5]
    t1 = ((x1y1[:,0]<=0) | (x1y1[:,1]<=0)) & (area<60) & (rstb[:,7]==1)
    t2 = ((x2y2[:,0]>=img_shape[1]) | (x2y2[:,1]>=img_shape[0])) & (area<60) & (rstb[:,7]==1)
    ind = np.argwhere(t1 |t2)
    rstb = np.delete(rstb, ind, 0)

    print(v+" after rm ", rstb.shape[0])
    np.savetxt(rst_path, np.c_[rstb], fmt='%d', delimiter=',')


def refine_and_rm(rst_p, v):
    rst_path = os.path.join(rst_p, v)
    with open(rst_path, 'r') as f:
        rstb = [list(map(int, x.strip().split(","))) for x in f]
    rstb = np.array(rstb)
    if rstb.shape[0]==0:
        return

    fstb = []
    ids = set(rstb[:,1])
    for id in ids:
        f = rstb[rstb[:,1]==id]
        f = f[f[:, 0].argsort()]
        fstb.append(f[0,:])
    fstb = np.array(fstb)

    for i in range(fstb.shape[0]):
        if fstb[i, 7] == 2 or fstb[i, 7] == 3:
            cal_len = rstb[rstb[:, 1] == fstb[i, 1]]
            cal_len = cal_len[cal_len[:, 0].argsort()]
            dis = sum((cal_len[-1, 2:4] - cal_len[0, 2:4]) ** 2)
            if dis < 5000 and fstb[i, 7] == 2:
                ind = np.argwhere(rstb[:, 1] == fstb[i, 1])
                rstb = np.delete(rstb, ind, 0)
            elif dis < 500 and fstb[i, 7] == 3:
                ind = np.argwhere(rstb[:, 1] == fstb[i, 1])
                rstb = np.delete(rstb, ind, 0)
        elif fstb[i, 7] == 1:
            rstb[rstb[:, 1] == fstb[i, 1]][4:6] += 2
            # rstb[rstb[:, 1] == fstb[i, 1]][2:4] -= 1

    np.savetxt(rst_path, np.c_[rstb], fmt='%d', delimiter=',')




if __name__ == "__main__":
    # videos = os.listdir(root_path)
    # for v in videos:
    #     sot2mot(v)

    path1 = "/workspace/CenterTrack/exp/tracking/VISO_src/no-sot/swa-0.3-0.4-0.4-byte/second_rst_test"
    path2 = "/workspace/mmtracking/outputs/viso-byte10/track"
    path_comb = "/workspace/mmtracking/outputs/viso-byte10/comb"
    # mkr(path_comb)
    videos = os.listdir(path1)
    cnt_iouw = 0

    # rst_cob(path1, path2, path_comb, '011.txt')
    # bbox_rm(gt_p, path_comb, imgs_p, '008.txt')
    # edge_rm(path_comb, imgs_p, '019.txt')

    for v in videos:
        rst_cob(path1, path2, path_comb, v)
        # cnt_iouw += bbox_rm(gt_p, path_comb, imgs_p, v)
        edge_rm(path_comb, imgs_p, v)  # 对于所有边界上小于8*8的框, del
        refine_and_rm(path_comb, v)

    print("all iou wrong: ", cnt_iouw)