import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
import cv2
import os

need_gt = True

# gt_p = "/workspace/duibi/Centertrack/second_rst_test-120/{}.txt"
# save_path = "/workspace/duibi/Centertrack/view/"
# root_path = "/workspace/VISO/test_data"

# gt_p = "/workspace/duibi/CFtrack/inftrain_intrain/second_rst_test/{}.txt"
# save_path = "/workspace/duibi/CFtrack/inftrain_intrain/paper/"



root_path = "/workspace/JLTrack/train"
imgs_p = root_path + "/{}/img"
sot_p = root_path + "/{}/sot"

gt_p = '/workspace/CenterTrack/exp/tracking/JL_ch_eyAtt_inftrain_intrain/second_rst_train/{}.txt'
# gt_p = '/workspace/VISO/val/{}/gt.txt' # root_path + "/{}/gt/gt.txt"
save_path = "/workspace/CenterTrack/exp/tracking/JL_ch_eyAtt_inftrain_intrain/view/"

save_mat = "video" # "video" "imgs"
color_map = [[0, 255, 0],  # Plum 3
             [0, 255, 0],  # Orange 3
             [0, 255, 0],  # Plum 3
             [0, 255, 0]  # Orange 3
             # [92, 53, 102],  # Plum 3
             # [206, 92, 0],  # Orange 3
             # [114, 159, 207], # Sky Blue ,
             # [233, 185, 110]  # Chocolate 1
            ]

tango_color = [[168, 230, 207],
               [220, 237, 193],
               [97, 191, 191],
               [250, 228, 217],
               [136, 133, 162],
               [187, 222, 215],
               [255, 211, 182],
               [255, 170, 165],
               #  [252, 233, 79],  # Butter 1
               # [237, 212, 0],  # Butter 2
               # [196, 160, 0],  # Butter 3
               # [138, 226, 52],  # Chameleon 1
               # [115, 210, 22],  # Chameleon 2
               # [78, 154, 6],  # Chameleon 3
               # [252, 175, 62],  # Orange 1
               # [245, 121, 0],  # Orange 2
               # [206, 92, 0],  # Orange 3
               # [114, 159, 207],  # Sky Blue 1
               # [52, 101, 164],  # Sky Blue 2
               # [32, 74, 135],  # Sky Blue 3
               # [173, 127, 168],  # Plum 1
               # [117, 80, 123],  # Plum 2
               # [92, 53, 102],  # Plum 3
               # [233, 185, 110],  # Chocolate 1
               # [193, 125, 17],  # Chocolate 2
               # [143, 89, 2],  # Chocolate 3
               # [239, 41, 41],  # Scarlet Red 1
               # [204, 0, 0],  # Scarlet Red 2
               # [164, 0, 0],  # Scarlet Red 3
               # [238, 238, 236],  # Aluminium 1
               # [211, 215, 207],  # Aluminium 2
               # [186, 189, 182],  # Aluminium 3
               # [136, 138, 133],  # Aluminium 4
               # [85, 87, 83],  # Aluminium 5
               # [46, 52, 54],  # Aluminium 6
               ]
tango_color = np.array(tango_color).reshape((-1, 3)) /255.0
# tango_color = np.array(tango_color, np.uint8).reshape((-1, 1, 1, 3))
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def gene_gtview(videoid, tx=None):
    if tx is None:
        tx = ["id", "score"]
    gt_path = gt_p.format(videoid)
    imgs_path = imgs_p.format(videoid)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    fontsize = 0.4

    with open(gt_path, 'r') as f:
        g = [list(map(float, x.strip().split(","))) for x in f]
    if len(g)==0:
        return
    g = np.array(g)
    g = g[g[:,0].argsort()]
    score = g[:,-3]
    gt = np.array(g, dtype=int)
    cnt = 0
    nums = len([x for x in os.listdir(imgs_path) if x.endswith('.jpg')])
    if save_mat == "video":
        frame = os.path.join(imgs_path, str(1).zfill(6) + ".jpg")
        img = cv2.imread(frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID
        out = cv2.VideoWriter(save_path + '{}.mp4'.format(videoid), fourcc, 15, (img.shape[1],img.shape[0]))
    for i in range(nums):
        i = i+1
        frame = os.path.join(imgs_path, str(i).zfill(6)+".jpg")
        img = cv2.imread(frame)
        # scale_w, scale_h = 512.0/img.shape[0], 512.0/img.shape[1]
        # img = cv2.resize(img, [512,512])
        if need_gt:
            if i in gt[:, 0]:
                while cnt<gt.shape[0] and gt[cnt,0] == i:
                    bbox = gt[cnt,:]
                    # bbox[2] = scale_w * bbox[2]
                    # bbox[4] = scale_w * bbox[4]
                    # bbox[3] = scale_h * bbox[3]
                    # bbox[5] = scale_h * bbox[5]
                    ct = (int(bbox[2] + bbox[4] / 2), int(bbox[3] + bbox[5] / 2))
                    # cv2.circle(img, ct, 3, [0,0,255], -1, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, (bbox[2], bbox[3]), (int(bbox[2]+bbox[4]), int(bbox[3]+bbox[5])), color_map[bbox[7]], 2)
                    txt = ""
                    if "id" in tx:
                        txt += str(bbox[1])
                    if "score" in tx:
                        txt += "  {:.2f}".format(score[cnt])
                    if len(tx) > 0:
                        cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
                        cv2.rectangle(img, (bbox[2], bbox[3] - cat_size[1] - 2),
                                      (bbox[2] + cat_size[0], bbox[3]), color_map[bbox[7]], -1)
                        cv2.putText(img, txt, (bbox[2], bbox[3] - thickness - 1),
                                    font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    cnt += 1
        if save_mat == "video":
            out.write(img)
        else:
            if not os.path.exists(save_path+str(videoid)):
                os.makedirs(save_path+str(videoid))
            save_n = os.path.join(save_path+str(videoid), str(i).zfill(6)+".jpg")
            cv2.imwrite(save_n, img)

    if save_mat == "video" and out is not None:
        out.release()

def scale_gtview(videoid, tx=None):
    scale_p = 1.0
    if tx is None:
        tx = ["id", "score"]
    gt_path = gt_p.format(videoid)
    imgs_path = imgs_p.format(videoid)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    fontsize = 0.3

    with open(gt_path, 'r') as f:
        g = [list(map(float, x.strip().split(","))) for x in f]
    if len(g)==0:
        return
    g = np.array(g)
    g = g[g[:,0].argsort()]
    score = g[:,-3]
    gt = np.array(g, dtype=int)
    cnt = 0
    nums = [int(x.split('.')[0]) for x in os.listdir(imgs_path) if x.endswith('.jpg')]
    # nums = len([x for x in os.listdir(imgs_path) if x.endswith('.jpg')])
    if save_mat == "video":
        frame = os.path.join(imgs_path, str(1).zfill(6) + ".jpg")
        img = cv2.imread(frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID
        out = cv2.VideoWriter(save_path + '{}.mp4'.format(videoid), fourcc, 15, (img.shape[1]/scale_p, img.shape[0]/scale_p))
    # for i in range(nums):
    #     i = i + 1
    for i in nums:
        frame = os.path.join(imgs_path, str(i).zfill(6)+".jpg")
        img = cv2.imread(frame)
        scale_h, scale_w = int(img.shape[0]/scale_p), int(img.shape[1]/scale_p)
        img = cv2.resize(img, [scale_w,scale_h])
        if need_gt:
            if i in gt[:, 0]:
                cnt = np.where(gt[:,0]==i)[0][0]
                while cnt<gt.shape[0] and gt[cnt,0] == i:
                    bbox = gt[cnt,:].astype(float)
                    bbox[2:6] = bbox[2:6]/scale_p
                    bbox[2:4] = np.round(bbox[2:4])
                    bbox[4:6] = np.ceil(bbox[4:6])
                    bbox = bbox.astype(int)
                    # bbox[2] = scale_w * bbox[2]
                    # bbox[4] = scale_w * bbox[4]
                    # bbox[3] = scale_h * bbox[3]
                    # bbox[5] = scale_h * bbox[5]
                    ct = (int(bbox[2] + bbox[4] / 2.0), int(bbox[3] + bbox[5] / 2.0))
                    # cv2.circle(img, ct, 3, [0,0,255], -1, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[2]+bbox[4], bbox[3]+bbox[5]), color_map[bbox[7]], 1)
                    txt = ""
                    if "id" in tx:
                        txt += str(bbox[1])
                    if "score" in tx:
                        txt += "  {:.2f}".format(score[cnt])

                    if txt != '':
                        cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
                        cv2.rectangle(img, (bbox[2], bbox[3] - cat_size[1] - 2),
                                      (bbox[2] + cat_size[0], bbox[3]), color_map[bbox[7]], -1)
                        cv2.putText(img, txt, (bbox[2], bbox[3] - thickness - 1),
                                font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    cnt += 1
        if save_mat == "video":
            out.write(img)
        else:
            if not os.path.exists(save_path+str(videoid)):
                os.makedirs(save_path+str(videoid))
            save_n = os.path.join(save_path+str(videoid), str(i).zfill(6)+".jpg")
            cv2.imwrite(save_n, img)

    if save_mat == "video" and out is not None:
        out.release()

def gt_analyze(gt_root, cls_num=2, label=None):
    if label is None:
        label = ["aircraft", "ship"]
    txts = sorted(os.listdir(gt_root))
    cnt = []     # 50 100 150 200 ... 1000 ...
    for i in range(cls_num):
        cnt.append({})
    for txt in txts:
        txt = os.path.join(gt_root, txt)
        with open(txt, 'r') as f:
            g = [list(map(float, x.strip().split(","))) for x in f]
            for i in range(len(g)):
                cls = int(g[i][-2])-1
                pix = g[i][4] * g[i][5]
                if int(pix/50) in cnt[cls]:
                    cnt[cls][int(pix/50)] += 1
                else:
                    cnt[cls][int(pix / 50)] = 1
    for i in range(cls_num):
        cnt[i] = dict(sorted(cnt[i].items()))

    save = json.dumps(dict(zip(label,cnt)))
    with open('JL_testgt.json', 'w') as f:
        f.write(save)
    return cnt

def gt_plt(json_path, cls_num=2, label=None):
    colors = ['#FAC748', '#1D2F6F', '#6EAF46', '#FAC748']  # dark blue 1D2F6F, pupple blue 8390FA, green 6EAF46, yellow FAC748
    # colors = ['#FAC748','#1D2F6F']
    if label is None:
        label = ['aircraft', 'ship']
    with open(json_path, 'r') as f:
        cnt = json.load(f)
    box_sum = 0
    data = []
    for i in cnt.keys():
        data_i = []
        for j in cnt[i].keys():
            box_sum += cnt[i][j]
            data_i.append([int(j), cnt[i][j]])
        data_i = np.array(data_i)
        data.append(data_i)

    fig, ax = plt.subplots(1, figsize=(12,10))
    # max_pix = 50 * (max(data[0][:,0].max(), data[1][:,0].max())+1)
    max_pix = 50 * 40
    x = np.array(list(range(25,max_pix+25,50)))
    bottom = np.zeros_like(x)

    for i in range(cls_num):
        y = np.zeros_like(x)
        for item in data[i]:
            if item[0] < y.size:
                y[item[0]] = item[1]
        plt.bar(x, y, width=30, bottom=bottom, color=colors[i])
        bottom = bottom + y

    perct90 = int(box_sum * 0.9)
    stack_num = np.cumsum(bottom)
    line90 = np.where(stack_num > perct90)[0][0]


    plt.legend(label, ncol=cls_num, frameon=False)
    plt.axvline((line90 + 1) * 50, 0, 7000, color="red", linestyle="--")
    plt.text((line90 + 1) * 50+10, 180000, '>90%', color="red")
    plt.xlabel('pix')
    plt.ylabel('num')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.xlim(-0.5, ax.get_xticks()[-1] + 0.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    fig.savefig('JL_testgt.jpg')

def plotMOTA():
    x = np.array([0.4, 0.5, 0.6, 0.7])
    y1 = np.array([51.94, 58.08, 60.36, 59.32])
    y1b = np.array([58.42, 58.42, 58.42, 58.42])
    y2 = np.array([79.36, 81.77, 80.17, 75.8])
    y2b = np.array([73.72, 73.72, 73.72, 73.72])
    y3 = (y1+y2)/2.0
    y3b = np.array([66.07, 66.07, 66.07, 66.07])
    plt.plot(x, y1, "o-", color='r', label='ship')
    plt.plot(x, y1b, "--", color='r')
    plt.plot(x, y2, "*-", color='b', label='airplane')
    plt.plot(x, y2b, "--", color='b')
    plt.plot(x, y3, "^-", color='g', label='all')
    plt.plot(x, y3b, "--", color='g')
    plt.ylim(50, 90)
    plt.xticks(x)
    plt.xlabel(r"Feedback threshold $\theta_2$")
    plt.ylabel("MOTA")
    plt.legend(loc="best")
    plt.savefig("sense-MOTA.jpg", bbox_inches='tight')

def plotIDF1():
    x = np.array([0.4, 0.5, 0.6, 0.7])
    y1 = np.array([73, 74.82, 75.39, 74.77])
    y1b = np.array([74.32, 74.32, 74.32, 74.32])
    y2 = np.array([89.55, 90.49, 89.12, 86.13])
    y2b = np.array([84.83, 84.83, 84.83, 84.83])
    y3 = (y1+y2)/2.0
    y3b = np.array([79.58, 79.58, 79.58, 79.58])
    plt.plot(x, y1, "o-", color='r', label='ship')
    plt.plot(x, y1b, "--", color='r')
    plt.plot(x, y2, "*-", color='b', label='airplane')
    plt.plot(x, y2b, "--", color='b')
    plt.plot(x, y3, "^-", color='g', label='all')
    plt.plot(x, y3b, "--", color='g')
    plt.ylim(70, 95)
    plt.xticks(x)
    plt.xlabel(r"Feedback threshold $\theta_2$")

    plt.ylabel("IDF1")
    plt.legend(loc="best")
    plt.savefig("sense-IDF1.jpg", bbox_inches='tight')

def plotIDP():
    x = np.array([0.4, 0.5, 0.6, 0.7])
    y1 = np.array([71.26, 76.97, 81.58, 83.8])
    y1b = np.array([80.76, 80.76, 80.76, 80.76])
    y2 = np.array([88.75, 92.99, 95.75, 95.81])
    y2b = np.array([95.76, 95.76, 95.76, 95.76])
    y3 = (y1+y2)/2.0
    y3b = np.array([72.495, 72.495, 72.495, 72.495])
    plt.plot(x, y1, "o-", color='r', label='ship')
    plt.plot(x, y1b, "--", color='r')
    plt.plot(x, y2, "*-", color='b', label='airplane')
    plt.plot(x, y2b, "--", color='b')
    plt.plot(x, y3, "^-", color='g', label='all')
    plt.plot(x, y3b, "--", color='g')
    plt.ylim(65, 100)
    plt.xticks(x)
    plt.xlabel(r"Feedback threshold $\theta_2$")

    plt.ylabel("IDP")
    plt.legend(loc="best")
    plt.savefig("sense-IDP.jpg", bbox_inches='tight')

def plotIDR():
    x = np.array([0.4, 0.5, 0.6, 0.7])
    y1 = np.array([74.83, 72.78, 70.07, 67.5])
    y1b = np.array([68.84, 68.84, 68.84, 68.84])
    y2 = np.array([90.37, 88.12, 83.34, 78.23])
    y2b = np.array([76.15, 76.15, 76.15, 76.15])
    y3 = (y1+y2)/2.0
    y3b = np.array([72.495, 72.495, 72.495, 72.495])
    plt.plot(x, y1, "o-", color='r', label='ship')
    plt.plot(x, y1b, "--", color='r')
    plt.plot(x, y2, "*-", color='b', label='airplane')
    plt.plot(x, y2b, "--", color='b')
    plt.plot(x, y3, "^-", color='g', label='all')
    plt.plot(x, y3b, "--", color='g')
    plt.ylim(65, 100)
    plt.xticks(x)
    plt.xlabel(r"Feedback threshold $\theta_2$")

    plt.ylabel("IDR")
    plt.legend(loc="best")
    plt.savefig("sense-IDR.jpg", bbox_inches='tight')

def collect_gts(psrc, pdest):
    '''
    :param psrc: "/workspace/JLTrack/test"
    :param pdest: "/workspace/JLTrack/gts/gt_test"
    '''
    videos = sorted(os.listdir(psrc))
    for v in videos:
        path = os.path.join(psrc, v, "gt", "gt.txt")
        shutil.copy(path, os.path.join(pdest, '{}.txt'.format(v)))

def collect_imgs(psrc, pdest):
    '''
    :param psrc: "/workspace/jilinTrack/First/test"
    :param pdest: "/workspace/jilinTrack/view"
    '''
    videos = sorted(os.listdir(psrc))
    n = psrc.split('/')
    if not os.path.exists(pdest):
        os.mkdir(pdest)
    for v in videos:
        path = os.path.join(psrc, v, "img/000001.jpg")
        gt_path = os.path.join(psrc, v, "gt/gt.txt")
        if os.path.exists(gt_path) and os.path.getsize(gt_path) == 0:
            continue
        shutil.copy(path, os.path.join(pdest, '{}-{}-{}.jpg'.format(n[-2],n[-1],v)))

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


def addval2train(train_p, val_p):
    tvideos = sorted(os.listdir(train_p))
    vvideos = sorted(os.listdir(val_p))
    for v in vvideos:
        vp = os.path.join(val_p, v)
        newid = str(int(tvideos[-1]) + int(v)).zfill(3)
        tp = os.path.join(train_p, newid)
        shutil.copytree(vp, tp)


def check_format(src_p, rst_p):
    mkr(rst_p)
    videos = os.listdir(src_p)
    for v in videos:
        src_path = os.path.join(src_p, v)
        rst_path = os.path.join(rst_p, v)

        with open(src_path, 'r') as f:
            rstb = [list(map(float, x.strip().split(","))) for x in f]
        rstb = np.array(rstb)
        rstb[:,7] = rstb[:,7] + 1
        np.savetxt(rst_path, np.c_[rstb], fmt='%d', delimiter=',')

def plot_visorst():
    mota = np.array([22.8, 2.3, 44.5, 68.0, 73.6, 58.2, 41.6, 50.9])
    motp = np.array([9.5, 28.0, 16.3, 15.2, 21.8, 28.6, 61.0, 65.0])
    size = np.array([24, 24, 24, 24, 24, 24, 24, 24]) * 3
    # size = np.array([17.2, 19.1, 18.2, 24.7, 24.7, 92.9, 2.1, 7.8]) *  3
    leg = ["CMOT", "FairMOT", "DTTP", "D&T", "Kalman", "MMB", "DSFNet", "CFTracker"]

    for m in range(8):
        plt.scatter(mota[m], motp[m], s=size[m], c=[tango_color[m]], label=leg[m])
    plt.xlabel("MOTA")
    plt.ylabel("MOTP")
    plt.legend(loc='best')
    plt.savefig("viso-campare.jpg", bbox_inches='tight')

if __name__ == "__main__":
    # plot_visorst()
    # plotMOTA()
    # plotIDF1()
    # plotIDR()

    # check_format('/workspace/duibi/FairMOT/mot_results', '/workspace/duibi/FairMOT/track')
    # addval2train("/workspace/VISO/train", "/workspace/VISO/val")
    # videos = os.listdir(root_path)
    # videos = ['009']  #'5', '13', '28', '34', '39','46'      '8', '31', '9','21','2','27'
    # videos = ['083', '084', '085', '086', '087', '088', '089']
    # mkr(save_path)
    # for v in videos:
        # sot2mot(v)
        # scale_gtview(v, tx=["id"])
    gene_gtview("39", tx=["id"])
    # # collect_gts('/workspace/viso1st/test', '/workspace/viso1st/gts/test-gts')
    # collect_gts(root_path, '/workspace/viso1st/gts/test-gts')
    # cnt = gt_analyze("/workspace/viso1st/gts/test-gts", cls_num=1, label=['car'])  # /workspace/JLTrack/gts/test-gts
    # gt_plt('viso1_testgt.json', cls_num=1, label=['car'])

    # cnt = gt_analyze("/workspace/JLTrack/gts/test-gts", cls_num=2, label=['airplane', 'ship'])  #
    # gt_plt('JL_testgt.json', cls_num=2, label=['airplane', 'ship'])





