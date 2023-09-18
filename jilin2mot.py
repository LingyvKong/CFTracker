import os
import shutil
import random

root_path = "/workspace/jilinTrack/"
target_root_path = "/workspace/JLTrack/"
sub_dir = ["First/train/", "First/test/", "Second/test/"]

if __name__ == "__main__":
    # txt_lists = os.listdir(os.path.join(root_path, "gt"))
    # for txt in txt_lists:
    #     txt_path = os.path.join(root_path, "gt", txt)
    #     if os.path.getsize(txt_path)==0:
    #         dataid = txt.split('.')[0]
    #         shutil.rmtree(os.path.join(root_path, "video", dataid))
    #     else:
    #         if not os.path.exists(os.path.join(root_path, "video", dataid, "gt")):
    #             os.mkdir(os.path.join(root_path, "video", dataid, "gt"))
    #         shutil.move(txt_path, os.path.join(root_path, "video", dataid, "gt/gt.txt"))

    all_video = []
    test_v = []
    istest = [[8,13,22],[17,24,30],[24,26,28,43]]
    seed=1230
    for i in range(3):
        path = root_path + sub_dir[i]
        videos = sorted(os.listdir(path))
        for video in videos:
            if int(video) in istest[i]:
                test_v.append(path + video)
            elif os.path.getsize(os.path.join(path+video, "gt/gt.txt")) != 0:
                all_video.append(path+video)
            else:
                print(path+video)

    random.Random(seed).shuffle(all_video)
    all_video = all_video + test_v
    len_train = round(0.7*len(all_video))
    print("all video nums: {}+{}={}".format(len_train,len(all_video)-len_train,len(all_video)))
    test_v = sorted(all_video[len_train:])

    with open("data_split.txt", 'w', encoding='utf-8') as f:
        f.writelines("all video nums: {}+{}={}\n".format(len_train, len(all_video) - len_train, len(all_video)))
        f.writelines([line+'\n' for line in test_v])
    # copy data
    for i in range(len_train):
        target_path = target_root_path+"train/"+str(i+1)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(all_video[i], target_path)
    for i in range(len_train, len(all_video)):
        target_path = target_root_path + "test/" +str(i+1-len_train)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(all_video[i], target_path)







