# CFTracker: Multi-Object Tracking With Cross-Frame Connections in Satellite Videos


## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.


## Use CenterTrack
First, `cd src/`
### 1. train
```
CUDA_VISIBLE_DEVICES=1,2 python main.py tracking --gpus 0,1 --dataset custom --custom_dataset_ann_path /workspace/JLTrack/annotations/ --custom_dataset_img_path /workspace/JLTrack/ --num_classes 2 --input_h 608 --input_w 992 --batch_size 32 --lr 1.25e-4 --lr_step '60,90' --save_point '70,100' --num_epochs 120 --val_intervals 60 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model ../models/crowdhuman.pth --atten_method corr --atten_space --atten_channel --exp_id JL_Att
```
### 2. test
```
python demo.py tracking --video_h 1080 --video_w 1920 --input_h 1088 --input_w 1920 --num_class 2 --demo /workspace/JLTrack/test/ --dataset custom --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att --load_model /workspace/CenterTrack/exp/tracking/JL_Att/model_last.pth --atten_method corr --atten_space --atten_channel
```

## Thanks
This code is heavily borrowed from [CenterTrack](https://github.com/xingyizhou/CenterTrack), thanks the authors.

