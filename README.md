# CFTracker: Multi-Object Tracking With Cross-Frame Connections in Satellite Videos

## Update
- [2024/4/25] Released the Evaluation code.
- [2024/3/4] Released the results in VISO dataset.
- [2023/10/30] Released the pth model weight for the AIR-MOT dataset.
- [2023/9/18] Released the CFTracker code.


## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

My env: torch1.8.1 + cudnn 11.1


## Use CenterTrack
First, `cd src/`

**For AIR-MOT dataset：**
### 1. train
```
CUDA_VISIBLE_DEVICES=1,2 python main.py tracking --gpus 0,1 --dataset custom --custom_dataset_ann_path /workspace/JLTrack/annotations/ --custom_dataset_img_path /workspace/JLTrack/ --num_classes 2 --input_h 608 --input_w 992 --batch_size 32 --lr 1.25e-4 --lr_step '60,90' --save_point '70,100' --num_epochs 120 --val_intervals 60 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model ../models/crowdhuman.pth --atten_method reweight --atten_space --atten_channel --exp_id JL_Att
```
### 2. test
```
python demo.py tracking --video_h 1080 --video_w 1920 --input_h 1088 --input_w 1920 --num_class 2 --demo /workspace/JLTrack/test/ --dataset custom --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att --load_model /workspace/CenterTrack/exp/tracking/JL_Att/model_last.pth --atten_method reweight --atten_space --max_age 30 --mode test
```

**For VISO dataset：**

coming soon...
(Due to the small objects' size in VISO, don't forget to use `--lowfeat --down_ratio 2` for this data)

## Evalution for VISO
First, `cd TrackEval_sat/`, then run `python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL CFTracker`

To evaluate your results, 
- put your results in `TrackEval_sat/data/trackers/mot_challenge/MOT16-val/Your`, 
- modify `tracker_name` in `scripts/run_sat_challenge.py` 
- run `python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL Your`.

Notice: Due to my discovery that the test-gt provided by VISO is not completely consistent with the gt provided by DSFNet, for a fair comparison, I used the DSFNet gt to obtain the following evaluation results:

| Model | Score | MOTA | MOTP | 
| :-: | :-: | :-: | :-: |
| CFTracker | 58.2 | 57.6 | 58.9 |


(The results in the paper can be obtained by replacing the files in `TrackEval_sat/data/gt/mot_challenge/MOT16-val/` with `TrackEval_sat/viso_gt_bak/`, threshold is 0.4)


## Thanks
This code is heavily borrowed from [CenterTrack](https://github.com/xingyizhou/CenterTrack), thanks the authors.


## Cite
```bibtex
@ARTICLE{10129959,
  author={Kong, Lingyu and Yan, Zhiyuan and Zhang, Yidan and Diao, Wenhui and Zhu, Zining and Wang, Lei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CFTracker: Multi-Object Tracking With Cross-Frame Connections in Satellite Videos}, 
  year={2023},
  volume={61},
  number={},
  pages={1-14},
  keywords={Videos;Satellites;Tracking;Training;Semantics;Feature extraction;Trajectory;Cross-frame feature update (CFU);cross-frame training flow (CT);joint detection and tracking (JDT);multi-object tracking (MOT)},
  doi={10.1109/TGRS.2023.3278107}}

```
