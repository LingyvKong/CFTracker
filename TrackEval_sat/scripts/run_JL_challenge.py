
""" run_sat_challenge.py

Run example:
run_JL_challenge.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL Lif_T

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""

import sys
import os
import argparse
import numpy as np
import shutil
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

def main(name, cls):
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.JLChallenge2DBox.get_default_dataset_config()
    default_dataset_config['TRACKERS_FOLDER'] = '/workspace/TrackEval/data/trackers/jl_challenge/'
    default_dataset_config['CLASSES_TO_EVAL'] = [cls]
    default_dataset_config['TRACKER_SUB_FOLDER'] = name
    default_metrics_config = {'METRICS': ['CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.JLChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    rst, _ = evaluator.evaluate(dataset_list, metrics_list)
    mota = rst["JLChallenge2DBox"]["JLTrack"]["COMBINED_SEQ"][cls]['CLEAR']["MOTA"]
    return mota

def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

# def split_cls_from_txt(path):
#     name_dict = {1: "airplane", 2: "ship"}
#     seqs = sorted([s for s in os.listdir(path) if not os.path.isdir(os.path.join(path, s))])
#     for c in range(1, len(name_dict)+1):
#         mkr(os.path.join(path, name_dict[c]))
#     for s in seqs:
#         with open(os.path.join(path, s), 'r') as f:
#             g = [list(map(float, x.strip().split(","))) for x in f]
#         g = np.array(g)
#         for c in range(1, len(name_dict)+1):
#             if g.shape[0] > 0:
#                 temp = g[g[:,-2]==c]
#                 temp[:, -3]=1
#             else:
#                 temp = g
#             np.savetxt(os.path.join(path, name_dict[c], s), np.c_[temp], fmt='%d', delimiter=',')

def split_cls_from_txt(path):
    name_dict = {1: "airplane", 2: "ship"}
    seqs = sorted([s for s in os.listdir(path) if not os.path.isdir(os.path.join(path, s))])
    for c in range(1, len(name_dict)+1):
        mkr(os.path.join(path, name_dict[c]))
    for s in seqs:
        with open(os.path.join(path, s), 'r') as f:
            g = [list(map(float, x.strip().split(","))) for x in f]
        g = np.array(g)
        for c in range(1,len(name_dict)+1):
            if g.shape[0] > 0:
                temp = g[g[:,7]==c]
                temp[:, 6]=1
            else:
                temp = g
            np.savetxt(os.path.join(path, name_dict[c], s), np.c_[temp], fmt='%d', delimiter=',')
            
if __name__ == '__main__':
    total = 0
    mota  = []
    cls_name = ['airplane','ship'] 
    split_cls_from_txt("/workspace/TrackEval/data/trackers/jl_challenge/MOT16-val/JLTrack/center")
    for c in cls_name:
        m = main('center/'+c, c)
        total += m
        mota.append(m)
    print("\n\n")
    print("total: {}".format(total/2.0))
    print("\n")
    
    for i in range(len(cls_name)):
        print(cls_name[i]+ ": %.4f" % mota[i])


