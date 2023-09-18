###python3 get_swa_model.py work_dirs/swin_tiny/ 8000 80000 work_dirs/swin_tiny/
## python get_swa_model.py /workspace/CenterTrack/exp/tracking/JL_ch_easyAtt 8000 80000 1
import os
from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'model_dir', help='the directory where checkpoints are saved')
    parser.add_argument(
        'starting_model_id',
        type=int,
        help='the id of the starting checkpoint for averaging, e.g. 1')
    parser.add_argument(
        'ending_model_id',
        type=int,
        help='the id of the ending checkpoint for averaging, e.g. 12')
    parser.add_argument(
        'step',
        type=int,
        help='the directory for saving the SWA model')
    args = parser.parse_args()

    model_dir = args.model_dir
    starting_id = int(args.starting_model_id)
    ending_id = int(args.ending_model_id)
    step = int(args.step)
    model_names = ['last', 'inf']# list(range(starting_id, ending_id + 1, step))
    model_dirs = [
        os.path.join(model_dir, 'model_' + str(i) + '.pth')
        for i in model_names
    ]
    print(model_dirs)
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()
    state_dict = models[-1]['state_dict']
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model['state_dict'] = new_state_dict

    save_dir = os.path.join(model_dir, "swa_final.pth")
    torch.save(ref_model, save_dir)
    print('Model is saved at', save_dir)

def check_same():
    parser = ArgumentParser()
    parser.add_argument(
        'model_dir', help='the directory where checkpoints are saved')
    parser.add_argument(
        'starting_model_id',
        type=int,
        help='the id of the starting checkpoint for averaging, e.g. 1')
    parser.add_argument(
        'ending_model_id',
        type=int,
        help='the id of the ending checkpoint for averaging, e.g. 12')
    parser.add_argument(
        'step',
        type=int,
        help='the directory for saving the SWA model')
    args = parser.parse_args()

    model_dir = args.model_dir
    model_names = ['last', 'inf']  # list(range(starting_id, ending_id + 1, step))
    model_dirs = [
        os.path.join(model_dir, 'model_' + str(i) + '.pth')
        for i in model_names
    ]
    print(model_dirs)
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()


    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        if torch.sum(abs(avg_weight - models[0]['state_dict'][key])) < 0.00001:
            print(key, "   same")
        else:
            print(key, "   diff")

if __name__ == '__main__':
    # main()
    check_same()