import os
import sys
import argparse
import traceback
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

sys.path.insert(0, '.')
from tools.utils import seeds

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=15.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 7    # 15
label_fontsize = 20

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_type', type=str, default='gp')
parser.add_argument('--exp_id', type=str, default='exptest')
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='rs')
parser.add_argument('--data_dir', type=str, default='./data/exp_results/')
parser.add_argument('--transfer_trials', type=int, default=100)
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--task_set', type=str, default='full')
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
args = parser.parse_args()

algo_id = args.algo_id
task_id = args.task_id
exp_id = args.exp_id
surrogate_type = args.surrogate_type
transfer_trials = args.transfer_trials
run_trials = args.trial_num
methods = args.methods.split(',')
data_dir = args.data_dir
task_set = args.task_set
rep = args.rep
start_id = args.start_id

if exp_id == 'exptest':
    data_dir = 'data/exp_results/main_random_full_19_50000'
elif exp_id == 'expresnet':
    data_dir = 'data/exp_results/main_random_full_2_2000'
elif exp_id == 'expnas':
    data_dir = 'data/exp_results/main_random_full_2_15625'
else:
    raise ValueError('Invalid exp id - %s.' % exp_id)

if exp_id == 'expresnet' and task_set == 'full':
    datasets = ['cifar-10', 'svhn', 'tiny-imagenet']
elif exp_id == 'expnas' and task_set == 'full':
    datasets = ['cifar-10', 'cifar-100', 'imagenet']
elif task_set == 'full':
    datasets = [
        'winequality_white', 'sick', 'page-blocks(2)', 'satimage',
        'segment', 'wind', 'delta_ailerons', 'abalone',
        'kc1', 'madelon', 'quake', 'musk', 'waveform-5000(2)',
        'space_ga', 'puma8NH', 'waveform-5000(1)', 'optdigits',
        'pollen', 'cpu_act', 'cpu_small',
    ]
else:
    datasets = task_set.split(',')
    print('datasets:', datasets)


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    names_dict = dict()
    method_ids = ['rs']
    method_names = ['Random']
    color_list = ['red', 'orchid', 'royalblue', 'brown', 'purple', 'orange', 'yellowgreen', 'navy', 'green', 'black']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', '+', 'H']

    undefined_cnt = 0

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]
        if name not in method_ids:
            names_dict[name] = name.replace('_', '\\_')
        else:
            names_dict[name] = method_names[method_ids.index(name)]

    for name in m_list:
        fill_values(name, undefined_cnt)
        print('fill', name, undefined_cnt)
        undefined_cnt = (undefined_cnt + 1) % len(color_list)
    return color_dict, marker_dict, names_dict, method_ids


def get_subplot_num(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 5
    color_dict, marker_dict, names_dict, method_ids = fetch_color_marker(methods)

    print(names_dict)
    method_list = list()
    _orders = list()
    for _method in methods:
        if _method in method_ids:
            _orders.append(method_ids.index(_method))
        else:
            _orders.append(0)
    print(methods)
    _methods = zip(methods, _orders)
    methods = [item[0] for item in sorted(_methods, key=lambda x: x[1])]
    print(methods)

    nx, ny = get_subplot_num(n=len(datasets))
    if len(datasets) <= 16:
        plt.figure(figsize=(4 * ny, 3 * nx))
    else:
        plt.figure(figsize=(3 * ny, 1.5 * nx))

    all_datasets_y = dict()
    for data_idx, dataset in enumerate(datasets):
        ax = plt.subplot(nx, ny, data_idx + 1)
        # fig, ax = plt.subplots()
        handles = list()

        for idx, method in enumerate(methods):
            all_data = []
            for rep_id in range(start_id, start_id + rep):
                seed = seeds[rep_id]
                filename = '%s_%s_%s_%d_%d_%s_%s_%d.pkl' % (
                    method, dataset, algo_id, transfer_trials, run_trials, surrogate_type, task_id, seed)
                path = os.path.join(data_dir, filename)
                with open(path, 'rb')as f:
                    data = pkl.load(f)
                all_data.append(data)

            mean_data = np.mean(all_data, axis=0)  # mean over repeats
            x = np.arange(mean_data.shape[0]) + 1
            y = mean_data[:, 0]
            if method not in all_datasets_y.keys():
                all_datasets_y[method] = []
            all_datasets_y[method].append(y)

            label_name = r'\textbf{%s}' % names_dict[method]
            lw = 2 if method in method_ids else 1
            # print(x, y)
            ax.plot(x, y, lw=lw,
                    label=label_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me
                    )

            line = mlines.Line2D([], [], lw=lw, color=color_dict[method], marker=marker_dict[method],
                                 markersize=ms, label=label_name)
            handles.append(line)

            item = np.round(mean_data[:, 1][-1], 6)  # last val perf

        if data_idx == 0:
            legend = ax.legend(handles=handles, loc=1, ncol=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        if len(datasets) <= 16:
            ax.set_xlabel('\\textbf{Number of Trials', fontsize=label_fontsize)
            ax.set_ylabel('\\textbf{NCE}', fontsize=label_fontsize)
            title_size = 16
        else:
            title_size = 10

        plt.title('[%s]-%s' % (args.algo_id.replace('_', '\\_'), dataset.replace('_', '\\_'),), fontsize=title_size)

    plt.tight_layout(pad=0.2)
    plt.show()

    for method, v in all_datasets_y.items():
        mean_y = np.mean(v, axis=0)  # mean over datasets
        x = np.arange(mean_y.shape[0]) + 1
        plt.plot(x, mean_y, label=r'\textbf{%s}' % names_dict[method], color=color_dict[method],
                 marker=marker_dict[method], markersize=ms, markevery=me)
    plt.legend()
    plt.title('[%s]' % (args.algo_id.replace('_', '\\_'), ))
    plt.xlabel('\\textbf{Number of Trials', fontsize=label_fontsize)
    plt.ylabel('\\textbf{Average NCE}', fontsize=label_fontsize)
    plt.tight_layout(pad=0.2)
    plt.show()
