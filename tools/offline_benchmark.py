import os

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS  # export NUMEXPR_NUM_THREADS=1

import re
import sys
import time
import pickle
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm, trange

sys.path.insert(0, '.')
from tlbo.facade.notl import NoTL
from tlbo.facade.rgpe import RGPE
from tlbo.facade.tst import TST
from tlbo.facade.rgpe_space import RGPESPACE, RGPESPACE_BO, RGPESPACE_RS, RGPESPACE_TST
from tlbo.facade.tst_space import TSTSPACE
from tlbo.facade.random_surrogate import RandomSearch
from tlbo.framework.smbo_offline import SMBO_OFFLINE
from tlbo.framework.smbo_baseline import SMBO_SEARCH_SPACE_Enlarge
from tlbo.config_space.space_instance import get_configspace_instance

from tools.utils import seeds, convert_method_name

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--exp_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest', choices=['random_forest', 'resnet', 'nas'])
parser.add_argument('--methods', type=str, default='rs')
parser.add_argument('--surrogate_type', type=str, default='gp')
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--init_num', type=int, default=0)
# parser.add_argument('--run_num', type=int, default=-1)
parser.add_argument('--num_source_trial', type=int, default=100)
parser.add_argument('--num_source_problem', type=int, default=-1)
parser.add_argument('--task_set', type=str, default='full')
parser.add_argument('--target_set', type=str, default='full')
parser.add_argument('--num_source_data', type=int, default=10000)
parser.add_argument('--num_random_data', type=int, default=50000)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

default_pmin, default_pmax = 5, 50
parser.add_argument('--pmin', type=int, default=default_pmin)
parser.add_argument('--pmax', type=int, default=default_pmax)
args = parser.parse_args()

algo_id = args.algo_id
exp_id = args.exp_id
task_id = args.task_id
task_set = args.task_set
targets = args.target_set
surrogate_type = args.surrogate_type
n_src_trial = args.num_source_trial
num_source_problem = args.num_source_problem
n_source_data = args.num_source_data
num_random_data = args.num_random_data
trial_num = args.trial_num
init_num = args.init_num
# run_num = args.run_num
test_mode = 'random'
baselines = args.methods.split(',')
rep = args.rep
start_id = args.start_id

pmin = args.pmin
pmax = args.pmax

data_dir = 'data/hpo_data/'
assert test_mode in ['random']
if init_num > 0:
    enable_init_design = True
else:
    enable_init_design = False
    # Default number of random configurations.
    init_num = 3

algorithms = ['random_forest', 'resnet', 'nas']
algo_str = '|'.join(algorithms)
src_pattern = '(.*)-(%s)-(\d+).pkl' % algo_str

if algo_id == 'nas':
    n_source_data = 250
    num_random_data = 15625
elif algo_id == 'resnet':
    n_source_data = 200
    num_random_data = 2000
else:
    n_source_data = 10000
    num_random_data = 50000


def get_data_set(set_name):
    if algo_id == 'resnet':
        return ['cifar-10', 'svhn', 'tiny-imagenet']

    elif algo_id == 'nas':
        return ['cifar-10', 'cifar-100', 'imagenet']

    elif set_name == 'full':
        data_set = [
            'winequality_white', 'sick', 'page-blocks(2)', 'satimage',
            'segment', 'wind', 'delta_ailerons', 'abalone',
            'kc1', 'madelon', 'quake', 'musk', 'waveform-5000(2)',
            'space_ga', 'puma8NH', 'waveform-5000(1)', 'optdigits',
            'pollen', 'cpu_act', 'cpu_small',
        ]
    else:
        raise ValueError(set_name)
    return data_set


def load_hpo_history():
    source_hpo_ids, source_hpo_data = list(), list()
    random_hpo_data = list()
    for _file in tqdm(sorted(os.listdir(data_dir))):
        if _file.endswith('.pkl') and _file.find(algo_id) != -1:
            result = re.search(src_pattern, _file, re.I)
            if result is None:
                continue
            dataset_id, algo_name, total_trial_num = result.group(1), result.group(2), result.group(3)
            if int(total_trial_num) != n_source_data:
                continue
            with open(os.path.join(data_dir, _file), 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
            p_max, p_min = np.max(perfs), np.min(perfs)
            if p_max == p_min:
                continue
            if (perfs == perfs[0]).all():
                continue
            if test_mode == 'random':
                _file = data_dir + '%s-%s-random-%d.pkl' % (dataset_id, algo_id, num_random_data)
                if not os.path.exists(_file):
                    continue
            source_hpo_ids.append(dataset_id)

            if test_mode == 'bo':
                raise NotImplementedError('TODO: Add test perf')
            if perfs.ndim == 2:
                assert perfs.shape[1] == 2
                _data = {k: v[0] for k, v in data.items()}
            else:
                _data = data
            source_hpo_data.append(_data)

    assert len(source_hpo_ids) == len(source_hpo_data)
    print('Load %s source hpo problems for algorithm %s.' % (len(source_hpo_ids), algo_id))

    # Load random hpo data to test the transfer performance.
    if test_mode == 'random':
        for id, hpo_id in tqdm(list(enumerate(source_hpo_ids))):
            _file = data_dir + '%s-%s-random-%d.pkl' % (hpo_id, algo_id, num_random_data)
            with open(_file, 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
                p_max, p_min = np.max(perfs), np.min(perfs)
                if p_max == p_min:
                    raise ValueError('The same perfs found in the %d-th problem' % id)
                    # data = source_hpo_data[id].copy()
                random_hpo_data.append(data)

    return source_hpo_ids, source_hpo_data, random_hpo_data


def extract_data(task_set):
    if task_set == 'full':
        hpo_ids, hpo_data, random_test_data = load_hpo_history()
    else:
        dataset_ids = get_data_set(task_set)

        hpo_ids, hpo_data, random_test_data = list(), list(), list()
        hpo_ids_, hpo_data_, random_test_data_ = load_hpo_history()
        for _idx, _id in enumerate(hpo_ids_):
            if _id in dataset_ids:
                hpo_ids.append(hpo_ids_[_idx])
                hpo_data.append(hpo_data_[_idx])
                random_test_data.append(random_test_data_[_idx])
    return hpo_ids, hpo_data, random_test_data


if __name__ == "__main__":
    hpo_ids, hpo_data, random_test_data = extract_data(task_set)
    algo_name = algo_id
    config_space = get_configspace_instance(algo_id=algo_name)
    num_source_problem = (len(hpo_ids) - 1) if num_source_problem == -1 else num_source_problem

    run_id = list()
    if targets in ['full']:
        targets = get_data_set(targets)
    else:
        targets = targets.split(',')
    for target_id in targets:
        target_idx = hpo_ids.index(target_id)
        run_id.append(target_idx)

    # Exp folder to save results.
    exp_dir = 'data/exp_results/%s_%s_%s_%d_%d/' % (exp_id, test_mode, task_set, num_source_problem, num_random_data)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    pbar = tqdm(total=rep * len(baselines) * len(run_id) * trial_num)
    for rep_id in range(start_id, start_id + rep):
        for id in run_id:
            for mth in baselines:
                seed = seeds[rep_id]
                print('=== start rep', rep_id, 'seed', seed)

                print('=' * 20)
                print('[%s-%s] Evaluate %d-th problem - %s[%d].' % (algo_id, mth, id + 1, hpo_ids[id], rep_id))
                pbar.set_description('[%s-%s] %d-th - %s[%d]' % (algo_id, mth, id + 1, hpo_ids[id], rep_id))
                start_time = time.time()

                # Generate the source and target hpo data.
                source_hpo_data = list()
                if test_mode == 'bo':
                    raise NotImplementedError
                else:
                    target_hpo_data = random_test_data[id]
                for _id, data in enumerate(hpo_data):
                    if _id != id:
                        source_hpo_data.append(data)

                # Select a subset of source problems to transfer.
                rng = np.random.RandomState(seed)
                shuffled_ids = np.arange(len(source_hpo_data))
                rng.shuffle(shuffled_ids)
                source_hpo_data = [source_hpo_data[id] for id in shuffled_ids[:num_source_problem]]

                mth = convert_method_name(mth, algo_id)

                if mth == 'rgpe':
                    surrogate_class = RGPE
                elif mth == 'notl':
                    surrogate_class = NoTL
                elif mth == 'tst':
                    surrogate_class = TST
                elif mth == 'rs':
                    surrogate_class = RandomSearch
                elif mth.startswith('rgpe-space'):
                    if 'gp' in mth or 'smac' in mth:
                        surrogate_class = RGPESPACE_BO
                    elif 'rs' in mth:
                        surrogate_class = RGPESPACE_RS
                    elif 'tst' in mth:
                        surrogate_class = RGPESPACE_TST
                    else:
                        surrogate_class = RGPESPACE     # rgpe
                elif mth.startswith('tst-space'):
                    surrogate_class = TSTSPACE
                elif mth in ['box-gp', 'ellipsoid-gp']:
                    surrogate_class = NoTL  # BO
                elif mth in ['box-rs', 'ellipsoid-rs']:
                    surrogate_class = RandomSearch
                else:
                    raise ValueError('Invalid baseline name - %s.' % mth)

                if 'smac' in mth:
                    surrogate_type = 'rf'
                else:
                    surrogate_type = 'gp'

                print('surrogate_class:', surrogate_class.__name__)
                surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data,
                                            seed=seed,
                                            surrogate_type=surrogate_type,
                                            num_src_hpo_trial=n_src_trial)

                if 'rf' in mth:
                    model = 'rf'
                elif 'knn' in mth:
                    model = 'knn'
                elif 'svm' in mth:
                    model = 'svm'
                else:
                    model = 'gp'

                if 'final' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all+-sample+-threshold', model=model)
                elif 'sample' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='sample', model=model)
                elif 'best' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='best', model=model)
                elif 'box' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='box', model=model)
                elif 'ellipsoid' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='ellipsoid', model=model)
                else:
                    smbo_framework = SMBO_OFFLINE

                smbo = smbo_framework(target_hpo_data, config_space, surrogate,
                                      random_seed=seed, max_runs=trial_num,
                                      source_hpo_data=source_hpo_data,
                                      num_src_hpo_trial=n_src_trial,
                                      surrogate_type=surrogate_type,
                                      enable_init_design=enable_init_design,
                                      initial_runs=init_num,
                                      acq_func='ei')

                if hasattr(smbo, 'p_min'):
                    smbo.p_min = pmin
                    smbo.p_max = pmax
                smbo.use_correct_rate = True

                result = list()
                for _iter_id in range(trial_num):
                    config, _, perf, _ = smbo.iterate()
                    time_taken = time.time() - start_time
                    nce, y_inc = smbo.get_nce(), smbo.get_inc_y()
                    result.append([nce, y_inc, time_taken])
                    pbar.update(1)
                print('In %d-th problem: %s' % (id, hpo_ids[id]), 'nce, y_inc', result[-1])
                print('min/max', smbo.y_min, smbo.y_max)
                print('mean,std', np.mean(smbo.ys), np.std(smbo.ys))

                mth_file = '%s_%s_%s_%d_%d_%s_%d.pkl' % (
                    mth, hpo_ids[id], algo_id, n_src_trial, trial_num, task_id, seed)
                with open(os.path.join(exp_dir, mth_file), 'wb') as f:
                    data = np.array(result)
                    pickle.dump(data, f)
    pbar.close()
