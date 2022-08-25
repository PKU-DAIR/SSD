# Transfer Learning based Search Space Design for Hyperparameter Tuning

Transfer Learning based Search Space Design for Hyperparameter Tuning (SIGKDD'22)

## Citation

```
@inproceedings{10.1145/3534678.3539369,
author = {Li, Yang and Shen, Yu and Jiang, Huaijun and Bai, Tianyi and Zhang, Wentao and Zhang, Ce and Cui, Bin},
title = {Transfer Learning Based Search Space Design for Hyperparameter Tuning},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539369},
doi = {10.1145/3534678.3539369},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {967–977},
numpages = {11},
keywords = {search space design, transfer learning, bayesian optimization, hyperparameter optimization},
location = {Washington DC, USA},
series = {KDD '22}
}
```

## Experimental Environment Installation

1. preparations: Python == 3.7
2. install SWIG:
    ```
    apt-get install swig3.0
    ln -s /usr/bin/swig3.0 /usr/bin/swig
    ```
3. install requirements:
    ```
    cat requirements.txt | xargs -n 1 -L 1 pip install
    ```

## Data Preparation

We generate offline benchmarks for 3 tasks:
+ Random Forest
+ Resnet
+ Nas-Bench-201

The benchmark data is publicly available on Google Drive (<https://drive.google.com/file/d/1xKGPHMyXLbFHkMwgqkgNxnxTJ7-WPgvg/view>)

To use the data, please `unzip` the benchmark data file into `data/hpo_data/` inside this project.

# Documentations

## Project Code Overview

+ `tlbo/` : the implemented method and compared baselines.
+ `tools/` : the python scripts in the experiments, and useful tools.

## Experiments Design

### Baselines

+ Random: rs
+ GP: gp
+ Box + Random: box-rs
+ Box + GP: box-gp
+ Ellipsoid + Random: ellipsoid-rs
+ Ellipsoid + GP: ellipsoid-gp
+ Ours + Random: ours-rs
+ Ours + GP: ours-gp

### Exp: Compare methods on Random Forest

```
python tools/offline_benchmark.py --algo_id random_forest --num_source_trial 100 --trial_num 50 --methods rs,box-rs,ellipsoid-rs,ours-rs,gp,box-gp,ellipsoid-gp,ours-gp --rep 20
```

### Exp: Compare methods on Resnet

```
python tools/offline_benchmark.py --algo_id resnet --num_source_trial 100 --trial_num 50 --methods rs,box-rs,ellipsoid-rs,ours-rs,gp,box-gp,ellipsoid-gp,ours-gp --rep 20
```

### Exp: Compare methods on Nas-Bench-201

```
python tools/offline_benchmark.py --algo_id nas --num_source_trial 50 --trial_num 50 --methods rs,gp,smac,ours-rs,ours-gp,ours-smac --rep 20
```

### Case study on universality and safeness

Universality:

```
python tools/offline_benchmark.py --algo_id random_forest --num_source_trial 100 --trial_num 50 --methods rs,gp,rgpe,tst,ours-rgpe,ours-tst --rep 20
```

Safeness:

```
python tools/offline_benchmark.py --algo_id random_forest --num_source_trial 100 --trial_num 200 --methods gp,box-gp,ellipsoid-gp,ours-gp --rep 20
```

## Ablation study

Promising Region Extraction:

```
python tools/offline_benchmark.py --algo_id random_forest --num_source_trial 100 --trial_num 50 --methods ours-knn,ours-svm,ours-rf,ours-gpc --rep 20
```

Target Search Space Generation:

```
python tools/offline_benchmark.py --algo_id random_forest --num_source_trial 100 --trial_num 50 --methods ours-v1-rs,ours-v2-rs,ours-v3-rs,ours-v1-gp,ours-v2-gp,ours-v3-gp --rep 20
```
