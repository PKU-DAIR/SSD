# Transfer Learning based Search Space Design for Hyperparameter Tuning

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

**Note:** For reproduction purposes, we also upload the benchmark data
(e.g., evaluation results and the corresponding scripts) along with this submission.
The benchmark data (with size – 90.9MB) exceeds the space limit (maximum 20Mb) on CMT3,
so we only upload the data of Nas-Bench-201.
To remain anonymous, we will make the complete benchmark publicly available on Google Drive after revision.

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
