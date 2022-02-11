import sys
import numpy as np


seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


def convert_method_name(mth, algo_id):
    clf = 'rf' if algo_id == 'nas' else 'gp'
    if mth == 'ours-rs':
        return 'rgpe-space-%s-final-rs' % clf
    if mth == 'ours-gp' or mth == 'ours-v3':
        return 'rgpe-space-%s-final-gp' % clf
    if mth == 'ours-smac':
        return 'rgpe-space-%s-final-smac' % clf
    if mth in ['gp', 'smac']:
        return 'notl'

    if mth == 'ours-rgpe':
        return 'rgpe-space-%s-final' % clf
    if mth == 'ours-tst':
        return 'rgpe-space-%s-final-tst' % clf

    if mth == 'ours-v1-rs':
        return 'rgpe-space-%s-best-rs' % clf
    if mth == 'ours-v2-rs':
        return 'rgpe-space-%s-sample-rs' % clf
    if mth == 'ours-v3-rs':
        return 'rgpe-space-%s-final-rs' % clf
    if mth == 'ours-v1-gp':
        return 'rgpe-space-%s-best-gp' % clf
    if mth == 'ours-v2-gp':
        return 'rgpe-space-%s-sample-gp' % clf
    if mth == 'ours-v3-gp':
        return 'rgpe-space-%s-final-gp' % clf

    if mth == 'ours-knn':
        return 'rgpe-space-knn-final-gp'
    if mth == 'ours-svm':
        return 'rgpe-space-svm-final-gp'
    if mth == 'ours-rf':
        return 'rgpe-space-rf-final-gp'
    if mth == 'ours-gpc':
        return 'rgpe-space-gp-final-gp'

    return mth
