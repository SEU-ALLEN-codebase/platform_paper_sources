#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : run_arbor_r2.py
#   Author       : Yufeng Liu
#   Date         : 2022-10-14
#   Description  : 
#
#================================================================

import os
import glob
import pandas as pd
from multiprocessing.pool import Pool

import sys
sys.path.append('../../common_lib')
from common_utils import load_type_from_excel


def run_autoarbor(*args):
    swc_file, low, high = args
    os.system(f'python autoarbor_v1_yf.py --filename {swc_file} --L {low} --H {high}')


celltype_file = '../../common_lib/41586_2021_3941_MOESM4_ESM.csv'
soma_types, soma_types_r, p2stypes = load_type_from_excel(celltype_file, keep_Car3=True)

# rerun arborization with estimated arbor num
swc_dir = '../data/axon_sort'

'''
pdict = {}
params = pd.read_csv(params_file)
for param in params.iterrows():
    print(param[1])
    ptype = param[1]['proj_type']
    stype = param[1]['soma_type']
    narbor = param[1]['median_arbor_num']
    pdict[(ptype, stype)] = narbor
'''

args_list = []
for stype in soma_types:
    print(f'<------------------- soma type: {stype} --------------------->')
    prefixs = soma_types[stype]
    #na = pdict[(ptype, stype)]
    na = 2

    for prefix in prefixs:
        print(f'===> {prefix}')
        swc_file = os.path.join(swc_dir, f'{prefix}_axon.swc')
        if os.path.exists(f'{swc_file}._m3_l.eswc'):
            continue
        args_list.append((swc_file, na, na))
    
# do multiprocessing processing
nprocessors = 10
pt = Pool(nprocessors)
pt.starmap(run_autoarbor, args_list)
pt.close()
pt.join()
