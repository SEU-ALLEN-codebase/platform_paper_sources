#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : brain_analyzer.py
#   Author       : Yufeng Liu
#   Date         : 2022-10-31
#   Description  : 
#
#================================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from file_io import load_image
from anatomy.anatomy_core import parse_id_map, parse_ana_tree

def load_region_distr(distr_file, remove_zero=True):
    """
    Args:
        - remove_zero: remove the counts for brain_seg index 0
    """

    distr = pd.read_csv(distr_file)
    if remove_zero:
        # remove statis with id == 0
        distr = distr[distr.region != 0]

    return distr

class BrainSignalAnalyzer(object):
    def __init__(self, res_id, region_file):
        assert (res_id < 0)
        self.res_ds = np.power(2, np.fabs(res_id)-1)
        self.region_distr = load_region_distr(region_file)

    def load_mask(self, mask_file):
        self.mask = load_image(mask_file)
        
    def set_orig_dims(self, orig_dims):
        self.orig_dims = np.array(orig_dims, dtype=np.int32)

    def set_mask_dims(self, mask_dims):
        self.mask_dims = np.array(mask_dims, dtype=np.int32)

    def multiplizer_cur2mask(self):
        return 1.0 * self.orig_dims / self.res_ds / self.mask_dims

    def calc_signal_ratio(self):
        num_sig_voxels = self.region_distr['count'].sum()
        multiplizer = self.multiplizer_cur2mask()
        ratio = num_sig_voxels / np.count_nonzero(self.mask) / np.prod(multiplizer)
        return ratio

class BrainsSignalAnalyzer(object):
    def __init__(self, res_id=-3, plot=True):
        self.res_id = res_id
        self.plot = plot

    def calc_signal_ratio(self, distr_dir, dim_file, mask_dir):
        dim_f = pd.read_csv(dim_file, index_col='ID', sep='\t')

        sr_dict = {}
        for distr_file in glob.glob(os.path.join(distr_dir, '[1-9]*.csv')):
            brain_id = int(os.path.splitext(os.path.split(distr_file)[-1])[0])
            
            max_res_dims = np.array([dim_f.loc[brain_id][0],dim_f.loc[brain_id][1],dim_f.loc[brain_id][2]])
            mask_dims = np.array([dim_f.loc[brain_id][3],dim_f.loc[brain_id][4],dim_f.loc[brain_id][5]])
            bsa = BrainSignalAnalyzer(res_id=self.res_id, region_file=distr_file)
            bsa.set_orig_dims(max_res_dims)
            bsa.set_mask_dims(mask_dims)

            mask_file = os.path.join(mask_dir, f'{brain_id}.v3draw')
            bsa.load_mask(mask_file)
            ratio = bsa.calc_signal_ratio()
            sr_dict[brain_id] = ratio

            print(f'{brain_id}: {ratio:.10f}')
        
        return sr_dict

    def plot_signal_ratio(self, sr_dict):
        srs = list(sr_dict.values())
        sr_df = pd.DataFrame(srs, columns=['ratio'])
        plt.figure(figsize=(6,6))
        sns.violinplot(x='ratio', data=sr_df, orient='v')
        plt.savefig('signal_ratio_distr.png', dpi=300)

    def region_distrs(self, distr_dir):
        merged_distr = {}
        for distr_file in glob.glob(os.path.join(distr_dir, '[1-9]*csv')):
            brain_id = int(os.path.splitext(os.path.split(distr_file)[-1])[0])
            bsa = BrainSignalAnalyzer(res_id=self.res_id, region_file=distr_file)
            
            darr = bsa.region_distr.to_numpy()
            for di in darr:
                if di[0] in merged_distr:
                    merged_distr[di[0]] += di[1]
                else:
                    merged_distr[di[0]] = di[1]

        sorted_distr = sorted(merged_distr.items(), key=lambda x:x[1], reverse=True)

        plot_lr = True
        if self.plot and plot_lr:
            # check equality of left-right regions
            lr_pairs = []
            mapper, rev_mapper = parse_id_map()
            for orig_id, cur_ids in rev_mapper.items():
                if len(cur_ids) == 2:
                    ids = sorted(cur_ids)
                    pair = []
                    for cur_id in ids:
                        if cur_id in merged_distr:
                            pair.append(merged_distr[cur_id])
                    if len(pair) == 2:
                        lr_pairs.append(pair)
            lr_pairs = np.array(lr_pairs)
            lr_df = pd.DataFrame(lr_pairs, columns=['left', 'right'])
            plt.figure(figsize=(6,6))
            g = sns.regplot(x="left", y="right", data=lr_df,
                robust=True,
                y_jitter=.02, scatter_kws={'s':10})

            plt.xlim([0,6e6])
            plt.ylim([0,6e6])
            plt.grid(which='major', alpha=0.5, linestyle='--')
            plt.xlabel('Labeled voxels of left-hemispheric region')
            plt.ylabel('Labeled voxels of right-hemispheric region')
            plt.savefig('left_right_distr_corr.png', dpi=300)

    def labeling_corr(self, distr_dir):
        dfiles = list(glob.glob(os.path.join(distr_dir, '[1-9]*csv')))

        # initialize
        mapper, rev_mapper = parse_id_map()
        ids = list(rev_mapper.keys())
        ids2ind = dict(zip(ids, range(len(ids))))

        darr = []
        for distr_file in dfiles:
            brain_id = int(os.path.splitext(os.path.split(distr_file)[-1])[0])
            bsa = BrainSignalAnalyzer(res_id=self.res_id, region_file=distr_file)
        
            rarr = bsa.region_distr.to_numpy()
            di = np.zeros(len(ids))
            for seg_id, count in rarr:
                sid = mapper[seg_id]
                di[ids2ind[sid]] = count
            darr.append(di)
        darr = np.array(darr)
        # remove empty regions
        darr_sum0 = darr.sum(axis=0)
        nz_sum = np.nonzero(darr_sum0)[0]

        darr_nz = darr[:,nz_sum]
        mean = darr_sum0[nz_sum].mean()
        nz_mean = np.nonzero(darr_sum0 > mean / 2)[0]
        
        c_thresh = 50
        nz_count = np.nonzero((darr > 0).sum(axis=0) > c_thresh)[0]
        nz = np.array(sorted(list(set(nz_sum.tolist()) & set(nz_mean.tolist()) & set(nz_count.tolist()))))
        
        darr = darr[:,nz]
        ids = np.array(ids)[nz]

        df = pd.DataFrame(darr, columns=ids)
        corr = df.corr()
        sns.heatmap(corr, cmap='coolwarm')
        plt.savefig('region_corr.png', dpi=300)
        
        return corr

        
        

if __name__ == '__main__':
    import pickle

    distr_dir = './statis_out_adaThr'
    dim_file = './ccf_info/TeraDownsampleMap.csv'
    mask_dir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/registration/Inverse'
    res_id = -3
    plot = True

    bssa = BrainsSignalAnalyzer(res_id=res_id, plot=True)

    if 0:
        # overal signal ratio distribution
        sr_file = 'signal_ratio_distr.pkl'
        if os.path.exists(sr_file):
            with open(sr_file, 'rb') as fp:
                sr_dict = pickle.load(fp)
        else:
            sr_dict = bssa.calc_signal_ratio(distr_dir, dim_file, mask_dir)
            with open(sr_file, 'wb') as fp:
                pickle.dump(sr_dict, fp)
        bssa.plot_signal_ratio(sr_dict)

    if 0:
        bssa.region_distrs(distr_dir)

    if 1:
        bssa.labeling_corr(distr_dir)
    
        


