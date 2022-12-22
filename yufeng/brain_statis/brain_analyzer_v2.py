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
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import seaborn as sns

import hdbscan
import umap

from file_io import load_image
from anatomy.anatomy_config import REGION671
from anatomy.anatomy_core import parse_id_map, parse_ana_tree, parse_regions316

from common_func import load_regions, get_region_mapper

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
        self.load_meta()

    #@deprecated
    def calc_left_right_corr(self, distr_dir):
        merged_distr = {}
        for distr_file in sorted(glob.glob(os.path.join(distr_dir, '[1-9]*csv'))):
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
            plt.close()

    def load_meta(self):
        # initialize
        print('Parsing the id mapping file')
        self.mapper, self.rev_mapper = parse_id_map()
        self.ana_dict = parse_ana_tree(keyname='id')


    def parse_distrs(self, distr_dir, ignore_lr=True):
        dfiles = sorted(glob.glob(os.path.join(distr_dir, '*', '*csv')), key=lambda x: os.path.splitext(os.path.split(x)[-1])[0])
        brains = [os.path.splitext(os.path.split(distr_file)[-1])[0] for distr_file in dfiles]
        
        #ids = list(self.rev_mapper.keys())   # ids is the u32 key
        if ignore_lr:
            ids = REGION671[:]
        else:
            raise ValueError
        ids_set = set(ids)

        nids = len(ids)
        nfiles = len(dfiles)
        df = pd.DataFrame(np.zeros((nfiles, nids+1)), columns=ids+['modality'], index=brains)
        print(df.shape)

        print('Loading distribution files...')
        for ifile, distr_file in enumerate(dfiles):
            brain_id = os.path.splitext(os.path.split(distr_file)[-1])[0]
            modality = os.path.split(os.path.split(distr_file)[0])[-1]
            #df['modality'][brain_id] = modality
            df.loc[brain_id, 'modality'] = modality  # at is faster than `loc`, but only for integar

            if modality == 'fMOST-Zeng':
                res_id = -3
            elif modality == 'fMOST-Huang':
                res_id = -3
            elif modality == 'LSFM-Osten':
                res_id = -3
            elif modality == 'LSFM-Wu':
                res_id = -1

            bsa = BrainSignalAnalyzer(res_id=res_id, region_file=distr_file)
        
            rarr = bsa.region_distr.to_numpy()
            for seg_id, count in rarr:
                sid = self.mapper[seg_id]    # seg_id: u16 region, sid: u32 region
                if sid not in ids_set:
                    continue
                df.at[brain_id, sid] += count
        
        # remove the empty regions
        #print(f'Original shape: {df.shape}')
        #df.drop([col for col, val in df.sum().iteritems() if val == 0], axis=1, inplace=True)
        print(f'Non-empty df shape: {df.shape}')
        print(df.sum())
        return df

    def plot_region_distrs_modalities(self, distr_dir):
        df = self.parse_distrs(distr_dir)
        # normalize
        df_d = df.loc[:, df.columns!='modality']
        df.loc[:, df.columns!='modality'] = df_d.div(df_d.sum(axis=1), axis=0)

        mods = np.unique(df['modality'])
        nmod = len(mods)
        fig, axes = plt.subplots(nmod, 1, sharex=True, sharey=True)
        for i in range(nmod):
            df_c = df.loc[df['modality'] == mods[i]].mean()
            axes[i].plot(np.arange(len(df_c)), df_c.to_numpy(), label=mods[i])
            axes[i].set_ylim(0, 0.1)
            axes[i].set_xlim(0, df_d.shape[1]-1)
            axes[i].legend(loc='upper left')

        axes[3].set_xticks([])
        axes[3].set_xlabel('Brain region', fontsize=15)
        fig.text(0.04, 0.5, 'Foreground ratio (%)', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.savefig('distr_modality.png', dpi=200)
        plt.close()

    def plot_region_distrs_labeling(self, distr_dir, label_file):
        df = self.parse_distrs(distr_dir)
        df = self.convert_modality_to_label(df, label_file, normalize=True)
        
        uni_l, counts_l = np.unique(labels['label'].to_numpy(), return_counts=True)
        argmax_cs = np.argpartition(counts_l, -4)[-4:][::-1]    # top4
        for argmax_c in argmax_cs:
            max_l = uni_l[argmax_c]
            max_c = counts_l[argmax_c]
            print(f'Labeling method {max_l} with count {max_c}')

            # plot instance distribution of each labeled brains
            df_sub = df[df['label'] == max_l].drop(['label'], axis=1)
            
            fig, axes = plt.subplots(4,1, sharex=True, sharey=True)
            cnt = 0
            for irow, row in df_sub.iterrows():
                row_arr = row.to_numpy()
                axes[cnt].plot(np.arange(df_sub.shape[1]), row_arr)
                axes[cnt].set_ylim(0, 0.1)
                axes[cnt].set_xlim(0, df_sub.shape[1]-1)
                #axes[cnt].legend(loc='upper right')

                if cnt >= 3: break
                cnt += 1

            axes[cnt].set_xticks([])
            axes[cnt].set_xlabel('Brain region', fontsize=15)
            fig.text(0.04, 0.5, 'Foreground ratio (%)', ha='center', va='center', rotation='vertical', fontsize=15)
            plt.suptitle(max_l, y=0.93, fontsize=18)
            plt.savefig(f'distr_label_{max_l.replace(";", "_")}.png', dpi=200)
            plt.close()

        

    def convert_modality_to_label(self, df, label_file, normalize=True):
        # normalize
        df_d = df.loc[:, df.columns!='modality']
        if normalize:
            df.loc[:, df.columns!='modality'] = df_d.div(df_d.sum(axis=1), axis=0)
        df.rename(columns={'modality': 'label'}, inplace=True)
        df.loc[:, df.columns == 'label'] = ''

        # load the labeling file of fMOST-Zeng
        labels = pd.read_csv(label_file, index_col=0)
        labels.index = labels.index.map(str)
        for irow, row in labels.iterrows():
            df.at[irow, 'label'] = row['label']
        return df
      

    def map_to_coarse_regions(self, df, level=1, last_column='modality'):
        """
        :params level: level==0 means use the original 316 regions, otherwise use parental 70 regions
        """
        rids_set = load_regions(level=level)
        # The set should be consensus with original 671

        # regions mapping
        orig_ids = [idx for idx in df.columns if idx != last_column]
        rc_dict = get_region_mapper(rids_set, orig_ids, self.ana_dict)
        nr = len(rc_dict)
        rids = sorted(rc_dict.keys())
        
        print(len(rc_dict), len(rids))
        ndf = pd.DataFrame(np.zeros((df.shape[0], nr+1)), columns=rids+[last_column], index=df.index)
        ndf[last_column] = df[last_column].copy()
        for nidx in rc_dict.keys():
            for idx in rc_dict[nidx]:
                ndf[nidx] = ndf[nidx] + df[idx]

        # remove zero regions
        #ndf.drop([col for col, val in ndf.sum().iteritems() if val == 0], axis=1, inplace=True)
        print(f'New data shape[{ndf.shape}] from original [{df.shape}]')

        return ndf
        

    def corr_clustermap(self, distr_dir):
        df = self.parse_distrs(distr_dir)
        df = self.map_to_coarse_regions(df, level=1)

        plot_corr = True
        if self.plot and plot_corr:
            print('Plotting')

            rmapper = {}
            for idx in df.columns:
                if idx != 'modality':
                    rmapper[idx] = self.ana_dict[idx]['acronym']


            df_corr = df.drop(['modality'], axis=1).rename(columns=rmapper)
            corr = df_corr.corr(min_periods=10)
            corr[corr < 0] = 0

            print(corr.mean().mean(), corr.max().min(), corr.min().min())
            corr = corr.fillna(0)
            clust_map = sns.clustermap(corr, cmap='coolwarm')

            names = df_corr.columns
            rids = df.columns[:-1]
            for i in range(corr.shape[0]):
                ind = clust_map.dendrogram_col.reordered_ind[i]
                region_name = names[ind]
                region_id = rids[ind]
                print(f'[{i}]{region_name}', end=': ')
                for r in self.ana_dict[region_id]['structure_id_path']:
                    print(self.ana_dict[r]['acronym'], end=', ')
                print('')
            plt.xticks([])
            plt.yticks([])
            #plt.xlabel('Brain region ID')
            #plt.ylabel('Brain region ID')
            #plt.title("Correlation coefficients between labeled regions")
            plt.savefig('region_corr.png', dpi=300)
            plt.close()
        

        
        

if __name__ == '__main__':
    import pickle

    distr_dir = '/home/lyf/Research/cloud_paper/brain_statistics/statis_out/statis_out_adaThr_all'
    dim_file = './ccf_info/TeraDownsampleSize.csv'
    mask_dir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/registration/Inverse'
    label_file = './fMOST-Zeng_labels_edited.csv'
    res_id = -3
    plot = True

    bssa = BrainsSignalAnalyzer(res_id=res_id, plot=True)

    if 0:
        #bssa.plot_region_distrs_modalities(distr_dir)
        bssa.plot_region_distrs_labeling(distr_dir, label_file)

    if 0:
        bssa.calc_left_right_corr(distr_dir)

    if 1:
        bssa.corr_clustermap(distr_dir)
    
        


