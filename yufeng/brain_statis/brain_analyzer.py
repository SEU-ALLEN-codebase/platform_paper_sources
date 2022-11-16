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

    #@deprecated
    def calc_signal_ratio(self, distr_dir, dim_file, mask_dir):
        dim_f = pd.read_csv(dim_file, index_col='ID', sep=',')

        sr_dict = {}
        for distr_file in sorted(glob.glob(os.path.join(distr_dir, '[1-9]*.csv'))):
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

    #@deprecated
    def plot_signal_ratio(self, sr_dict):
        srs = sorted(sr_dict.items(), key=lambda x: x[0])
        srs = [[i, v[1]] for i, v in enumerate(srs)]
        sr_df = pd.DataFrame(srs, columns=['index', 'ratio'])#.set_index('brainID')
        plt.figure(figsize=(6,1))
        sns.lineplot(data=sr_df, x='index', y='ratio', )
        plt.xlim(sr_df['index'].max(), sr_df['index'].min())
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig('signal_ratio_distr.png', dpi=300)
        plt.close()

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

    def labeling_corr(self, distr_dir, neighbor_file='/home/lyf/Softwares/installation/pylib/anatomy/resources/regional_neighbors_res25_radius5.pkl'):
        dfiles = sorted(glob.glob(os.path.join(distr_dir, '*', '*csv')), key=lambda x: os.path.splitext(os.path.split(x)[-1])[0])

        # initialize
        mapper, rev_mapper = parse_id_map()
        ana_dict = parse_ana_tree(keyname='id')

        ids = list(rev_mapper.keys())
        ids2ind = dict(zip(ids, range(len(ids))))

        darr = []
        modalities = []
        for distr_file in dfiles:
            brain_id = os.path.splitext(os.path.split(distr_file)[-1])[0]
            modality = os.path.split(os.path.split(distr_file)[0])[-1]
            modalities.append(modality)
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
            di = np.zeros(len(ids))
            for seg_id, count in rarr:
                sid = mapper[seg_id]
                #if count < 20:
                #    count = 0
                di[ids2ind[sid]] = count
            darr.append(di)
        darr = np.array(darr)

        # remove empty regions
        darr_sum0 = darr.sum(axis=0)
        nz_sum = np.nonzero(darr_sum0)[0]

        darr_nz = darr[:,nz_sum]
        ids_sum_nz = np.array(ids)[nz_sum]
        
        df_nz = pd.DataFrame(darr_nz, columns=ids_sum_nz)

        if self.plot:
            # get the level3 brain regions
            bregs3 = []
            for idx in ids_sum_nz:
                sip = ana_dict[idx]['structure_id_path']
                if len(sip) < 3:
                    bregs3.append(ana_dict[sip[-1]]['acronym'])
                else:
                    bregs3.append(ana_dict[sip[2]]['acronym'])

            normalize = False
            if normalize:
                df_nz_copy = df_nz.div(df_nz.sum(axis=1)+1e-6, axis=0)
            else:
                df_nz_copy = df_nz.copy()

            # echo the commonly most abundent labeled regions
            sort_type = 'number-of-high' # total
            if sort_type == 'total':
                df_sum = df_nz_copy.sum()
                high_reg_ids = df_sum[df_sum > 1].index
                
            elif sort_type == 'number-of-high':
                df_sum = (df_nz_copy > 0.02).sum()
                high_reg_ids = df_sum[df_sum > 10].index

            for idx in high_reg_ids:
                print(f"[{df_sum[idx]:.2f}] {ana_dict[idx]['acronym']}: {ana_dict[idx]['name']}")
            print('\n')
                
            '''
            sns.heatmap(df_nz, norm=LogNorm(), cmap='coolwarm')
            
            '''
            cm = plt.get_cmap('coolwarm')
            ncolors_col = len(np.unique(bregs3))
            lut_col = dict(zip(np.unique(bregs3), cm(np.arange(ncolors_col)/ncolors_col)))
            col_colors = [lut_col[bi] for bi in bregs3]
            print(bregs3)

            rcolors_ref = 'rbgymck'
            ncolors_row = len(np.unique(modalities))
            lut_row = dict(zip(np.unique(modalities), rcolors_ref[:ncolors_row]))
            row_colors = [lut_row[bi] for bi in modalities]
            
            
            clust_map = sns.clustermap(df_nz_copy, norm=LogNorm(), cmap='coolwarm', 
                                #col_colors=col_colors,
                                row_colors=row_colors,
                                cbar_kws={'label': '# of labeled voxels'},
                                xticklabels=False, yticklabels=False)

            # legend for row/col_colors
            handles_row = [Patch(facecolor=lut_row[name]) for name in lut_row]
            plt.legend(handles_row, lut_row, title='modality-source', bbox_to_anchor=(0,0), bbox_transform=plt.gcf().transFigure, loc='lower left')
            #handles_col = [Patch(facecolor=lut_col[name]) for name in lut_col]
            #plt.legend(handles_col, lut_col, title='Regions (level 3)', bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure, loc='upper right')

            plt.savefig('region_distr_heatmap.png', dpi=300)
            plt.close()


        if self.plot and False:
            # clustering
            nz_t = df_nz.copy()
            # standardization
            nz_t = nz_t.div(nz_t.sum(axis=1)+1e-6, axis=0)
            nz_t = nz_t.transpose()
            
            # do umap projection
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(nz_t)

            labels = hdbscan.HDBSCAN().fit_predict(embedding)
            print(len(np.unique(labels)))
            palette = sns.color_palette('deep', np.unique(labels).max() + 1)
            colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
            plt.scatter(embedding.T[0], embedding.T[1], c=colors)
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.savefig('cluster.png', dpi=300)
            plt.close()
            
            


        plot_corr = True
        if self.plot and plot_corr:
            # remove empty regions
            nvox_thr = 0
            darr_copy = darr.copy()
            darr_copy[darr_copy < nvox_thr] = 0

            darr_sum0_copy = darr_copy.sum(axis=0)
            nz_sum_copy = np.nonzero(darr_sum0_copy)[0]

            darr_nz_copy = darr_copy[:,nz_sum_copy]
            ids_sum_nz_copy = np.array(ids)[nz_sum_copy]
            ids_sum_nz_copy = [ana_dict[idx]['acronym'] for idx in ids_sum_nz_copy]

            df_corr = pd.DataFrame(darr_nz_copy, columns=ids_sum_nz_copy)
            print(df_corr.shape)
            
            df_corr.replace(0, np.nan, inplace=True)
            corr = df_corr.corr(min_periods=10)
            corr[corr < 0] = 0

            if False:
                # we should zero-masking the neighboring regions
                with open(neighbor_file, 'rb') as fp:
                    nb_dict = pickle.load(fp)

                print(corr.mean().mean(), corr.max().min(), corr.min().min())
                ids_set = set(ids.tolist())
                for key, values in nb_dict.items():
                    if key not in ids_set:
                        continue
                    vms = []
                    for v in values:
                        if v in ids_set:
                            vms.append(v)

                    if len(vms) == 0:
                        continue
                    else:
                        corr[key][vms] = 0
            
            print(corr.mean().mean(), corr.max().min(), corr.min().min())
            sns.heatmap(corr, cmap='coolwarm')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Brain region ID')
            plt.ylabel('Brain region ID')
            plt.title("Correlation coefficients between labeled regions")
            plt.savefig('region_corr.png', dpi=300)
            plt.close()
        

        
        

if __name__ == '__main__':
    import pickle

    distr_dir = './statis_out_adaThr'
    dim_file = './ccf_info/TeraDownsampleSize.csv'
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
        bssa.calc_left_right_corr(distr_dir)

    if 1:
        bssa.labeling_corr(distr_dir)
    
        


