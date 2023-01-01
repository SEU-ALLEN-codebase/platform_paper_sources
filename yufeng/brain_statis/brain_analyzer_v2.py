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
import torch

import hdbscan
import umap

from file_io import load_image, save_image
from image_utils import get_mip_image
from anatomy.anatomy_config import REGION671, MASK_CCF25_FILE
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

def format_precomputed_df(df, last_column='modality'):
    rnames = [rname for rname in df.columns if rname != last_column]
    cname_mapper = dict(zip(df, map(int, rnames)))
    df.rename(columns=cname_mapper, inplace=True)
    df.rename(index=str, inplace=True)
    return df


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


    def parse_distrs(self, distr_dir, ignore_lr=True, precomputed_file=None):
        if precomputed_file:
            df = pd.read_csv(precomputed_file, index_col=0)
            return df

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
        print(f'Non-empty df shape: {df.shape}')

        save_result = False
        if save_result:
            df.to_csv("precomputed_distrs.csv", sep=',')
        
        return df

    def load_somata(self, somata_dir='/media/lyf/Carry/paper/fig1bstype/marker_regi_25', precomputed_file=None):
        if precomputed_file:
            df = pd.read_csv(precomputed_file, index_col=0)
            return df

        sfiles = sorted(glob.glob(os.path.join(somata_dir, '*/*.swc')))
        brains = [os.path.split(os.path.split(sfile)[0])[-1] for sfile in sfiles]
        ids = REGION671[:]
        ids_set = set(ids)
        # initialize dataframe
        df = pd.DataFrame(np.zeros((len(sfiles), len(ids_set)+1)), columns=ids+['modality'], index=brains)

        print('Loading the CCFv3-25um mask file for region assignment')
        mask = load_image(MASK_CCF25_FILE)
        zm, ym, xm = mask.shape # z,y,x order

        print('Loading somata data')
        for sfile in sfiles:
            brain = os.path.split(os.path.split(sfile)[0])[-1]
            print(f'--> Processing for {brain}')
            df.loc[brain, 'modality'] = 'fMOST-Zeng'

            coords = np.genfromtxt(sfile)
            if coords.size == 0:
                print(f'<-- No somata is found!')
                continue
            elif coords.ndim == 1:
                print(f'Warning: only 1 somata is found!')
                coords = coords.reshape(1,-1)
            
            coords = np.floor(coords[:,2:5]).astype(np.int32)  
            # discard possible out-of-box coordinates, in case of error
            inrange_mask = (coords >= 0) & (coords < np.array([xm,ym,zm]))
            inrange_mask = np.sum(inrange_mask, axis=1) == 3
            # get the region according to the coordinates
            coords = coords[inrange_mask]
            # accessing the region index
            regions = mask[coords[:,2], coords[:,1], coords[:,0]]
            regions = regions[regions > 0]
            rs, cs = np.unique(regions, return_counts=True)
            for r,c in zip(rs, cs):
                df.at[brain, r] = c

        df.to_csv('precomputed_somata.csv', sep=',')
            
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

    def plot_region_distrs_labeling2(self, precomputed_file, out_img_file='temp.png', height=10, aspect=1.0, sizes=(1,50)):
        if type(precomputed_file) is str:
            df = pd.read_csv(precomputed_file, index_col=0)
        else:
            df = precomputed_file
        
        sx = df.shape[0]
        sy = df.shape[1] - 1
        ndf = pd.DataFrame(np.zeros((sx*sy, 4)), columns=('region', 'brain', 'scale', 'label'))
        rnames = [self.ana_dict[idx]['acronym'] for idx in df.columns.drop('label')]
        ndf['region'] = np.array(rnames).repeat(sx).reshape(sy, sx).transpose().reshape(-1)
        ndf['brain'] = df.index.repeat(sy)
        ndf['scale'] = df.drop('label', axis=1).to_numpy().reshape(-1)
        ndf['label'] = df['label'].to_numpy().repeat(sy)

        cs = 'brgcmyk'
        nlabel = len(np.unique(df.label))
        k = int(np.ceil(nlabel / len(cs)))
        palette = [c for c in (cs * k)[:nlabel]]
        g = sns.relplot(
            data=ndf,
            x='region',
            y='brain',
            size='scale',
            hue='label',
            height=height,
            aspect=aspect,
            sizes=sizes,
            size_norm=(0,1),
            palette=palette
        )
        plt.xticks(rnames, fontsize=13)
        plt.yticks(labels=None, fontsize=0)
        plt.xlabel('')
        plt.ylabel('Brain', fontsize=20)
        plt.grid(alpha=0.5)
        plt.savefig(out_img_file, dpi=300)
        plt.close()
   
    def plot_region_distrs_labeling2_comp(self, precomputed_somata, precomputed_signal, region_level=1):
        """
            Plot the region-vs-brain relplot distribution of somata and signal at the same time, so
        that it is convenient to use the same set of high-density regions for comparison
        """

        def preprocess(df, region_level):
            format_precomputed_df(df, last_column='modality')
            df = self.map_to_coarse_regions(df, level=region_level, last_column='modality')
            df = self.convert_modality_to_label(df, normalize=True)
            df = df[df.label != '']
            df.rename_axis('brain').sort_values(by=['label', 'brain'], ascending=[True, True], inplace=True)
            return df


        df_somata = pd.read_csv(precomputed_somata, index_col=0)
        df_signal = pd.read_csv(precomputed_signal, index_col=0)

        # find the common abundent regions between somata and signal
        dfn1 = preprocess(df_somata, region_level=region_level)
        dfn2 = preprocess(df_signal, region_level=region_level)
        
        somata_d = dfn1.drop('label', axis=1).sum()
        signal_d = dfn2.drop('label', axis=1).sum()
        mthresh = 4 * somata_d.sum() / somata_d.shape[0]
        idx1 = somata_d[somata_d > mthresh].index
        idx2 = signal_d[signal_d > mthresh].index
        idxs = (idx1 | idx2).astype(np.int32)
        # select subset of data according to idxs: high-set for main penal of fig3, low-set for suppl figure
        hsom = dfn1[idxs]
        hsom = hsom.assign(label=dfn1['label'])
        lsom = dfn1.drop(idxs, axis=1);
        hsig = dfn2[idxs]
        hsig = hsig.assign(label=dfn2['label'])
        lsig = dfn2.drop(idxs, axis=1);

        # select by brain
        bthresh = 0.8
        hi1 = hsom.sum(axis=1) > bthresh
        hi2 = hsig.sum(axis=1) > bthresh
        hi = hi1 | hi2
        hsom = hsom[hi]
        hsig = hsig[hi]
        print('hsom: ', hsom)
        print('hsig: ', hsig)
        
        
        # plot the the figures using function `plot_region_distrs_labeling2`
        # 4 plots: high-set of somata, high-set of signal, low-set of somata, low-set of signal
        self.plot_region_distrs_labeling2(hsom, out_img_file='region_distr_hsom.png', height=5, aspect=1., sizes=(0, 500))
        #self.plot_region_distrs_labeling2(dfn1, out_img_file='region_distr_som.png')
        self.plot_region_distrs_labeling2(hsig, out_img_file='region_distr_hsig.png', height=5, aspect=1., sizes=(0, 500))
        #self.plot_region_distrs_labeling2(dfn2, out_img_file='region_distr_sig.png')

    def plot_distribution_all(self, precomputed_somata, precomputed_signal, region_level=1):
        def preprocess(df, region_level):
            format_precomputed_df(df, last_column='modality')
            df = self.map_to_coarse_regions(df, level=region_level, last_column='modality')
            df = self.convert_modality_to_label(df, normalize=False)
            #df = df[df.label != '']
            df.rename_axis('brain').sort_values(by=['label', 'brain'], ascending=[True, True], inplace=True)
            return df


        df_somata = pd.read_csv(precomputed_somata, index_col=0)
        df_signal = pd.read_csv(precomputed_signal, index_col=0)

        # find the common abundent regions between somata and signal
        dfn1 = preprocess(df_somata, region_level=region_level)
        dfn2 = preprocess(df_signal, region_level=region_level)

        som_reg = dfn1.sum().drop('label')
        #som_reg /= som_reg.sum()
        som_bra = dfn1.drop('label', axis=1).sum(axis=1)
        nsom = som_bra.shape[0]
        #som_bra /= (som_bra.sum() * sig_bra.shape[0] / nsom)

        sig_reg = dfn2.sum().drop('label')
        sig_reg /= (sig_reg.mean() / som_reg.mean())
        sig_bra = dfn2.drop('label', axis=1).sum(axis=1)
        #sig_bra /= sig_bra.sum()
        sig_bra /= (sig_bra.mean() / som_bra.mean())

        # fill the missing values
        for idx in sig_bra.index:
            if idx not in som_bra.index:
                som_bra.loc[idx] = np.NaN


        
        # merge the data
        rdistr = np.vstack((som_reg.to_numpy(), sig_reg.to_numpy()))
        ind1 = rdistr[1].argsort()
        rdistr = rdistr[:,ind1].reshape(-1)
        reg = pd.DataFrame(
            {
                'regional distr': rdistr, 
                'region_id':np.hstack((som_reg.index[ind1], sig_reg.index[ind1])), 
                'region':np.hstack((range(som_reg.shape[0]), range(sig_reg.shape[0]))),
                'type': ['somata' for i in range(som_reg.shape[0])] + ['signal' for i in range(sig_reg.shape[0])]
            })

        bdistr = pd.DataFrame({
                'som': np.zeros(sig_bra.shape[0]),
                'sig': sig_bra.to_numpy()}, index=sig_bra.index)
        for idx in bdistr.index:
            bdistr.loc[idx, 'som'] = som_bra.loc[idx]
        bdistr = bdistr.to_numpy().transpose()

        ind2 = bdistr[1].argsort()
        bdistr = bdistr[:,ind2].reshape(-1)
        bra = pd.DataFrame(
            {
                'brain-wide distr': bdistr, 
                'brain_id':np.hstack((sig_bra.index[ind2], sig_bra.index[ind2])), 
                'brain':np.hstack((range(som_bra.shape[0]), range(sig_bra.shape[0]))),
                'type': ['somata' for i in range(som_bra.shape[0])] + ['signal' for i in range(sig_bra.shape[0])]
            })

        # plotting
        sns.set_style("darkgrid")
        sns.scatterplot(
            data=reg, x="region", y="regional distr", hue="type"
        )
        plt.yscale('log')
        plt.xlabel('Brain region', fontsize=16)
        plt.ylabel('#Signal', fontsize=16)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('regional_distr.png', dpi=300)
        plt.close('all')

        sns.scatterplot(
            data=bra, x="brain", y="brain-wide distr", hue="type"
        )
        plt.yscale('log')
        xlim = plt.xlim()
        plt.axvspan(nsom, xlim[1], color='#388E3C', alpha=0.1)
        plt.xlim(xlim)
        plt.xlabel('Brain', fontsize=16)
        plt.ylabel('#Signal', fontsize=16)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('brainwide_distr.png', dpi=300)
        plt.close('all')

    def plot_sparsity_versus_labeling(self, precomputed_somata, precomputed_signal, region_level=1):
        def preprocess(df, region_level):
            format_precomputed_df(df, last_column='modality')
            df = self.map_to_coarse_regions(df, level=region_level, last_column='modality')
            df = self.convert_modality_to_label(df, normalize=False)
            df = df[df.label != '']
            df = df[df.label != 'Ai139']
            df.sort_values(by=['label'], ascending=[True], inplace=True)
            return df

        df_somata = pd.read_csv(precomputed_somata, index_col=0)
        df_signal = pd.read_csv(precomputed_signal, index_col=0)
        dfn1 = preprocess(df_somata, region_level=region_level)
        dfn2 = preprocess(df_signal, region_level=region_level)

        # somata vs labeling
        som_bra = dfn2.sum(axis=1).rename('signal').to_frame()
        som_bra['label'] = dfn2['label']
        plt.figure(figsize=(8,8))
        sns.set_style("darkgrid")
        sns.scatterplot(data=som_bra, x='signal', y='label')
        plt.xscale('log')
        plt.xlabel('#Signal', fontsize=20)
        plt.ylabel('Labeling', fontsize=20)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('labeling_sparsity.png', dpi=300)
        plt.close('all')

    def regional_level_plexing(self, precomputed_signal, brain, pos_thresh=0.001):
        df_signal = pd.read_csv(precomputed_signal, index_col=0)
        format_precomputed_df(df_signal, last_column='modality')
        
        distr = df_signal.loc[brain].drop(['modality'])

        mask = load_image(MASK_CCF25_FILE)
        vpos = distr.sum() * pos_thresh
        pos_regions = distr[distr > vpos].index
        # the processing of large number of regions masking is expensive, use gpu
        mask_gpu = torch.from_numpy(mask.astype(np.int64)).cuda()
        m = torch.zeros(mask_gpu.shape, dtype=torch.bool, device=mask_gpu.device)
        m.fill_(0)
        for i, r in enumerate(pos_regions):
            print(i, r)
            m = m | (mask_gpu == r)
        print(m.sum().item(), m.size())
        mc = m.cpu().numpy()
        del mask_gpu, m

        mask_out = np.zeros(mask.shape, dtype=np.uint8)
        mask_out[mask > 0] = 1
        mask_out[mc] = 2

        # coloring
        mip1 = get_mip_image(mask_out, 0)
        cmip1 = np.zeros((*mip1.shape, 3), dtype=np.uint8)
        
        color = (0,255,255)
        bg = (123,123,123)
        cmip1[mip1==2] = color
        cmip1[mip1==1] = bg
        save_image('mip1.png', cmip1)


    def convert_modality_to_label(self, df, label_file='./fMOST-Zeng_labels_edited.csv', normalize=True):
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
            if irow in df.index:
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
            cnames = corr[corr.sum() == 0].index.to_numpy().tolist()
            corr = corr.drop(cnames).drop(cnames, axis=1)

            print(corr.mean().mean(), corr.max().min(), corr.min().min())
            corr = corr.fillna(0)
            clust_map = sns.clustermap(corr, cmap='coolwarm')
            clust_map.cax.set_visible(False)

            names = df_corr.columns
            rids = df.columns[:-1]
            for i in range(corr.shape[0]):
                ind = clust_map.dendrogram_col.reordered_ind[i]
                region_name = names[ind]
                region_id = rids[ind]
                #print(f'[{i}]{region_name}', end=': ')
                #for r in self.ana_dict[region_id]['structure_id_path']:
                #    print(self.ana_dict[r]['acronym'], end=', ')
                #print('')
                print(region_name, end=', ')
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

    if 1:
        precomputed_signal = 'precomputed_distrs.csv'
        precomputed_somata = 'precomputed_somata.csv'
        #bssa.plot_region_distrs_modalities(distr_dir)
        #bssa.plot_region_distrs_labeling2_comp(precomputed_somata, precomputed_signal, region_level=1)
        bssa.plot_distribution_all(precomputed_somata, precomputed_signal, region_level=1)
        #bssa.plot_sparsity_versus_labeling(precomputed_somata, precomputed_signal, region_level=1)
        
    if 0:
        bssa.calc_left_right_corr(distr_dir)

    if 0:
        bssa.corr_clustermap(distr_dir)
        #bssa.load_somata()
    
        


