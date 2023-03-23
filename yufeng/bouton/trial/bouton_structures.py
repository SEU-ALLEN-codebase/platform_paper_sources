#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : bouton_structures.py
#   Author       : Yufeng Liu
#   Date         : 2023-03-22
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd
import pickle
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from file_io import load_image
from swc_handler import parse_swc, scale_swc, get_specific_neurite, NEURITE_TYPES, find_soma_node
from morph_topo.morphology import Morphology, Topology
from anatomy.anatomy_config import MASK_CCF25_FILE

class BoutonStructure:
    def __init__(self, swcfile):
        tree = self.load_swc(swcfile, scale=25.)
        self.morph = Morphology(tree)
        
    @staticmethod
    def load_swc(swcfile, scale=25.):
        with open(swcfile, 'rb') as fp:
            tree = pickle.load(fp)
        tree = scale_swc(tree, scale)
        return tree

    def extract_segments_with_bouton(self):
        topo_tree, seg_dict = self.morph.convert_to_topology_tree()
        frag_lengths, frag_lengths_dict = self.morph.calc_frag_lengths()
        topo = Topology(topo_tree)
        soma_node = find_soma_node(self.morph.tree)
        soma_pos = np.array(self.morph.pos_dict[soma_node][2:5])
        # iterate all fragments
        segs = {}
        for seg_id, seg_nodes in seg_dict.items():
            if self.morph.pos_dict[seg_id][1] not in [2,5]:
                continue
            pre_seg_id = topo.pos_dict[seg_id][-1]
            cur_nodes = [pre_seg_id] + seg_nodes[::-1] + [seg_id]
            # check the number of boutons
            bs = []
            fl = 0
            for pid, cid in zip(cur_nodes[:-1], cur_nodes[1:]):
                fl += frag_lengths_dict[cid]
                if self.morph.pos_dict[cid][1] == 5:
                    bs.append(fl)
            pcoord = np.array(self.morph.pos_dict[pre_seg_id][2:5])
            edist2soma = np.linalg.norm(soma_pos - pcoord)
            if seg_id in self.morph.child_dict:
                childs = self.morph.child_dict[seg_id]
            else:
                childs = []
            segs[seg_id] = (np.array(bs), fl, edist2soma, pre_seg_id, childs, topo.order_dict[seg_id])
        return segs

class BoutonMotif:
    def __init__(self, segs_file, discard_bif=True):
        with open(segs_file, 'rb') as fp:
            self.segs_all = pickle.load(fp)
        self.discard_bif = discard_bif

    def seg_motif(self):
        i = 0
        feats = []
        for prefix, segs in self.segs_all.items():
            if i % 50 == 0:
                print(i)
            for seg_id, seg in segs.items():
                bs, fl = seg[:2]
                if len(bs) > 0 and bs[-1] == fl and self.discard_bif:
                    bs = bs[:-1]
                nbs = len(bs)
                if nbs == 0: continue
                feat = [fl, nbs, seg[2], len(seg[4]), seg[5]]
                feats.append(feat)

            i += 1
        # do clustering
        feats = pd.DataFrame(feats, columns=['segLength', 'numBouton', 'edist2soma', 'nchildren', 'level'])
        sns.scatterplot(feats, x='segLength', y='numBouton', s=5, alpha=0.5)
        #plt.xlim(0, 1000)
        #plt.ylim(0, 26)
        plt.savefig('seg_feat_distr.png', dpi=300)
        plt.close('all')

    def interbouton_distribution(self):
        i = 0
        ibdists = []
        for prefix, segs in self.segs_all.items():
            if i % 50 == 0:
                print(i)
            for seg_id, seg in segs.items():
                bs = seg[0]
                if len(bs) > 0 and bs[-1] == seg[1] and self.discard_bif:
                    bs = bs[:-1]
                ibdists.extend(bs)
            i += 1

        ibdists = np.array(ibdists)
        ibdists = ibdists[ibdists > 5]
        df = pd.DataFrame(ibdists, columns=['interbouton'])

        fig, axes = plt.subplots(1,1,figsize=(4,4))
        hp = sns.histplot(df, x='interbouton')
        #x = np.array([v.get_x() fo v in hp.axes.containers[1]])
        #y = np.array([v.get_height() for v in hp.axes.containers[1]])
        print('fitting...')
        samp = df.sort_values(['interbouton']).to_numpy()
        #param=scipy.stats.lognorm.fit(samp) # fit the sample data
        param = scipy.stats.exponweib.fit(samp, loc=0.02, scale=80)
        print(param)
        print(f'Finished fitting')
        x = np.linspace(0, 600, 300)
        #pdf_fitted = scipy.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2]) # fitted distribution
        pdf_fitted = scipy.stats.exponweib.pdf(x, *param)
        pdf_fitted *= (8300 / pdf_fitted.max()) # the number 8300 should be calculated!!!
        plt.plot(x,pdf_fitted,'r-', label=f'Exponential Weibull with\nexp={param[0]:.2f}, shape={param[1]:.2f}')
        plt.subplots_adjust(left=0.16)

        plt.xlim(0, 600)
        plt.xlabel(r'Inter-bouton distance ($\mu$m)')
        plt.legend()
        plt.savefig('interbouton_distr.png', dpi=300)
        plt.close('all')

    def interbouton_dependence(self):
        i = 0
        ibpairs = []
        for prefix, segs, in self.segs_all.items():
            if i % 50 == 0:
                print(i)
            for seg_id, seg in segs.items():
                bs = seg[0]
                if len(bs) > 0 and bs[-1] == seg[1] and self.discard_bif:
                    bs = bs[:-1]
                if len(bs) > 2:
                    diff = bs[1:] - bs[:-1]
                    for j in range(len(diff) - 1):
                        l1, l2 = diff[j:j+2]
                        ibpairs.append((l1, l2))
            i += 1

        ibpairs = np.array(ibpairs)
        pl = 'Inter-bouton distance1 ($\mu$m)'
        al = 'Inter-bouton distance2 ($\mu$m)'
        df = pd.DataFrame(ibpairs, columns=[pl, al])
        # binning
        rngs = [0, 8, 16, 32, 64, 10000]
        palette = {}
        colors = ('r', 'g', 'b', 'm', 'y', 'c', 'k')
        ldict = {}
        for j in range(len(rngs)-1):
            rng = [rngs[j], rngs[j+1]]
            vrng = (rng[0] + rng[1]) / 2.
            vstr = f'[{rng[0]}, {rng[1]})'
            ldict[str(vrng)] = vstr

            mask = (df[pl] >= rng[0]) & (df[pl] < rng[1])
            rmean = df[al][mask].mean()
            print(f'{vstr}: {mask.sum()}, {rmean}')
            df.loc[:, pl][mask] = vrng
            palette[vrng] = colors[j]
       
        fig, axes = plt.subplots(1,1,figsize=(4,4))
        xmax = 60
        g = sns.kdeplot(df, x=al, hue=pl, bw_adjust=1, fill=False, alpha=0.7, 
                    linewidth=1.5, clip=(0,xmax), palette=palette, common_norm=False)
        g.legend_.set_title(pl)
        g.legend_.set_frame_on(False)

        for t in g.legend_.texts:
            t.set_text(ldict[t.get_text()])
        plt.yticks([])
        plt.xlim(0, xmax)
        plt.savefig('interbouton_dependence.png', dpi=300)
        plt.close('all')
                        
        
    def branch_dependence(self):
        i = 0
        bpairs = []
        for prefix, segs, in self.segs_all.items():
            if i % 50 == 0:
                print(i)
            for seg_id, seg in segs.items():
                bs = seg[0]
                if len(bs) > 0 and bs[-1] == seg[1] and self.discard_bif:
                    bs = bs[:-1]
                nbs = len(bs)
                pre_seg_id = seg[3]
                if pre_seg_id in segs:
                    pre_seg = segs[pre_seg_id]
                    pbs = pre_seg[0]
                    if len(pbs) > 0 and pbs[-1] == pre_seg[1] and self.discard_bif:
                        pbs = pbs[:-1]
                    bpairs.append((len(pbs), nbs))

            i += 1

        bpairs = np.array(bpairs)
        pl = '#boutons of parent branch'
        al = '#boutons of current branch'
        df = pd.DataFrame(bpairs, columns=[pl, al])
        # binning
        rngs = [0, 1, 5, 10, 20, 100]
        palette = {}
        colors = ('r', 'g', 'b', 'm', 'y', 'c', 'k')
        ldict = {}
        for j in range(len(rngs)-1):
            rng = [rngs[j], rngs[j+1]]
            vrng = (rng[0] + rng[1]) / 2.
            vstr = f'[{rng[0]}, {rng[1]})'
            ldict[str(vrng)] = vstr

            mask = (df[pl] >= rng[0]) & (df[pl] < rng[1])
            rmean = df[al][mask].mean()
            print(f'{vstr}: {mask.sum()}, {rmean}')
            df.loc[:, pl][mask] = vrng
            palette[vrng] = colors[j]
       
        fig, axes = plt.subplots(1,1,figsize=(4,4))
        xmax = 20
        g = sns.kdeplot(df, x=al, hue=pl, bw_adjust=1, fill=False, alpha=0.7, 
                    linewidth=1.5, clip=(0,xmax), palette=palette, common_norm=False)
        g.legend_.set_title(pl)
        g.legend_.set_frame_on(False)

        for t in g.legend_.texts:
            t.set_text(ldict[t.get_text()])
        plt.yticks([])
        plt.xlim(0, xmax)
        plt.savefig('branch_dependence.png', dpi=300)
        plt.close('all')
        
    
    def segment_distr(self):
        bbs1 = []
        bbs4 = []
        bbs8 = []
        bbs = []
        i = 0
        for prefix, segs in self.segs_all.items():
            if i % 50 == 0:
                print(i)
            for seg in segs.values():
                bs, fl = seg[:2]
                nbs = len(bs)
                if nbs > 0 and self.discard_bif and bs[-1] == fl:
                    bs = bs[:-1]
                    nbs = len(bs)
                if nbs > 0:
                    bs = np.array(bs)
                    bs /= fl
                    if nbs == 1:
                        bbs1.extend(bs)
                    elif nbs <= 4:
                        bbs4.extend(bs)
                    elif nbs <= 8:
                        bbs8.extend(bs)
                    else:
                        bbs.extend(bs)
            i += 1

        df_bbs = pd.DataFrame(bbs, columns=['bouton'])
        df_bbs1 = pd.DataFrame(bbs1, columns=['bouton'])
        df_bbs4 = pd.DataFrame(bbs4, columns=['bouton'])
        df_bbs8 = pd.DataFrame(bbs8, columns=['bouton'])

        sns.displot(df_bbs, x='bouton')
        plt.savefig('bbs8+.png', dpi=300)
        plt.close('all')
        sns.displot(df_bbs1, x='bouton')
        plt.savefig('bbs1.png', dpi=300)
        plt.close('all')
        sns.displot(df_bbs4, x='bouton')
        plt.savefig('bbs4.png', dpi=300)
        plt.close('all')
        sns.displot(df_bbs8, x='bouton')
        plt.savefig('bbs8.png', dpi=300)
        plt.close('all')
        
    
    
    

if __name__ == '__main__':
    if 0:
        swc_dir = '../bouton_v20230110_swc_pickle'
        segs_dict = {}
        nprocessed = 0
        for swcfile in glob.glob(os.path.join(swc_dir, '*pkl')):
            prefix = os.path.split(swcfile)[-1].split('.swc')[0]
            print(f'[{nprocessed}/1891]: {prefix}')
            bs = BoutonStructure(swcfile)
            segs = bs.extract_segments_with_bouton()
            segs_dict[prefix] = segs
            
            nprocessed += 1
            if nprocessed  == 150:
                break
            
        with open('segs_boutons.pkl', 'wb') as fp:
            pickle.dump(segs_dict, fp)

    if 1:
        segs_file = 'segs_boutons150.pkl'
        bm = BoutonMotif(segs_file)
        #bm.segment_distr()
        #bm.seg_motif()
        #bm.interbouton_distribution()
        #bm.interbouton_dependence()
        bm.branch_dependence()

