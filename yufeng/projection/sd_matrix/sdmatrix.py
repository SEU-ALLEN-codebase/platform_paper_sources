#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : sdmatrix.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-30
#   Description  : 
#
#================================================================

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
sys.path.append('../../common_lib')
from common_utils import plot_sd_matrix, stype2struct, struct_dict, CorticalLayers, PstypesToShow

def get_coords(main_tract_dir, outfile):
    cdict = {}
    ic = 0
    for swcfile in glob.glob(os.path.join(main_tract_dir, '*_tract.swc')):
        sname = os.path.split(swcfile)[-1][:-17]
        stype = sname.split('_')[0]
        prefix = '_'.join(sname.split('_')[1:])
        coords = np.genfromtxt(swcfile, delimiter=' ', usecols=(2,3,4))

        if stype not in cdict:
            cdict[stype] = [(prefix, coords)]
        else:
            cdict[stype].append((prefix, coords))

        ic += 1
        if ic % 10 == 0:
            print(ic)

    with open(outfile, 'wb') as fp:
        pickle.dump(cdict, fp)
    return cdict

class MainTractFeatures(object):
    def __init__(self, path_coords):
        self.coords = path_coords
    
    def plength(self):
        pts1 = self.coords[:-1]
        pts2 = self.coords[1:]
        length = np.linalg.norm(pts2 - pts1, axis=1).sum()
        return length

    def elength(self):
        pt1 = self.coords[0]
        pt2 = self.coords[-1]
        length = np.linalg.norm(pt2 - pt1)
        return length

    def orientation(self):
        pt1 = self.coords[0] # termini
        pt2 = self.coords[-1]
        v = pt1 - pt2
        v = v / np.linalg.norm(v)
        return v

    def volume(self):
        pca = PCA(3)
        new = pca.fit_transform(self.coords)
        diff = new.max(axis=0) - new.min(axis=0)
        return diff.prod()

    def run(self, include_coord=False):
        pl = self.plength()
        el = self.elength()
        orient = self.orientation()
        vo = self.volume()
        if include_coord:
            return pl, el, *orient, vo, *self.coords[-1]
        else:
            return pl, el, *orient, vo

def plot_all(main_tract_file, figname, include_coord=False, normalize=True):
    with open(main_tract_file, 'rb') as fp:
        cdict = pickle.load(fp)

    structs = []
    features = []
    for region, clist in cdict.items():
        structs.append(stype2struct[region])
        fs_list = []
        for neuron in clist:
            coords = neuron[1]
            mtf = MainTractFeatures(coords)
            fs = mtf.run(include_coord=include_coord)
            fs_list.append(fs)
        fs_list = np.array(fs_list)
        fv = np.hstack((fs_list.mean(axis=0)))
        features.append(fv)
        
    df = pd.DataFrame(features)
    # normalize
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, corr, figname, '')

def plot_struct(main_tract_file, figname, regions, include_coord=False, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    with open(main_tract_file, 'rb') as fp:
        cdict = pickle.load(fp)

    structs = []
    features = []
    for region, clist in cdict.items():
        #print(region)
        if region not in regions:
            continue
        for neuron in clist:
            coords = neuron[1]
            mtf = MainTractFeatures(coords)
            fs = mtf.run(include_coord=include_coord)
            features.append(fs)
            structs.append(region)
        
    df = pd.DataFrame(features)
    # normalize
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, regions, corr, figname, '', vmin=vmin, vmax=vmax, annot=annot)

def plot_struct_with_cortical_layer(main_tract_file, celltype_file, figname, regions, include_coord=False, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    with open(main_tract_file, 'rb') as fp:
        cdict = pickle.load(fp)
    df_ct = pd.read_csv(celltype_file, index_col='Cell name')

    structs = []
    features = []
    for region, clist in cdict.items():
        #print(region)
        if region not in regions:
            continue
        for neuron in clist:
            coords = neuron[1]
            mtf = MainTractFeatures(coords)
            fs = mtf.run(include_coord=include_coord)
            features.append(fs)

            prefix = neuron[0]
            info = df_ct.loc[prefix]
            stype, cl = info.Manually_corrected_soma_region, info.Cortical_layer
            if cl is np.NaN:
                cl = ''
            else:
                cl = f'-{cl}'
            region_l = f'{stype}{cl}'
            structs.append(region_l)
    
    # remove layers with few numbers
    #rs, counts = np.unique(structs, return_counts=True)
    #rs = rs[counts >= 5]
    rs = CorticalLayers
    structs_sel = []
    features_sel = []
    for s, f in zip(structs, features):
        if s in rs:
            structs_sel.append(s)
            features_sel.append(f)
    rs = sorted(rs, key=lambda x: x.split('-')[-1])

    df = pd.DataFrame(features_sel)
    
    # normalize
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    print(rs)
    plot_sd_matrix(structs_sel, rs, corr, figname, '', vmin=vmin, vmax=vmax, annot=annot)

def plot_struct_with_ptype(main_tract_file, celltype_file, figname, regions, ptypes, include_coord=False, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    def sort_lambda(x):
        sp = x.split('-')
        if len(sp) == 1:
            return f'1_{x}'
        else:
            return f'0_{sp[-1]}_{x}'

    with open(main_tract_file, 'rb') as fp:
        cdict = pickle.load(fp)
    df_ct = pd.read_csv(celltype_file, index_col='Cell name')

    structs = []
    features = []
    for region, clist in cdict.items():
        #print(region)
        if region not in regions:
            continue
        for neuron in clist:
            coords = neuron[1]
            mtf = MainTractFeatures(coords)
            fs = mtf.run(include_coord=include_coord)
            features.append(fs)

            prefix = neuron[0]
            info = df_ct.loc[prefix]
            stype, pt = info.Manually_corrected_soma_region, info.Subclass_or_type
            if pt is np.NaN:
                pt = ''
            else:
                pt = pt.split('_')[-1]
                pt = f'-{pt}'
            region_l = f'{stype}{pt}'
            structs.append(region_l)
    
    # remove layers with few numbers
    #structs = [struct for struct in structs if struct != 'CP-others']
    #rs, counts = np.unique(structs, return_counts=True)
    #rs = rs[counts >= 5]
    #rs = sorted(rs, key=sort_lambda)

    structs_sel = []
    features_sel = []
    for s, f in zip(structs, features):
        if s in ptypes:
            structs_sel.append(s)
            features_sel.append(f)

    df = pd.DataFrame(features_sel)
    
    # normalize
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs_sel, ptypes, corr, figname, '', vmin=vmin, vmax=vmax, annot=annot)

if __name__ == '__main__':
    main_tract_file = 'main_tract.pkl'
    celltype_file = '../../common_lib/41586_2021_3941_MOESM4_ESM.csv'
    include_coord = False

    for structure, regions in struct_dict.items():
        #figname = f'sdmatrix_motif_{structure.lower()}'
        #plot_struct(main_tract_file, figname, regions, include_coord=include_coord, vmin=-0.4, vmax=0.8, annot=False)
        
        #if structure == 'CTX':
        #    figname = f'sdmatrix_motif_{structure.lower()}_withLayer'
        #    plot_struct_with_cortical_layer(main_tract_file, celltype_file, figname, regions, include_coord=include_coord, vmin=-0.4, vmax=0.8, annot=False)
    
        figname = f'sdmatrix_motif_{structure.lower()}_withPtype'
        plot_struct_with_ptype(main_tract_file, celltype_file, figname, regions, PstypesToShow[structure], include_coord=include_coord, vmin=-0.4, vmax=0.8, annot=False)
        
