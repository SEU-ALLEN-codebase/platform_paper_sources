#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : valid.py
#   Author       : Yufeng Liu
#   Date         : 2022-08-06
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import random
from scipy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from swc_handler import parse_swc


def calc_best_viewpoint(pts):
    pca = PCA()
    pca.fit(pts)
    x,y,z = pca.components_[2]
    elev = np.rad2deg(np.arcsin(z))
    azim = np.rad2deg(np.arctan(x/y))
    return elev, azim

def plot_main_tracts(class_name, key='tract.swc'):
    # visualize and check the path
    show_pts = 200
    figname = f'{class_name}_main_tract_vis.png'
    scale = 1000

    # intitalize the fig
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # load all the main paths
    pts = []
    termini = []
    somata = []
    files = ['../main_tracts_types/CP_GPe-CP_18455_00012_axonal_tract.swc']
    for pathfile in files:
        tree = parse_swc(pathfile)
        coords = np.array([node[2:5] for node in tree][::-1]) # soma to terminal
        coords /= scale

        nnodes = len(coords)
        step = max(nnodes // show_pts, 1)
        coords_sub = coords[::step]
        pts.extend(coords)
        termini.append(coords[-1])
        somata.append(coords[0])

        # plot
        ax.scatter(coords_sub[0,0], coords_sub[0,1], coords_sub[0,2], marker='o', color='b', s=200)
        #ax.scatter(coords_sub[-1,0], coords_sub[-1,1], coords_sub[-1,2], marker='^', color='k', s=150)
        ax.plot(coords_sub[:,0], coords_sub[:,1], coords_sub[:,2], color='red', lw=5, alpha=1)
    pts = np.array(pts)

    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    ax.set_adjustable("box")
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    offset = 0.

    ax.set_xlim(xmin+offset, xmax-offset)
    ax.set_ylim(ymin+offset, ymax-offset)
    ax.set_zlim(zmin+offset, zmax-offset)

    elev, azim = calc_best_viewpoint(pts)
    print(elev, azim)
    elev, azim = 90, 20
    ax.view_init(elev, azim)
    
    tick_label_size = 18

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(figname, dpi=200)
    plt.close('all')

    return len(files)

ctypes = ['CP_GPe-CP']
plot_main_tracts(ctypes[0])


