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
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from swc_handler import parse_swc


def estimate_radius2d(pts, method=0):
    '''
    :params method:
                    0: mean distance as radius
                    1: median distance as radius
                    2: 75% percentile as radius
                    3: max distance as radius
    '''
    pca = PCA()
    tpts = pca.fit_transform(pts)
    pts2d = tpts[:,:2]
    center = pts2d.mean(axis=0)
    shift = pts2d - center
    lengths = np.linalg.norm(shift, axis=1)
    if method == 0:
        radius = lengths.mean()
    elif method == 1:
        radius = np.median(lengths)
    elif method == 2:
        radius = np.percentile(lengths, 75)
    elif method == 3:
        radius = np.max(lengths)

    return radius


def truncated_cone(ax, p0, p1, R0, R1, color, alpha=0.5):
    """
    adapted from https://stackoverflow.com/questions/48703275/3d-truncated-cone-in-python
    """
    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R *
               np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=color, linewidth=0, antialiased=False, alpha=alpha)


def calc_best_viewpoint(pts):
    pca = PCA()
    pca.fit(pts)
    x,y,z = pca.components_[2]
    elev = np.rad2deg(np.arcsin(z))
    azim = np.rad2deg(np.arctan(x/y))
    return elev, azim


if False:
    def plot_main_tracts(class_name, key='tract.swc'):
        # visualize and check the path
        mpath_dir = '../main_tracts_types'
        show_pts = 200
        show_instances = 50
        figname = f'{class_name}_main_tract_vis.png'
        scale = 1000

        # intitalize the fig
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        # load all the main paths
        pts = []
        termini = []
        somata = []
        for pathfile in glob.glob(os.path.join(mpath_dir, f'{class_name}*{key}'))[:show_instances]:
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
            ax.scatter(coords_sub[0,0], coords_sub[0,1], coords_sub[0,2], marker='o', color='r', s=40)
            ax.scatter(coords_sub[-1,0], coords_sub[-1,1], coords_sub[-1,2], marker='^', color='k', s=40)
            ax.plot(coords_sub[:,0], coords_sub[:,1], coords_sub[:,2], alpha=0.5)

        # get the best view orientation
        #elev, azim = calc_best_viewpoint(pts)
        #ax.view_init(elev, azim)

        if False:
            # plot the projection cone
            vertice = np.array(somata).mean(axis=0)
            termini = np.array(termini)
            cone_b = termini.mean(axis=0)
            R = estimate_radius2d(termini, method=2)
            print(R)
            truncated_cone(ax, vertice, cone_b, 0, R, 'blue', alpha=0.2)
     
        cn_split = class_name.split('-')
        ptype = cn_split[0]
        stype = '-'.join(cn_split[1:])
        ax.set_title(f'{stype} --> {ptype}', fontsize=40, y=1.0)
        tick_label_size = 18
        ax.set_xlabel(r'X-coord ({}$\mu$m)'.format(scale), fontsize=tick_label_size)
        ax.set_ylabel(r'Y-coord ({}$\mu$m)'.format(scale), fontsize=tick_label_size)
        ax.set_zlabel(r'Z-coord ({}$\mu$m)'.format(scale), fontsize=tick_label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        #plt.legend()
        plt.tight_layout()
        plt.savefig(figname, dpi=200)
        plt.close()

    ctypes = ['CP_GPe-CP', 'CP_SNr-CP', 'CTX_ET-MOp', 'CTX_ET-MOs', 'CTX_ET-RSPv', 'CTX_ET-SSp-bfd', 'CTX_ET-SSp-m', 'CTX_ET-SSp-n', 'CTX_ET-SSp-ul', 'CTX_ET-SSs', 'CTX_IT-MOp', 'CTX_IT-MOs', 'CTX_IT-SSp-bfd', 'CTX_IT-SSp-m', 'CTX_IT-SSp-n', 'CTX_IT-SSs', 'CTX_IT-VISp', 'TH_core-LGd', 'TH_core-MG', 'TH_core-SMT', 'TH_core-VPL', 'TH_core-VPLpc', 'TH_core-VPM', 'TH_matrix-LP', 'TH_matrix-VM']
    #ctypes = ['TH_matrix-LP']
    
    for class_name in ctypes:
        print(f'--> Plotting for {class_name}...')
        plot_main_tracts(class_name)


if True:    # radius estimation along main tracts
    # params
    mpath_dir = '../main_tracts_types'
    show_pts = 200
    show_instances = 200

    np.set_printoptions(precision=2)


    # plot path evolution
    def calc_tract_radii1(class_name):
        pathfiles = list(glob.glob(os.path.join(mpath_dir, f'{class_name}*_tract.swc')))
        if show_instances < len(pathfiles):
            pathfiles = random.sample(pathfiles, show_instances)
        print(f'--> {class_name}: {len(pathfiles)}')

        pts = []
        min_npts = show_pts
        for pathfile in pathfiles:
            tree = parse_swc(pathfile)
            coords = np.array([node[2:5] for node in tree][::-1]) # soma to terminal
            
            nnodes = len(coords)
            step = max(nnodes // show_pts, 1)
            coords_sub = coords[::step]

            pts.append(coords_sub)
            if coords_sub.shape[0] < min_npts:
                min_npts = coords_sub.shape[0]
        
        pts = np.array(list([pt[:min_npts] for pt in pts]))
        radii = [estimate_radius2d(pts[:,i], method=2) for i in range(min_npts)]

        return radii

    def calc_tract_radii2(class_name):
        pathfiles = list(glob.glob(os.path.join(mpath_dir, f'{class_name}*_tract.swc')))
        if show_instances < len(pathfiles):
            pathfiles = random.sample(pathfiles, show_instances)
        print(f'--> {class_name}: {len(pathfiles)}')

        paths = []
        for pathfile in pathfiles:
            tree = parse_swc(pathfile)
            coords = np.array([node[2:5] for node in tree][::-1]) # soma to terminal
            paths.append(coords)

        path_select = 'median'
        if path_select == 'median':
            # find the path with median length
            termini = np.array([c[-1] for c in paths])
            tindices = np.array([len(c)-1 for c in paths])
            tm = termini.mean(axis=0)
            tdists = np.linalg.norm(termini - tm, axis=1)
            midx = np.argmin(tdists) #np.argsort(tdists)[(len(tdists) - 1) // 2]
            mpath = paths[midx]
        elif path_select == 'longest':
            lengths = np.array([len(c) for c in paths])
            tindices = np.array([len(c)-1 for c in paths])
            midx = np.argmax(lengths)
            mpath = paths[midx]

        
        # calculate the radius through nearest points matching
        dmins, imins = [], []
        for i, path in enumerate(paths):
            kdtree = KDTree(path, leaf_size=2)
            dmin, imin = kdtree.query(mpath, k=1)
            dmins.append(dmin[:,0])
            imins.append(imin[:,0])
        dmins = np.array(dmins).transpose()
        imins = np.array(imins).transpose()

        radii = []
        for pt, dmin, imin in zip(mpath, dmins, imins):
            tflags = imin != tindices
            if tflags.sum() < len(imin) / 2:
                break
            dists = dmin[tflags]
            dist = np.median(dists)
            radii.append(dist)
        #for dmin, imin in zip(dmins, imins):
        #    cur_pts = []
        #    for i, idx in enumerate(imin):
        #        pt = paths[i][idx]
        #        cur_pts.append(pt)
        #    radius = estimate_radius2d(cur_pts, method=2)
        #    radii.append(radius)

        print(f'{len(radii)} / {len(mpath)}')

        return radii

    def plot_tract_radii(class_name):
        radii = calc_tract_radii1(class_name)

        # plot
        plt.plot(np.linspace(0, 1, len(radii)), radii)
        plt.title(class_name, fontsize=18)
        plt.xlabel('Normalized distance to soma', fontsize=15)
        plt.ylabel(r'Radius of cross-section ($\mu$m)', fontsize=15)
        plt.savefig(f'{class_name}_radii.png', dpi=150)
        plt.close()

        return radii

    def plot_tracts_radii(ctypes, figname, radius_type=2):
        scale = 1000
        fig = plt.figure(figsize=(8,8))

        for class_name in ctypes:
            if radius_type == 1:
                radii = np.array(calc_tract_radii1(class_name)) / scale
            else:
                radii = np.array(calc_tract_radii2(class_name)) / scale
            plt.plot(np.linspace(0, 1, len(radii)), radii, label=class_name, linewidth=2)
        
        plt.title(figname, fontsize=40)
        plt.xticks([])
        if radius_type == 1:
            plt.yticks([0,1,2,3,4], fontsize=25)
            plt.ylim([0,4])
            plt.ylabel(r'Radius-method1 (1000$\mu$m)', fontsize=25)
        else:
            plt.yticks([0,1,2], fontsize=25)
            plt.ylim([0,2])
            plt.ylabel(r'Radius-method2 (1000$\mu$m)', fontsize=25)

        plt.xlabel('Normalized distance to soma', fontsize=25)
        plt.legend(loc='upper left', frameon=False)
        #plt.grid(which='major', linestyle='--', alpha=0.5)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)

        plt.tight_layout()
        plt.savefig(f'{figname}_radii.png', dpi=200)
        plt.close()


    # calculating
    type_dict = {
        'CP': ['CP_GPe-CP', 'CP_SNr-CP'],
        'CTX_ET': ['CTX_ET-MOp', 'CTX_ET-MOs', 'CTX_ET-RSPv', 'CTX_ET-SSp-bfd', 'CTX_ET-SSp-m', 'CTX_ET-SSp-n', 'CTX_ET-SSp-ul', 'CTX_ET-SSs'],
        'CTX_IT': ['CTX_IT-MOp', 'CTX_IT-MOs', 'CTX_IT-SSp-bfd', 'CTX_IT-SSp-m', 'CTX_IT-SSp-n', 'CTX_IT-SSs', 'CTX_IT-VISp'],
        'TH_core': ['TH_core-LGd', 'TH_core-MG', 'TH_core-SMT', 'TH_core-VPL', 'TH_core-VPLpc', 'TH_core-VPM'],
        'TH_matrix': ['TH_matrix-LP', 'TH_matrix-VM']
    }

    for cls_name, ctypes in type_dict.items():
        plot_tracts_radii(ctypes, cls_name, radius_type=2)
 

if False:    # visualize terminal of CTX_ET neurons

    # params
    mpath_dir = '../main_tracts_types'
    show_instances = 200

    def plot_CTX_ET_neurons(class_name):
        pathfiles = list(glob.glob(os.path.join(mpath_dir, f'{class_name}*_tract.swc')))
        if show_instances < len(pathfiles):
            pathfiles = random.sample(pathfiles, show_instances)
        print(f'--> {class_name}: {len(pathfiles)}')

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        pts = []
        for pathfile in pathfiles:
            tree = parse_swc(pathfile)
            pts.append(tree[0][2:5])
        pts = np.array(pts)

        ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker='^', color='k')
        plt.savefig(f'{class_name}.png', dpi=150)
        plt.close()


    ctx_type = 'CTX_ET-SSp-m'
    plot_CTX_ET_neurons(ctx_type)
    



