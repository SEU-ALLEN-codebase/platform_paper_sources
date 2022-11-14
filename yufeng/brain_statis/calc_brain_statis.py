#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : calc_brain_statis.py
#   Author       : Yufeng Liu
#   Date         : 2022-10-20
#   Description  : 
#
#================================================================

import os
import sys
import glob
import time
import csv
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from file_io import load_image, save_image, get_tera_res_path
from image_utils import get_mip_image

def get_filesize(tera_dir, res_id=-3, outdir=None):
    np.random.seed(1024)

    res_path = get_tera_res_path(tera_dir, res_id, False)
    fsizes = []
    counter = 0
    t0 = time.time()
    for imgfile in glob.glob(os.path.join(res_path, '*/*/[0-9]*[0-9].tif')):
        fs = os.path.getsize(imgfile) / 1000. / 1000.
        fsizes.append(fs)

        # randomly save 2D MIP images for checking and debugging
        output_min_size = 0.4
        output_max_size = 10.0
        prob = 0.2
        if (fs < output_max_size) and (fs > output_min_size):
            if np.random.random() < prob:
                img = load_image(imgfile)
                img2d = get_mip_image(img)
                max_pv = img2d.max()
                min_pv = img2d.min()
                img2d = ((img2d - min_pv) / (max_pv - min_pv + 1e-10) * 255).astype(np.uint8)
                
                fname = os.path.split(imgfile)[-1]
                prefix = os.path.splitext(fname)[0]
                outfile = os.path.join(outdir, f'{prefix}_fs{fs:.3f}_vmax{max_pv}.png')
                save_image(outfile, img2d)

        counter += 1
        if counter % 100 == 0:
            print(f'--> Processed {counter} files in {time.time() - t0:.4f} seconds')

    fsizes = np.array(fsizes)
    sns.histplot(fsizes, bins=200, binrange=(fsizes.min(),4))
    plt.ylim([0,300])
    plt.xlabel('TeraFly block image size')
    plt.savefig(f'filesize_distr.png', dpi=300)

def get_block_counts(zdim, ydim, xdim, h=5, d=3):
    dh = (d+1) * h
    yv, zv, xv = np.meshgrid(np.arange(ydim), np.arange(zdim), np.arange(xdim))
    yc = np.maximum(dh - yv - 1, 0)//h + np.maximum(dh - (ydim - yv), 0)//h
    zc = np.maximum(dh - zv - 1, 0)//h + np.maximum(dh - (zdim - zv), 0)//h
    xc = np.maximum(dh - xv - 1, 0)//h + np.maximum(dh - (xdim - xv), 0)//h
    bc = 2 * d * 3 + 1 - xc - yc - zc
    #print(bc.max(), bc.min(), bc.mean(), bc[:11,:11,:11], yc[:11,:11,:11], zc[:11,:11,:11], xc[:11,:11,:11])
    bc = torch.from_numpy(np.expand_dims(np.expand_dims(bc, 0), 0).astype(np.float32))
    return bc

def ada_thresholding(img, block_counts, h=5, d=3, cuda=True):
    imgt = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    k = 2 * d + 1
    weight = torch.ones((1,1,k,k,k))
    if cuda:
        imgt = imgt.cuda()
        weight = weight.cuda()
    conved = F.conv3d(imgt, weight, dilation=h, padding=d*h)
    if (conved.shape[2] != block_counts.shape[2]) or \
        (conved.shape[3] != block_counts.shape[3]) or \
        (conved.shape[4] != block_counts.shape[4]):
        block_counts = get_block_counts(*img.shape)
        if cuda:
            block_counts = block_counts.cuda()
    conved = conved / block_counts
    diff = imgt - conved
    #print(diff.max(), diff.min(), diff.abs().mean())
    conved[diff < 0] = 0

    if cuda:
        conved = conved.cpu()
    conved = conved.numpy()[0][0]
    return conved

class CalcBrainStatis(object):
    def __init__(self, tera_dir, res_id_statis=-3, mip_dir='', cuda=True):
        self.tera_dir = tera_dir
        self.res_id_statis = res_id_statis
        assert (res_id_statis < 0)  # -1: the highest resolution, -2: second-highest, -3,...,
        self.multiplier = np.power(2, np.fabs(res_id_statis)-1)
        self.mip_dir = mip_dir
        self.cuda = cuda

    def set_region_mask(self, region_mask, max_res_dims, mask_dims):
        """
        max_res_dims & mask_dims: np.ndarray, in order [y, x, z]. Do not get it wrong!
        """
        self.region_mask = region_mask
        self.mask_dims = mask_dims
        # self.coord_factor = max_res_dims / mask_dims / self.multiplier
        self.coord_factor = max_res_dims / mask_dims 

    def set_strides_through_block(self, block):
        """
        dimension of each block, in order [y, x, z]. WARNING: The order is unconventional! 
        """
        zs, ys, xs = block.shape[-3:]  # in case multi-channel images
        self.strides = np.array([ys, xs, zs])   # to numpy array batch-computing convenience
        # initialize the local coordinates of current block
        #zz, yy, xx = np.meshgrid(np.arange(ys), np.arange(zs), np.arange(xs))

    def get_image_range(self, idx=0):
        """
        Pre-estimate the overall statistics of the brain, using the lowest resolution to speed up
        """
        res_path = get_tera_res_path(self.tera_dir, idx, False)
        vmin = 2**16
        vmax = 0
        vmean = []
        vstd = []
        for imgfile in glob.glob(os.path.join(res_path, '*/*/[0-9]*[0-9].tif')):
            img = load_image(imgfile)
            vmin_, vmax_ = img.min(), img.max()
            vmean.append(img.mean())
            vstd.append(img.std())
            vmax = max(vmax_, vmax)
            vmin = min(vmin_, vmin)

        vmean = np.median(vmean)
        vstd = np.median(vstd)
        return vmax, vmin, vmean, vstd


    def block_statis(self, block_path, img, thresh):
        """
        @args: thresh is for segmenting the background from foreground
        """
        # infer the coordinate from the block_path
        fname = os.path.splitext(os.path.split(block_path)[-1])[0]
        start_hres = np.array(list(map(int, fname.split('_')))) # y-x-z order
        start_res = start_hres
        # assert int(start_hres[0] / self.multiplier) == start_hres[0] / self.multiplier
        # start_res = start_hres // self.multiplier
        # print(start_hres)
        #img = load_image(block_path)    # Reminder: TeraFly use uint16 by default
        assert img.ndim == 3, "Only 3 dimensional image is supported!"
        # signal binary 
        img_bin = np.zeros(img.shape, dtype=np.uint8)
        img_bin[img > thresh] = 1
        # now you should get brain regions from the mask image
        fg_pos = np.nonzero(img_bin)

        pos_mapped_z = np.round((fg_pos[0] * self.multiplier  + start_res[2]/10) / self.coord_factor[2]).astype(np.int32)

        pos_mapped_z = np.clip(pos_mapped_z, 0, self.mask_dims[2]-1)
        pos_mapped_y = np.round((fg_pos[1] * self.multiplier  + start_res[0]/10) / self.coord_factor[0]).astype(np.int32)
        pos_mapped_y = np.clip(pos_mapped_y, 0, self.mask_dims[0]-1)
        pos_mapped_x = np.round((fg_pos[2] * self.multiplier  + start_res[1]/10) / self.coord_factor[1]).astype(np.int32)
        pos_mapped_x = np.clip(pos_mapped_x, 0, self.mask_dims[1]-1)

        regions = self.region_mask[0,pos_mapped_z, pos_mapped_y, pos_mapped_x]
        # then you can use the regions of foreground voxels
        region_counter = Counter(regions)
        return region_counter

    def brain_statis(self, filesize_thresh=1.7, vmax_thresh=300, save_mip=True):
        res_path = get_tera_res_path(self.tera_dir, res_ids=self.res_id_statis, bracket_escape=False)
        n_processed = 0
        idx_file = 0
        n_small_block = 0
        n_lowQ_block = 0
        n_highQ_block = 0
        display_freq = 10
        t0 = time.time()
        brain_counter = Counter()
        for block_file in glob.glob(os.path.join(res_path, '*/*/[0-9]*[0-9].tif')):
            n_processed += 1
            if n_processed % display_freq == 0:
                print(f'--> Procssed: {n_processed} in {time.time() - t0:.4f}s. \t\tStatis: small: {n_small_block}, low-quality: {n_lowQ_block}, high-quality: {n_highQ_block}')

            fs = os.path.getsize(block_file) / 1000. / 1000.
            if fs < filesize_thresh:
                n_small_block += 1
                continue

            # filter with vmax
            img = load_image(block_file)
            if idx_file == 0:   # get the dimension of each block, only once!
                self.set_strides_through_block(img)
                self.bc = get_block_counts(self.strides[2], self.strides[0], self.strides[1])
                if self.cuda:
                    self.bc = self.bc.cuda()
                idx_file += 1

            vmax = img.max()
            if vmax < vmax_thresh:
                n_lowQ_block += 1
                continue

            n_highQ_block += 1
            # do ada_thresholding
            img_a = ada_thresholding(img, self.bc, cuda=self.cuda)
            cur_counter = self.block_statis(block_file, img_a, vmax_thresh)
            brain_counter = brain_counter + cur_counter

            if save_mip and np.random.random() < 0.05:
                # save mip for inspection
                img2d = get_mip_image(img)
                max_pv = img2d.max()
                min_pv = img2d.min()
                img_thr = np.zeros(img2d.shape, dtype=np.uint8)
                img_thr[img2d > vmax_thresh] = 255

                # normalize for better visualization
                img2d = ((img2d - min_pv) / (max_pv - min_pv + 1e-10) * 255).astype(np.uint8)
                img2d_merge = np.hstack((img2d, img_thr))

                fname = os.path.split(block_file)[-1]
                prefix = os.path.splitext(fname)[0]
                outfile = os.path.join(self.mip_dir, f'{prefix}_fs{fs:.3f}_vmax{max_pv}_thresh{vmax_thresh:.1f}.png')
                save_image(outfile, img2d_merge)

        print(f'<-Summary of current brain: -> small: {n_small_block}, low-quality: {n_lowQ_block}, high-quality: {n_highQ_block}')

        return brain_counter

def brain_statis_wrapper(tera_dir, mask_file_dir, out_dir, max_res_dims, mask_dims, filesize_thresh, brain_id, res_ids=-3, source='fMOST-Zeng', cuda=True):
    csv_out = os.path.join(out_dir, f'{brain_id}.csv')
    if os.path.exists(csv_out):
        return 

    print(f'===> Processing {brain_id}')
    mask_file = os.path.join(mask_file_dir, f'{brain_id}.v3draw')
    mask = load_image(mask_file)

    mip_dir = os.path.join(out_dir, f'mip2d_{brain_id}')
    if not os.path.exists(mip_dir):
        os.mkdir(mip_dir)
    cbs = CalcBrainStatis(tera_dir, mip_dir=mip_dir, cuda=cuda, res_id_statis=res_ids)
    cbs.set_region_mask(mask, max_res_dims, mask_dims)
    # statistics
    _, _, vmean, vstd = cbs.get_image_range()
    if source == 'fMOST-Zeng':
        vmax_thresh = min(max(vmean + 1.5 * vstd, 400), 1000)
        print(vmean, vstd, vmax_thresh)
    else:
        vmax_thresh = 800

    brain_counter = cbs.brain_statis(filesize_thresh=filesize_thresh, vmax_thresh=vmax_thresh)
    #print(f'{brain_id}: {cbs.get_image_range()}')
    
    if len(brain_counter) > 0:
        with open(csv_out, 'w', newline='') as f:
            header = ['region','count'] #csv列名
            writer = csv.writer(f)
            writer.writerow(header)
            for key,value in brain_counter.items():
                writer.writerow([key,value])
        



if __name__ == '__main__':
    from multiprocessing.pool import Pool

    tera_downsize_file = './ccf_info/TeraDownsampleSize.csv'
    mask_file_dir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/registration/Inverse'
    source = 'fMOST-Huang'
    out_dir = f'./statis_out_adaThr/{source}'
    res_ids = -3
    filesize_thresh = 1.7
    nproc = 1

    if source == 'fMOST-Zeng':
        match_str = 'mouse*[0-9]'
        tera_path = '/PBshare/TeraconvertedBrain'
    elif source == 'fMOST-Huang':
        match_str = 'mouse*[0-9]'
        tera_path = '/PBshare/Huang_Brains'
    elif source == 'STPT-Huang':
        match_str = '[1-9]*processed'
        tera_path = '/PBshare/Huang_Brains'
    elif source == 'LSFM-Wu':
        tera_path = '/PBshare/Zhuhao_Wu'
        match_str = 'WHOLE_mouse_B*'
    elif source == 'LSFM_Osten':
        pass
    elif source == 'LSFM_Dong':
        tera_path = '/PBshare/DongHW_Brains/20220315_SW220203_03_LS_6x_1000z'
        match_str = '*TeraFly'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
   
    dim_f = pd.read_csv(tera_downsize_file, index_col='ID', sep=',')
    args_list = []
    for tera_dir in glob.glob(os.path.join(tera_path, match_str)):
        if source == 'fMOST-Zeng':
            brain_id = int(os.path.split(tera_dir)[-1].split('_')[0][5:])
        elif source == 'fMOST-Huang':
            if os.path.exists(os.path.join(tera_dir, 'terafly')):
                tera_dir = os.path.join(tera_dir, 'terafly')
                brain_folder = os.path.split(os.path.split(tera_dir)[0])[-1]
            else:
                brain_folder = os.path.split(tera_dir)[-1]
            brain_id = int(brain_folder[-6:])
        elif source == 'LSFM-Wu':
            pass

        max_res_dims = np.array([dim_f.loc[str(brain_id)][0],dim_f.loc[str(brain_id)][1],dim_f.loc[str(brain_id)][2]])
        mask_dims = np.array([dim_f.loc[str(brain_id)][3],dim_f.loc[str(brain_id)][4],dim_f.loc[str(brain_id)][5]])
            
        args = tera_dir, mask_file_dir, out_dir, max_res_dims, mask_dims, filesize_thresh, brain_id, res_ids, source
        #brain_statis_wrapper(*args)
        args_list.append(args)
    
    #sys.exit()
    print(f'Number of brains to process: {len(args_list)}')
    pt = Pool(nproc)
    pt.starmap(brain_statis_wrapper, args_list)
    pt.close()
    pt.join()



