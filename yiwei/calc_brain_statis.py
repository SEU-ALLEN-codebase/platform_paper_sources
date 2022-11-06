

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
from collections import Counter

import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import filters
import seaborn as sns
from file_io import load_image, save_image, get_tera_res_path
from image_utils import get_mip_image

def cal_signal_ratio(signal_region_csv,mask_region):
    brain_region_count = np.count_nonzero(mask_region)
    signal_region_count = sum(signal_region_csv['count'].iloc[1:])
    signal_ratio = signal_region_count / brain_region_count

    return signal_ratio
    
def brain_signal_statistics(signal_region_csv):
    signal_statistics = np.zeros(2654)
    signal_region_count = sum(signal_region_csv['count'].iloc[1:])
    new_list = [x/signal_region_count for x in signal_region_csv['count'][1:].tolist()]
    signal_statistics [(signal_region_csv['region'][1:]-1).tolist()]  = new_list
    return signal_statistics

def signal_ratio_plot(signal_region_path, mask_region_path, outpath):
    result = []
    x_label = []
    for signal_file in glob.glob(os.path.join(signal_region_path, f'*.csv')):
        signal_region_csv = pd.read_csv(signal_file)
        brain_id = os.path.split(signal_file)[-1].split('.')[0]
        print(brain_id)
        print(mask_region_path +'/'+ brain_id + '.v3draw')
        mask_region = load_image(mask_region_path +'/'+ brain_id + '.v3draw')
        signal_ratio = cal_signal_ratio(signal_region_csv,mask_region)
        result.append(signal_ratio)
        x_label.append(brain_id)
    X_ticks = np.arange(0,len(result)) 
    plt.plot(X_ticks,result,linestyle='')
    plt.xticks(X_ticks,x_label)
    plt.ylabel('signal_ratio')
    plt.savefig(outpath+'/signal_ratio_plot.png',dpi=300)

def region_brainid_plot(signal_region_path, outpath): 
    result = []
    y_label = []   
    x_label = []
    for signal_file in glob.glob(os.path.join(signal_region_path, f'*.csv')):
        signal_region_csv = pd.read_csv(signal_file)
        brain_id = os.path.split(signal_file)[-1].split('.')[0]
        result.append(brain_signal_statistics(signal_region_csv))
        x_label.append(brain_id)
    X_ticks = np.arange(0,len(x_label))
    
    result = np.array(result) 

    sns.heatmap(result.transpose(),center=0.01)
    plt.xticks(X_ticks,x_label)
    plt.ylabel('brain_region')
    plt.savefig(outpath+'/region_brainid_plot.png',dpi=300)

def testify_threshold(tera_dir, res_id=-3, threshold=400,step=4,sep=100, outdir=None):
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
                img = filters.gaussian(img, (1,3,3), preserve_range=True)
                img2d = get_mip_image(img)
                max_pv = img2d.max()
                min_pv = img2d.min()
                # print(img2d.shape)
                if max_pv>300:
                    img_threshold = np.zeros((img2d.shape[0],(step+1)*img2d.shape[1]),dtype = np.uint8)  
                    for i in range(0,step):
                        img_tmp = np.zeros(img2d.shape,dtype = np.uint8)
                        img_tmp[ img2d > (threshold+sep*i) ] = 255 
                        img_threshold[:,(i+1)*img2d.shape[1]:(i+2)*img2d.shape[1]] = img_tmp  
                    # print(img_threshold.shape)         
                    img2d = ((img2d - min_pv) / (max_pv - min_pv + 1e-10) * 255).astype(np.uint8)
                    img_threshold[:,0:img2d.shape[1]] = img2d              
                    fname = os.path.split(imgfile)[-1]
                    prefix = os.path.splitext(fname)[0]
                    outfile = os.path.join(outdir, f'{prefix}_fs{fs:.3f}_vmax{max_pv}_vmin{min_pv}_threshold400.png')
                    # save_image(outfile, img2d)
                    save_image(outfile, img_threshold)

        counter += 1
        if counter % 100 == 0:
            print(f'--> Processed {counter} files in {time.time() - t0:.4f} seconds')
            
def get_filesize(tera_dir, res_id=-3, threshold=300, outdir=None):
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
                img_threshold = np.zeros(img2d.shape[0],dtype = np.uint8)
                img_threshold[ img2d > threshold ]=255
                
                max_pv = img2d.max()
                min_pv = img2d.min()
                img2d = ((img2d - min_pv) / (max_pv - min_pv + 1e-10) * 255).astype(np.uint8)
                
                fname = os.path.split(imgfile)[-1]
                prefix = os.path.splitext(fname)[0]
                outfile = os.path.join(outdir, f'{prefix}_fs{fs:.3f}_vmax{max_pv}_vmin{min_pv}.png')
                outfile1 = os.path.join(outdir, f'{prefix}_fs{fs:.3f}_threshold{threshold}.png')
                # save_image(outfile, img2d)
                save_image(outfile1, img_threshold)

        counter += 1
        if counter % 100 == 0:
            print(f'--> Processed {counter} files in {time.time() - t0:.4f} seconds')

    fsizes = np.array(fsizes)
    sns.histplot(fsizes, bins=200, binrange=(fsizes.min(),4))
    plt.ylim([0,300])
    plt.xlabel('TeraFly block image size')
    plt.savefig(f'filesize_distr.png', dpi=300)
    


class CalcBrainStatis(object):
    def __init__(self, tera_dir, res_id_statis=-3):
        self.tera_dir = tera_dir
        self.res_id_statis = res_id_statis
        assert (res_id_statis < 0)  # -1: the highest resolution, -2: second-highest, -3,...,
        self.multiplier = np.power(2, np.fabs(res_id_statis)-1)

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
        for imgfile in glob.glob(os.path.join(res_path, '*/*/[0-9]*[0-9].tif')):
            img = load_image(imgfile)
            vmin_, vmax_ = img.min(), img.max()
            vmax = max(vmax_, vmax)
            vmin = min(vmin_, vmin)
        return vmax, vmin


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

        # pos_mapped_z = np.clip(pos_mapped_z, 0, self.mask_dims[2]-1)
        pos_mapped_y = np.round((fg_pos[1] * self.multiplier  + start_res[0]/10) / self.coord_factor[0]).astype(np.int32)
        # pos_mapped_y = np.clip(pos_mapped_y, 0, self.mask_dims[0]-1)
        pos_mapped_x = np.round((fg_pos[2] * self.multiplier  + start_res[1]/10) / self.coord_factor[1]).astype(np.int32)
        # pos_mapped_x = np.clip(pos_mapped_x, 0, self.mask_dims[1]-1)
        
        #save_mask[0,pos_mapped_z-1, pos_mapped_y-1, pos_mapped_x-1] = self.region_mask[0, pos_mapped_z-1, pos_mapped_y-1, pos_mapped_x-1]
        regions = self.region_mask[0, pos_mapped_z-1, pos_mapped_y-1, pos_mapped_x-1]
        mask_id0_idx = np.where(regions==0)
        print("mask_id ==0, [z, y, x]:", [pos_mapped_z[mask_id0_idx], pos_mapped_y[mask_id0_idx], pos_mapped_x[mask_id0_idx]])
            
        
        # then you can use the regions of foreground voxels
        
        region_counter = Counter(regions)
        
        return region_counter

    def brain_statis(self, filesize_thresh=1.7, vmax_thresh=300):
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
                idx_file += 1

            vmax = img.max()
            if vmax < vmax_thresh:
                n_lowQ_block += 1
                continue

            n_highQ_block += 1
            cur_counter = self.block_statis(block_file, img, vmax_thresh)
            # print(brain_counter)
            # print(cur_counter)
            brain_counter = brain_counter + cur_counter
            
        # save_image('D:/22spring/cal_brain_stats/pylib-main/brain_counter/15257.tif',save_mask)    
        
        print(f'<-Summary of current brain: -> small: {n_small_block}, low-quality: {n_lowQ_block}, high-quality: {n_highQ_block}')
        
        return brain_counter


def brain_statis_wrapper(tera_dir, mask_file_dir, out_dir, max_res_dims, mask_dims, filesize_thresh, vmax_thresh):
    brain_id = int(os.path.split(tera_dir)[-1].split('_')[0][5:])
    print(f'===> Processing {brain_id}')
    mask_file = os.path.join(mask_file_dir, f'{brain_id}.v3draw')
    mask = load_image(mask_file)

    cbs = CalcBrainStatis(tera_dir)
    cbs.set_region_mask(mask, max_res_dims, mask_dims)
    brain_counter = cbs.brain_statis(filesize_thresh=filesize_thresh, vmax_thresh=vmax_thresh)
    
    with open(os.path.join(out_dir, f'{brain_id}.csv'), 'w', newline='') as f:
        header = ['region','count'] #csv列名
        writer = csv.writer(f)
        writer.writerow(header)
        for key,value in brain_counter.items():
            writer.writerow([key,value])
        



if __name__ == '__main__':
    from multiprocessing.pool import Pool

    tera_downsize_file = 'D:/22spring/cal_brain_stats/pylib-main/TeraDownsampleSize.csv'
    tera_path = 'Z:/TeraconvertedBrain'
    mask_file_dir = 'Z:/SEU-ALLEN/Users/ZhixiYun/data/registration/Inverse'
    out_dir = 'D:/22spring/cal_brain_stats/pylib-main/brain_counter'
    threshold = 300
    res_ids = -3
    filesize_thresh = 1.7
    vmax_thresh = 400
    nproc = 4
    signal_region_path = 'Z:/SEU-ALLEN/Users/YiweiLi/Projects/platform_paper/brain_statistic'
    fig_outpath = 'Z:/SEU-ALLEN/Users/YiweiLi/Projects/platform_paper/brain_statistic_fig'
    region_brainid_plot(signal_region_path,fig_outpath)
    signal_ratio_plot(signal_region_path,mask_file_dir,fig_outpath)
 
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
   
    # dim_f = pd.read_csv(tera_downsize_file, index_col='ID')
    # args_list = []
    # i = 0 
    # for tera_dir in glob.glob(os.path.join(tera_path, f'mouse[1-9]*[0-9]*')):
    #     brain_id = int(os.path.split(tera_dir)[-1].split('_')[0][5:])
    #     max_res_dims = np.array([dim_f.loc[brain_id][0],dim_f.loc[brain_id][1],dim_f.loc[brain_id][2]])
    #     mask_dims = np.array([dim_f.loc[brain_id][3],dim_f.loc[brain_id][4],dim_f.loc[brain_id][5]])
        
    #     args = tera_dir, mask_file_dir, out_dir, max_res_dims, mask_dims, filesize_thresh, vmax_thresh
    #     i = i+1
    #     if i >=113:
    #         args_list.append(args)
              
    # print(f'Number of brains to process: {len(args_list)}')
    # pt = Pool(nproc)
    # pt.starmap(brain_statis_wrapper, args_list)
    # pt.close()
    # pt.join()
    
    
    # brain_id1 = ['196472','15702','18455','18872','182712','201606','236174']
    # for brain_id in brain_id1:
    #     tera_dir = f'Z:/TeraconvertedBrain/mouse{brain_id}_teraconvert'
    #     outdir = f'Z:/SEU-ALLEN/Users/YiweiLi/brain_mip/{brain_id}/mip2d/'

    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)
        
    #     testify_threshold(tera_dir, -3,400,4,100,outdir=outdir)



