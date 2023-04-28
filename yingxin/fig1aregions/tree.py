#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : tree.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-23
#   Description  : 
#
#================================================================

import os,glob
import json
import pickle
import copy
import math
import numpy as np
import pandas as pd
import seaborn as sns

def write_json(write_dict,dict_file):
    dict_object = json.dumps(write_dict)
    with open(dict_file, "w") as f:
        f.write(dict_object)
        
def load_json(dict_file):
    with open(dict_file, "r") as f:
        load_dict = json.loads(f.read())
    return load_dict

def load_json_key2int(dict_file):
    with open(dict_file, "r") as f:
        load_dict = json.loads(f.read())
    new_dict = {}
    for k,v in load_dict.items():
        new_dict[int(k)] = v
    return new_dict

def load_json_keys2int(dict_file):
    with open(dict_file, "r") as f:
        load_dict = json.loads(f.read())
    new_dict = {}
    for k,v in load_dict.items():
        new_dict[k] = {}
        for kk,vv in v.items():
            new_dict[k][int(kk)] = vv
    return new_dict

def get_tree():
    with open('../assets/tree_yzx.json','r') as f:
        tree = json.loads(f.read())
    return tree

def get_tree_from(info,hold=[]):
    info_tree = {}
    tree = get_tree()
    for t in tree:
        if (not hold) or (t[info] in hold):
            info_tree[t[info]] = t  
    return info_tree

def load_level_u32csv():
    level_file = '../assets/level.csv'
    level_df = pd.read_csv(level_file)
    return level_df

def get_n1327_n4_u32dict():
    level_df = load_level_u32csv()
    n1327_n5_u32dict = dict(zip(level_df['n1327_u32_id'].tolist(),level_df['n4_u32_id'].tolist()))
    return n1327_n5_u32dict                          

def get_u32_u16_id_dict():
    u32_u16_id_dict_file = f'../assets/u32_u16_dict.json'
    u32_u16_id_dict = load_json_key2int(u32_u16_id_dict_file)
    return u32_u16_id_dict

def get_u16_u32_id_dict():
    u16_u32_id_dict_file = '../assets/u16_u32_dict.json'
    u16_u32_id_dict = load_json_key2int(u16_u32_id_dict_file)
    return u16_u32_id_dict

def get_voxel_center_neighbor_dict(level,uint):
    dict_file = f'../assets/n{level}_u{uint}_voxel_center_neighbor_dict.json'
    load_dict = load_json_keys2int(dict_file)
    return load_dict

def get_dict(level,uint,key=['voxel','center','neighbor'][0]):
    voxel_center_neighbor_dict = get_voxel_center_neighbor_dict(level,uint)
    get_dict = voxel_center_neighbor_dict[f'n{level}_u{uint}_{key}_dict']
    return get_dict

def get_region_name_list(region_unit_id_list,uint):
    u32_tree = get_tree_from('id')
    u16_u32_id_dict = get_u16_u32_id_dict()
    region_name_list = []
    for region_unit_id in region_unit_id_list:
        if uint==32:
            try:
                region_name = u32_tree[region_unit_id]['acronym']
            except:
                region_name = ''
        else:
            try:
                ends = ['_l','_r'][region_unit_id%2]
                region_u32_id = u16_u32_id_dict[region_unit_id]
                region_name = u32_tree[region_u32_id]['acronym'] + ends 
            except:
                region_name = ''
        region_name_list.append(region_name)
    return region_name_list
    
def get_region_rgb_list(region_uint_list,uint=32):
    region_rgb_list = []
    u32_id_tree = get_tree_from('id')
    u16_u32_id_dict = get_u16_u32_id_dict()
    for region_uint in region_uint_list:
        if uint==16:
            region_uint_id = u16_u32_id_dict[region_unit_id]
        try:
            region_rgb_list.append(u32_id_tree[region_uint]['rgb_triplet'])
        except:
            region_rgb_list.append([255,255,255])
    return region_rgb_list

def color_map(map_dict,colors):
    values = set(list(map_dict.values()))
    colors = (colors*10)[:len(values)]
    value_color_dict = dict(zip(values,colors))
    print(value_color_dict)
    new_dict = dict([(k,value_color_dict[v]) for k,v in map_dict.items()])
    return new_dict
