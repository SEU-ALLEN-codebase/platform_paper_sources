#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : select_neuron_types.py
#   Author       : Yufeng Liu
#   Date         : 2023-05-17
#   Description  : 
#
#================================================================

import os
import glob
import pandas as pd

import sys
sys.path.append('../../common_lib')
from common_utils import struct_dict, CorticalLayers, PstypesToShow


def select_neurons_by_sptype(rdict, df):
    #regions = ['MOp-ET', 'MOs-ET', 'SSp-bfd-ET', 'SSp-m-ET', 'MOp-IT', 'MOs-IT', 'SSp-bfd-IT', 'SSp-m-IT']
    
    r2n_dict = {}
    regions = []
    for area, regs in PstypesToShow.items():
        if area != 'CTX':
            continue
        for region in regs:
            if area == 'CTX':
                stype = region[:-3]
                pt = f'CTX_{region[-2:]}'
            elif area == 'TH':
                stype = region.split('-')[0]
                pt = f'TH_{region.split("-")[-1]}'
            else:
                area = region.split('-')[0]
                pt = 'Error'
            mask = (df.Manually_corrected_soma_region == stype) & (df.Subclass_or_type == pt)
            r2n_dict[region] = df[mask]['Cell name']

            regions.append(region)

    return r2n_dict, regions   


def select_neurons_by_sltype(rdict, df):
    regions = CorticalLayers
    regions_set = set(regions)
    
    r2n_dict = {}
    for region in regions_set:
        stype = '-'.join(region.split('-')[:-1])
        lt = region.split('-')[-1]
        #print(stype, lt)
        mask = (df.Manually_corrected_soma_region == stype) & (df.Cortical_layer == lt)
        r2n_dict[region] = df[mask]['Cell name']

    return r2n_dict, regions   


def select_neurons_by_stype(rdict, df, only_pltypes=False):
    regions = struct_dict['CTX'] + struct_dict['TH'] + struct_dict['STR']
    
    r2n_dict = {}
    if only_pltypes:
        r1, _ = select_neurons_by_sptype(rdict, df)
        r2, _ = select_neurons_by_sltype(rdict, df)
        nsets1 = set([n for ns in r1.values() for n in ns])
        nsets2 = set([n for ns in r2.values() for n in ns])
        nsets = nsets1 & nsets2
        for area in ['CTX', 'TH', 'STR']:
            for region in struct_dict[area]:
                if area == 'CTX':
                    if region == 'AId': 
                        continue    # Car3 region
                    r2n_dict[region] = df[(df.Manually_corrected_soma_region == region) & (df['Cell name'].isin(nsets))]['Cell name']
                else:
                    r2n_dict[region] = df[df.Manually_corrected_soma_region == region]['Cell name']
        print(r2n_dict)
    else:
        for region in regions:
            r2n_dict[region] = df[df.Manually_corrected_soma_region == region]['Cell name']

    return r2n_dict, regions

