#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : common_func.py
#   Author       : Yufeng Liu
#   Date         : 2022-12-20
#   Description  : 
#
#================================================================

import numpy as np
from anatomy.anatomy_core import parse_regions316

def load_regions(level=0, remove_fiber_tracts=True):
    """
    @params level
                    - 0: 316
                    - 1: 70
    """

    # the load target regions
    regions = parse_regions316()
    if level == 1:
        mask = regions['parent_id'] == 315
        rids_set = np.hstack((regions['parent_id'][~mask].to_numpy(), regions['structure ID'][mask].to_numpy()))
        rids_set = set(rids_set.tolist())
        # take care of fiber tracts, with is a subset of root
        rids_set.remove(997)
        if not remove_fiber_tracts:
            rids_set.add(1009)
    else:
        rids_set = set(regions['structure ID'])
        if remove_fiber_tracts:
            rids_set.remove(1009)   # remove fiber tract

    return rids_set

def get_region_mapper(coarse_regions, fine_regions, ana_dict, reverse_mapper=False):
    """
    The item of regions is index in u32
    """

    rc_dict = {}
    rrc_dict = {}
    for idx in fine_regions:
        if idx == 0:
            continue
        id_path = ana_dict[idx]['structure_id_path'][::-1]  # self to parent
        found = False
        for ip in id_path:
            if ip in coarse_regions:
                if ip in rc_dict:
                    rc_dict[ip].append(idx)
                else:
                    rc_dict[ip] = [idx]
                rrc_dict[idx] = ip

                found = True
                break
    if reverse_mapper:
        return rc_dict, rrc_dict
    return rc_dict

    

