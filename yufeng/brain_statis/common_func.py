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

from anatomy.anatomy_core import parse_regions316

def load_regions(num_regions=70):
    # the load target regions
    regions = parse_regions316()
    if num_regions == 70:
        rids_set = set(regions['parent_id'])
        # take care of fiber tracts, with is a subset of root
        rids_set.remove(997)
        rids_set.add(1009)
    else:
        rids_set = set(regions['structure ID'])

    return rids_set

def get_region_mapper(coarse_regions, fine_regions, ana_dict):
    """
    The item of regions is index in u32
    """

    rc_dict = {}
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

                found = True
                break

    return rc_dict

    

