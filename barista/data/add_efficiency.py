#! /usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import os
import sys
import math
import concurrent.futures
import traceback
#import gzip
#import pickle
#import json
import time
#import numexpr
#import array
from functools import partial
import re

import uproot
import numpy as np
from coffea import hist
from coffea import util
import coffea.lookup_tools as lookup_tools
import coffea.processor as processor
import awkward
import copy
import glob
from coffea.analysis_objects import JaggedCandidateArray

from brazil.aguapreta import *
import brazil.dataframereader as dataframereader
from brazil.Bcand_accumulator import Bcand_accumulator

np.set_printoptions(threshold=np.inf)

'''
Adds efficiencies as a new column to Bcands arrays, from the bulk processing stage.
Assumes some folder/file structure, saves a new set of coffea files to a subfolder
'''

import re
re_btype = re.compile("(?P<btype>Bu|Bs|Bd)")
re_side = re.compile("(?P<side>tag|probe)")
re_trigger = re.compile(f"(?P<trigger>{'|'.join(['HLT_Mu7_IP4', 'HLT_Mu9_IP5', 'HLT_Mu9_IP6', 'HLT_Mu12_IP6'])})")

def add_efficiency(coffea_file, test=False):
    print("\nProcessing file {}".format(coffea_file))
    if test:
        print("DEBUG : add_efficiency({})".format(coffea_file))
        return

    # Load efficiency histograms into coffea evaluator
    extractor = lookup_tools.extractor()
    for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
        for btype in ["Bu", "Bs", "Bd"]:
            for side in ["tag", "probe"]:
                extractor.add_weight_sets([f"eff_{btype}_{side}_{trigger} h_trigeff2D_{btype}_{side}_{trigger} /home/dyu7/BFrag/data/efficiency/efficiency2D.root"])
    extractor.finalize()
    evaluator = extractor.make_evaluator()

    # Loop over Bcands in coffea file
    # Dictionary: <obj_name> : <dataset_subjobN> : <column name> : <values array>
    coffea_stuff = util.load(coffea_file)
    for key1 in coffea_stuff:
        if not "Bcands" in key1:
            continue

        if "_opt" in key1:
            continue

        # Make new defaultdict_accumulator with new columns
        new_Bcands = processor.defaultdict_accumulator(
            partial(Bcand_accumulator, cols=[ "pt", "eta", "y", "phi", "mass", "eff", "w_eff"]))

        #print(f"\n *** Processing {key1} ***")
        # Figure out which efficiency to apply
        match_btype = re_btype.search(key1)
        if not match_btype:
            raise ValueError("Regex btype failed on key {}".format(key1))
        btype = match_btype.group("btype")

        match_side = re_side.search(key1)
        if not match_side:
            raise ValueError("Regex side failed on key {}".format(key1))
        side = match_side.group("side")

        match_trigger = re_trigger.search(key1)
        if not match_trigger:
            raise ValueError("Regex trigger failed on key {}".format(key1))
        trigger = match_trigger.group("trigger")

        for key2 in coffea_stuff[key1]: # subjobs
            #print(key2)
            #print(Bcands_obj["y"])
            coffea_stuff[key1][key2].add_column("eff", np.array(
                evaluator[f"eff_{btype}_{side}_{trigger}"](coffea_stuff[key1][key2]["pt"].value, abs(coffea_stuff[key1][key2]["y"].value))
            ))
            coffea_stuff[key1][key2].add_column("w_eff", np.array(
                1. / max(evaluator[f"eff_{btype}_{side}_{trigger}"](coffea_stuff[key1][key2]["pt"].value, abs(coffea_stuff[key1][key2]["y"].value), 1.e-20))
            ))

            new_Bcands[key2] = coffea_stuff[key1][key2]
        coffea_stuff[key1] = new_Bcands

    output_folder = os.path.dirname(coffea_file) + "/add_eff"
    os.system("mkdir -pv {}".format(output_folder))
    output_file = f"{output_folder}/{os.path.basename(coffea_file)}"
    print("Saving to {}".format(output_file))
    #print("Dry run: I would be saving to...")
    #print(output_file)
    util.save(coffea_stuff, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add coffea subjobs back together")
    parser.add_argument("-d", "--dir", type=str, help="Folder containing subjobs")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError("Directory {} does not exist".format(args.dir))

    print("Adding efficiencies to coffea subjob output files...")

    runpart_dirs = sorted(glob.glob(f"{args.dir}/Run2018*_part*/"))
    runparts = [x.split("/")[-2] for x in runpart_dirs]
    addeff_args = []
    for runpart in runparts:
        addeff_args.extend([[x] for x in glob.glob(f"{args.dir}/{runpart}/DataHistograms_Run*_part*subjob*.coffea")])

    import concurrent.futures
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
        futures_set = set(executor.submit(add_efficiency, *argset) for argset in addeff_args)
        njobs = len(futures_set)
        start = time.time()
        nfinished = 0
        while len(futures_set) > 0:
            finished = set(job for job in futures_set if job.done())
            futures_set.difference_update(finished)
            nfinished += len(finished)
            for fjob in finished:
                try:
                    fjob.result()
                except Exception as e:
                    print("Caught exception:")
                    print(e)
                    print(traceback.print_exc())
                    #print(traceback.format_exc(e))
                #except Exception as e:
                #    print("Caught exception")
                #    print(e)
            print("Finished: {} / {}".format(nfinished, njobs))
            print("Outstanding: {} / {}".format(len(futures_set), njobs))
            print("Time [s]: {}".format(time.time() - start))
            time.sleep(10)
    except Exception as e:
        print(e)
        sys.exit(1)

'''
['Bcands_Bu_tag_HLT_Mu7_IP4',
 'Bcands_Bd_tag_HLT_Mu7_IP4',
 'Bcands_Bs_tag_HLT_Mu7_IP4',
 'Bcands_Bu_tag_HLT_Mu9_IP5',
 'Bcands_Bd_tag_HLT_Mu9_IP5',
 'Bcands_Bs_tag_HLT_Mu9_IP5',
 'Bcands_Bu_tag_HLT_Mu9_IP6',
 'Bcands_Bd_tag_HLT_Mu9_IP6',
 'Bcands_Bs_tag_HLT_Mu9_IP6',
 'Bcands_Bu_tag_HLT_Mu12_IP6',
 'Bcands_Bd_tag_HLT_Mu12_IP6',
 'Bcands_Bs_tag_HLT_Mu12_IP6',
 'Bcands_Bu_tag_HLT_Mu9_IP5_only',
 'Bcands_Bd_tag_HLT_Mu9_IP5_only',
 'Bcands_Bs_tag_HLT_Mu9_IP5_only',
 'Bcands_Bu_tag_HLT_Mu9_IP6_only',
 'Bcands_Bd_tag_HLT_Mu9_IP6_only',
 'Bcands_Bs_tag_HLT_Mu9_IP6_only',
 'Bcands_Bu_tag_HLT_Mu12_IP6_only',
 'Bcands_Bd_tag_HLT_Mu12_IP6_only',
 'Bcands_Bs_tag_HLT_Mu12_IP6_only',
 'Bcands_Bu_probe_HLT_Mu7_IP4',
 'Bcands_Bd_probe_HLT_Mu7_IP4',
 'Bcands_Bs_probe_HLT_Mu7_IP4',
 'Bcands_Bu_probe_HLT_Mu9_IP5',
 'Bcands_Bd_probe_HLT_Mu9_IP5',
 'Bcands_Bs_probe_HLT_Mu9_IP5',
 'Bcands_Bu_probe_HLT_Mu9_IP6',
 'Bcands_Bd_probe_HLT_Mu9_IP6',
 'Bcands_Bs_probe_HLT_Mu9_IP6',
 'Bcands_Bu_probe_HLT_Mu12_IP6',
 'Bcands_Bd_probe_HLT_Mu12_IP6',
 'Bcands_Bs_probe_HLT_Mu12_IP6',
 'Bcands_Bu_probe_HLT_Mu9_IP5_only',
 'Bcands_Bd_probe_HLT_Mu9_IP5_only',
 'Bcands_Bs_probe_HLT_Mu9_IP5_only',
 'Bcands_Bu_probe_HLT_Mu9_IP6_only',
 'Bcands_Bd_probe_HLT_Mu9_IP6_only',
 'Bcands_Bs_probe_HLT_Mu9_IP6_only',
 'Bcands_Bu_probe_HLT_Mu12_IP6_only',
 'Bcands_Bd_probe_HLT_Mu12_IP6_only',
 'Bcands_Bs_probe_HLT_Mu12_IP6_only',
'''