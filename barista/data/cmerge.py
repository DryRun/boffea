'''
Postprocessing for data condor jobs.
- Merge the subjob coffea files. Result: 1 coffea file per run-period-part. 
- Convert coffea to ROOT files, for RooFit.
@arg dir = folder containing subjobs, e.g. /home/dyu7/BFrag/data/histograms/condor/job20200609_115835
            (Contains folders Run2018*_part*)
'''
import os
import sys
from coffea import util, hist
import glob
import re
from pprint import pprint
import uproot
from collections import defaultdict
from functools import partial
import numpy as np
import time

import re
re_subjob = re.compile("(?P<runp>Run2018[A-D])")
def cmerge(output_file, input_files, force=False):
    print("cmerge(output_file={}, input_files={}".format(output_file, input_files))
    if os.path.isfile(output_file) and not force:
        raise ValueError("Output file {} already exists. Use option force to overwrite.".format(output_file))
    output = None
    for input_file in input_files:
        print(input_file)
        this_content = util.load(input_file)
        # Merge datasets to save space
        keys = list(this_content.keys())
        for key in keys:
            if "Bcands" in key or "cutflow" in key:
                continue
            if type(this_content[key]).__name__ == "Hist":
                if "dataset" in [x.name for x in this_content[key].axes()]:
                    subjobs = this_content[key].axis("dataset").identifiers()
                    mapping = {}
                    for subjob in subjobs:
                        runp = re_subjob.search(subjob.name).group()
                        if not runp in mapping:
                            mapping[runp] = []
                        mapping[runp].append(subjob.name)
                    this_content[key] = this_content[key].group(
                        "dataset", 
                        hist.Cat("dataset", "Primary dataset"), 
                        mapping)

        if not output:
            output = this_content
        else:
            output.add(this_content)
    print(f"Saving output to {output_file}")
    util.save(output, output_file)

def coffea2roofit(input_files, input_objs, output_file, output_objs=None, combine_datasets=True, force=False):
    '''
    Convert {dataset:Bcand_accumulator} output to TTree.
    Bcand_accumulator has dict-like structure {branch: array}
    input_obj = key or keys needed to find Bcand object
    datasets are added together.
    '''
    print("Welcome to coffea2roofit")
    print("Input files:")
    print(input_files)
    print("Output file:")
    print(output_file)

    if os.path.isfile(output_file) and not force:
        raise ValueError("[coffea2roofit] Output file {} already exists. Use option force to overwrite.".format(output_file))

    with uproot.recreate(output_file) as output_filehandle:
        # Make list of datasets and branches
        branches = {}
        branch_types = {}
        input_stuff = util.load(input_files[0])
        for input_obj in input_objs:
            branches[input_obj] = []
            branch_types[input_obj] = {}
            for k1 in input_stuff[input_obj].keys():
                for branch_name in input_stuff[input_obj][k1].keys():
                    branches[input_obj].append(branch_name)
                    branch_types[input_obj][branch_name] = input_stuff[input_obj][k1][branch_name].value.dtype
                break

        if not output_objs:
            output_objs = {}
            for input_obj in input_objs:
                output_objs[input_obj] = input_obj

        bad_input_files = []
        nevents = 0
        for input_file in input_files:
            #print("Processing {}".format(input_file))
            if os.path.getsize(input_file) < 1000:
                print(f"WARNING: Input file {input_file} looks corrupt. Skipping.")
                bad_input_files.append(input_file)
                continue
            input_stuff = util.load(input_file)
            #print(input_stuff["nevents"])
            for key, value in input_stuff["nevents"].items():
                nevents += value
            for i, input_obj in enumerate(input_objs):
                for k1 in input_stuff[input_obj].keys():
                    # Determine which tree to fill, and create it if it doesn't exist
                    if combine_datasets:
                        tree_name = output_objs[input_obj]
                    else:
                        tree_name = f"{output_objs[input_obj]}_{k1}"
                    if not tree_name in output_filehandle:
                        #print("Creating tree {}".format(tree_name))
                        #print(branch_types[input_obj])
                        output_filehandle[tree_name] = uproot.newtree(branch_types[input_obj])

                    # Make {branch : array} dict for filling
                    bcand_accumulator = input_stuff[input_obj][k1]
                    bcand_array = {}
                    for branch in branches[input_obj]:
                        bcand_array[branch] = bcand_accumulator[branch].value

                    # Fill
                    #print(bcand_array)
                    output_filehandle[tree_name].extend(bcand_array)
                # End loop over input branches in Bcand array
            # End loop over Bcand arrays
        
        # Write nevents to file
        output_filehandle["nevents"] = np.histogram([0], bins=[-0.5, 0.5], weights=[nevents])

        if len(bad_input_files) >= 1:
            print("Some input files were skipped due to small size:")
            print(bad_input_files)

def _cancel(job):
    try:
        # this is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add coffea subjobs back together")
    parser.add_argument("-d", "--dir", type=str, help="Folder containing subjobs")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-r", "--rootonly", action="store_true", help="Skip coffea part")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError("Directory {} does not exist".format(args.dir))

    runpart_dirs = sorted(glob.glob(f"{args.dir}/Run2018*_part*/"))
    runparts = [x.split("/")[-2] for x in runpart_dirs]

    if not args.rootonly:
        print("Merging coffea subjobs...")
        cmerge_args = []
        for runpart in runparts:
            #if not os.path.isdir(f"{args.dir}/{runpart}/add_eff"):
            #    raise ValueError(f"{args.dir}/{runpart}/add_eff does not exist. Did you run add_efficiency.py first?")
            output_file = f"{args.dir}/{runpart}.coffea"
            input_files = glob.glob(f"{args.dir}/{runpart}/DataHistograms_Run*_part*subjob*.coffea")
            cmerge_args.append([output_file, input_files, args.force])

        for cmerge_arg in cmerge_args:
            cmerge(*cmerge_arg)
        '''
        import concurrent.futures
        try:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
            futures_set = set(executor.submit(cmerge, *cmerge_arg) for cmerge_arg in cmerge_args)
            njobs = len(futures_set)
            start = time.time()
            nfinished = 0
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                nfinished += len(finished)
                for fjob in finished:
                    fjob.result()
                print("Finished: {} / {}".format(nfinished, njobs))
                print("Outstanding: {} / {}".format(len(futures_set), njobs))
                print("Time [s]: {}".format(time.time() - start))
                time.sleep(10)
        except Exception:
            for job in futures_set:
                _cancel(job)
            raise
        '''

        print("...done merging coffea subjobs.")

    print("Converting coffea files to ROOT TTrees for RooFit")
    # Determine which objects to convert (any coffea object with "Bcands" in name)
    input_objs = []
    stuff = util.load(f"{args.dir}/{runparts[0]}.coffea")
    for key in stuff.keys():
        if "Bcands" in key:
            input_objs.append(key)
    print("List of objects to convert to TTrees:")
    print(input_objs)
    coffea2roofit_kwargs = []
    for runpart in runparts:
        input_coffea_file = f"{args.dir}/{runpart}.coffea"
        output_root_file = f"{args.dir}/{runpart}.root"
        coffea2roofit_kwargs.append({
            "input_files": [input_coffea_file], 
            "input_objs": input_objs, 
            "output_file": output_root_file, 
            "combine_datasets": True,
            "force": args.force})
    for coffea2roofit_kwarg in coffea2roofit_kwargs:
        coffea2roofit(**coffea2roofit_kwarg)
    #try:
    #    executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
    #    futures_set = set(executor.submit(coffea2roofit, **coffea2roofit_kwarg) for coffea2roofit_kwarg in coffea2roofit_kwargs)
    #    njobs = len(futures_set)
    #    start = time.time()
    #    nfinished = 0
    #    while len(futures_set) > 0:
    #        finished = set(job for job in futures_set if job.done())
    #        futures_set.difference_update(finished)
    #        nfinished += len(finished)
    #        for fjob in finished:
    #            fjob.result()
    #        print("Finished: {} / {}".format(nfinished, njobs))
    #        print("Outstanding: {} / {}".format(len(futures_set), njobs))
    #        print("Time [s]: {}".format(time.time() - start))
    #        time.sleep(10)
    #except KeyboardInterrupt:
    #    for job in futures_set:
    #        _cancel(job)
    #    if status:
    #        print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.", file=sys.stderr)
    #        print("Running jobs:", sum(1 for j in futures_set if j.running()), file=sys.stderr)
    #except Exception:
    #    for job in futures_set:
    #        _cancel(job)
    #    raise
