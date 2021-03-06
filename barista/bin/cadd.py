#!/usr/bin/env python 
import os
import sys
from coffea import util, hist
import re
from copy import deepcopy
import psutil
process = psutil.Process(os.getpid())

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Add coffea subjobs back together")
  parser.add_argument("output_file", type=str, help="Output path")
  parser.add_argument("input_files", type=str, nargs="+", help="Input files")
  parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
  args = parser.parse_args()

  print(f"[cadd] Output file = {args.output_file}")

  output = None
  re_subjob = re.compile("(?P<subjob_tag>_subjob\d+)")
  for i, input_file in enumerate(args.input_files):
    print(f"[cadd] \tInput file {i}/{len(args.input_files)} => {input_file}")
    this_input = util.load(input_file)

    # Delete "Bcand" stuff
    del_keys = []
    for key in this_input.keys():
      if "Bcand" in key or "eta_vs_pt_vs_mass" in key or "pt_y_mass" in key:
        #print("Skipping {}".format(key))
        del_keys.append(key)
    for key in del_keys:
      del this_input[key]

    # Collapse cutflows, otherwise get one per subjob
    for key in this_input.keys():
      if "cutflow" in key:
        cutflow_copy = None
        for k, v in this_input[key].items():
          print(k)
          print(v)
          if not cutflow_copy:
            cutflow_copy = deepcopy(v)
          else:
            cutflow_copy.add(v)
        #print("DEBUG: cutflow_copy=")
        #print(cutflow_copy)
        if cutflow_copy:
          this_input[key] = cutflow_copy

    # Collapse subjobs: needed temporarily to reduce giant numbers of datasets, but now DataProcessor does a better job.
    #for key in this_input.keys():
    #  obj = this_input[key]
    #  if isinstance(obj, hist.hist_tools.Hist):
    #    if "dataset" in obj.axes():
    #      dataset_rebin = {}
    #      for dataset_bin in obj.axis("dataset").identifiers():
    #        match = re_subjob.search(dataset_bin.name)
    #        if match:
    #          newbin = dataset_bin.name.replace(match.group("subjob_tag"), "")
    #        else:
    #          raise ValueError(f"Couldn't parse bin {dataset_bin}")
    #        if not newbin in dataset_rebin:
    #          dataset_rebin[newbin] = []
    #        dataset_rebin[newbin].append(dataset_bin.name)
    #      # End loop over datasets
    #      this_input[key] = obj.group(obj.axis("dataset"), hist.Cat("dataset", "Primary dataset"), dataset_rebin)

    if not output:
      output = deepcopy(this_input)
    else:
      #assert set(output.keys()) == set(this_input.keys())
      #print(output)
      output.add(this_input)
    del this_input
    print("Done with this file. mem={}".format(print(process.memory_info().rss)))
  print("Trying to save now. mem={}".format(print(process.memory_info().rss)))
  util.save(output, args.output_file)
