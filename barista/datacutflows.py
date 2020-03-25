import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from pprint import pprint

import mplhep
plt.style.use(mplhep.style.ROOT)
plt.tight_layout()
from brazil.aguapreta import *

figure_directory = "/home/dryu/BFrag/data/figures/"

input_files = [
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018A.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018B.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018C.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D1.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D2.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D3.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D4.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D5.coffea",
"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018A.coffea",
"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018B.coffea",
"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018C.coffea",
"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018D.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part1.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part2.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part3.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part4.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part5.coffea",
]
subjob_cutflows = {}
for i, input_file in enumerate(input_files):
    print(input_file)
    this_hists = util.load(input_file)

    for item_name, item in this_hists.items():
        if  "cutflow" in item_name:
            pass
            if item_name in subjob_cutflows:
                subjob_cutflows[item_name].add(item)
            else:
                subjob_cutflows[item_name] = item
        else:
            continue

cutflows = {}
for k1, d1 in subjob_cutflows.items():
    cutflows[k1] = None
    for k2, v in d1.items():
        if not cutflows[k1]:
            cutflows[k1] = v
        else:
            cutflows[k1].add(v)
pprint(cutflows)


for channel, cutflow in cutflows.items():
    print("\n" + channel)
    print("\\begin{table}")
    print("\t\\begin{tabular}{|l|r|c|}")
    print("\t\t\\hline")
    print("\t\tCut & Events & Fraction \\\\")
    print("\t\t\\hline")
    keys = sorted(cutflow.keys(), key=lambda x: cutflow[x], reverse=True)
    for key in keys:
        key_formatted = key.replace("_", "\\_")
        eff = cutflow[key] / cutflow['inclusive']
        efficiency_str = '%s' % float('%.3g' % eff)
        print(f"\t\t{key_formatted} \t&\t {cutflow[key]} \t&\t {efficiency_str} \\\\")
    print("\t\t\\hline")
    print("\t\\end{tabular}")
    print("\\end{table}")
