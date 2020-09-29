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
    "/home/dryu/BFrag/data/histograms/condor/job20200513_215445/DataHistograms_Run2018.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018A.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018B.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018C.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D1.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D2.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D3.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D4.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistograms_Run2018D5.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018A.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018B.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018C.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsTT_Run2018D.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part1.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part2.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part3.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part4.coffea",
#"/home/dryu/BFrag/data/histograms/DataHistogramsT_Run2018D_part5.coffea",
]
hists = {}
subjob_cutflows = {}
for i, input_file in enumerate(input_files):
    print(input_file)
    this_hists = util.load(input_file)

    # Integrate dataset to save space
    for key in this_hists.keys():
        if not "NM1" in key:
            continue
        obj = this_hists[key]
        if isinstance(obj, hist.hist_tools.Hist):
            if "dataset" in obj.axes():
                this_hists[key] = obj.integrate("dataset")
                del obj
    #if i == 0:
    #    pprint(this_hists.keys())
    for item_name, item in this_hists.items():
        if isinstance(item, hist.Hist):
            if item_name in hists:
                hists[item_name].add(item)
            else:
                hists[item_name] = item

y_slice = slice(-2.4, 2.4)
pt_slice = slice(3., 100.)

print([x for x in hists.keys() if "NM1" in x])
print(hists["NM1_BsToKKMuMu_phi_m"])
print(hists["NM1_BsToKKMuMu_phi_m"].axis("selection").identifiers())
fig, ax = plt.subplots()
hist.plot1d(hists["NM1_BsToKKMuMu_phi_m"], overlay="selection", ax=ax)
ax.set_xlabel(r"$m_{\phi\rightarrow K^{+}K^{-}}$ [GeV]")
plt.tight_layout()
fig.savefig(f"{figure_directory}/kinematic/data/NM1_phi_m.png")
