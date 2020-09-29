import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from pprint import pprint
import glob

import mplhep
plt.style.use(mplhep.style.ROOT)
plt.tight_layout()
from brazil.aguapreta import *

figure_directory = "/home/dryu/BFrag/data/kinematic/"

input_files = {
    "data": glob.glob("/home/dryu/BFrag/data/histograms/Run*coffea"),
    "Bu": ["/home/dryu/BFrag/boffea/barista/Bu2KJpsi2KMuMu/MCEfficiencyHistograms.coffea"],
    "Bd": ["/home/dryu/BFrag/boffea/barista/Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms.coffea"],
    "Bs": ["/home/dryu/BFrag/boffea/barista/Bs2PhiJpsi2KKMuMu/MCEfficiencyHistograms.coffea"],
}
coffea = {}
for what in input_files.keys():
    for f in input_files[what]:
        coffea_tmp = util.load(f)
        
        # Delete Bcands trees
        for key in list(coffea_tmp.keys()):
            if "Bcands" in key or "cutflow" in key:
                del coffea_tmp[key]

        # For data, combine subjobs
        #if what == "data":
        #    subjobs = [x.name for x in coffea_tmp["BuToKMuMu_fit_pt_absy_mass"].axis("dataset").identifiers()]
        #    print(subjobs)
        #    for key in list(coffea_tmp.keys()):
        #        if type(coffea_tmp[key]).__name__ == "Hist":
        #            if "dataset" in [x.name for x in coffea_tmp[key].axes()]:
        #                print("DEBUG : Attempting to group axes.")
        #                print("DEBUG : Input identifiers = ")
        #                print(coffea_tmp[key].axis("dataset").identifiers())
        #                print("DEBUG : attempt to group")
        #                print(subjobs)
        #                coffea_tmp[key] = coffea_tmp[key].group("dataset", 
        #                    hist.Cat("dataset", "Primary dataset"), 
        #                    {"Run2018": subjobs})

        # Persistify
        if not what in coffea:
            coffea[what] = coffea_tmp
        else:
            coffea[what].add(coffea_tmp)

print(coffea["Bu"]["BuToKMuMu_fit_pt_absy_mass"].axes())
print(coffea["Bu"]["BuToKMuMu_fit_pt_absy_mass"].axis("dataset").identifiers())
plot_index = {
    "Bu": {
        "fit_pt":{
            "hist_mc": coffea["Bu"]["BuToKMuMu_fit_pt_absy_mass"]\
                .group("dataset", hist.Cat("dataset", "Primary dataset"), {"mc": ["Bu2KJpsi2KMuMu_probefilter"]})\
                .integrate("fit_mass")\
                .integrate("fit_absy", slice(0., 2.25))\
                .rebin("fit_pt", hist.Bin("pt", r"$p_{T}$ [GeV]", 50, 0., 50.)),
            "hist_data": coffea["data"]["BuToKMuMu_fit_pt_absy_mass"]\
                .integrate("fit_mass")\
                .integrate("fit_absy", slice(0., 2.25))\
                .rebin("fit_pt", hist.Bin("pt", r"$p_{T}$ [GeV]", 50, 0., 50.)),
            "xlim": [0., 50.], 
            "xscale": "linear", 
            "xlabel": r"$p_{T}$ [GeV]",
            "ylim": "auto", 
            "yscale": "log", 
            "ylabel": "Events",
        },
        "fit_absy":{
            "hist_mc": coffea["Bu"]["BuToKMuMu_fit_pt_absy_mass"]\
                .group("dataset", hist.Cat("dataset", "Primary dataset"), {"mc": ["Bu2KJpsi2KMuMu_probefilter"]})\
                .integrate("fit_mass")\
                .integrate("fit_pt", slice(0., 30.))\
                .rebin("fit_absy", hist.Bin("absy", r"|y|$", 10, 0., 2.5)),
            "hist_data": coffea["data"]["BuToKMuMu_fit_pt_absy_mass"]\
                .integrate("fit_mass")\
                .integrate("fit_pt", slice(0., 30.))\
                .rebin("fit_absy", hist.Bin("absy", r"|y|$", 10, 0., 2.5)),
            "xlim": [0., 3.0], 
            "xscale": "linear", 
            "xlabel": r"$|y|$",
            "ylim": "auto", 
            "yscale": "log", 
            "ylabel": "Events",
        },
        "fit_mass":{
            "hist_mc": coffea["Bu"]["BuToKMuMu_fit_pt_absy_mass"]\
                .group("dataset", hist.Cat("dataset", "Primary dataset"), {"mc": ["Bu2KJpsi2KMuMu_probefilter"]})\
                .integrate("fit_absy", slice(0., 2.25))\
                .integrate("fit_pt", slice(0., 30.)),
            "hist_data": coffea["data"]["BuToKMuMu_fit_pt_absy_mass"]\
                .integrate("fit_absy", slice(0., 2.25))\
                .integrate("fit_pt", slice(0., 30.)),
            "xlim": [5.05, 5.5], 
            "xscale": "linear", 
            "xlabel": r"Fitted $B_{u}$ mass [GeV]",
            "ylim": "auto", 
            "yscale": "log", 
            "ylabel": "Events",
        }
    }
}

figure_directory = "/home/dryu/BFrag/data/kinematic"
def plot(hist_mc=None, hist_data=None, xlim=[], xscale="", xlabel="", ylim=[], yscale="", ylabel="", data_selection="", mc_selection="", savetag=""):
    hist_mc = hist_mc.integrate("selection", mc_selection)
    print(hist_data.axis("selection").identifiers())
    hist_data = hist_data.integrate("selection", data_selection)

    # Normalize MC to data
    print(hist_data)
    print(hist_data.values())
    data_norm = hist_data.values().sum()

    hist_all = copy.deepcopy(hist_data).add(hist_mc)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    hist.plot1d(hist_all, overlay="dataset", ax=ax[0])
    ax[0].set_xlim(xlim)
    ax[0].set_xscale(xscale)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylim(ylim)
    ax[0].set_yscale(yscale)
    ax[0].set_ylabel(ylabel)

    hist.plotratio(
        num=hist_all.integrate("dataset", "Run2018"), 
        den=hist_all.integrate("dataset", "Bu2KJpsi2KMuMu_probefilter"), 
        unc='num',
        ax=ax[1])
    ax[1].set_xlim(xlim)
    ax[1].set_xscale(xscale)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("Data / MC")

    fig.savefig(f"{figure_directory}/{savetag}.png")

if __name__ == "__main__":
    mc_selection = "recomatch_HLT_Mu9_IP5"
    data_selection = "recotrig_HLT_Mu9_IP5"
    for btype in ["Bu"]:
        for plot_name, metadata in plot_index[btype].items():
            plot(**metadata, savetag=f"{plot_name}_reco", mc_selection=mc_selection, data_selection=data_selection)