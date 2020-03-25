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
hists = {}
subjob_cutflows = {}
for i, input_file in enumerate(input_files):
    print(input_file)
    this_hists = util.load(input_file)

    # Integrate dataset to save space
    for key in this_hists.keys():
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
        elif "cutflow" in item_name:
            pass
            if item_name in subjob_cutflows:
                subjob_cutflows[item_name].add(item)
            else:
                subjob_cutflows[item_name] = item
#pprint(hists)
#pprint(subjob_cutflows)

#print(subjob_cutflows.keys())
#print(subjob_cutflows["reco_Bu_cutflow"].keys())
cutflows = {}
for k1, d1 in subjob_cutflows.items():
    cutflows[k1] = None
    for k2, v in d1.items():
        if not cutflows[k1]:
            cutflows[k1] = v
        else:
            cutflows[k1].add(v)
pprint(cutflows)


y_slice = slice(-2.4, 2.4)
pt_slice = slice(3., 100.)

for channel in ["BuToKMuMu", "BsToKKMuMu", "BdToKPiMuMu"]:
    for side in ["tag", "probe"]:
        # Mass plot with phase space cuts
        fig_mass, ax_mass = plt.subplots()
        h1 = hists[f"{channel}_fit_pt_y_mass"]\
            .integrate("fit_y", y_slice)\
            .integrate("fit_pt", pt_slice)
        sels = [f"{side}_{x}" for x in ["inclusive", "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]]
        hist.plot1d(h1[(sels),:], ax=ax_mass, overlay="selection")
        if "Bu" in channel:
            ax_mass.set_xlim(5.05-0.1, 5.5+0.1)
            ax_mass.set_xlabel(r"$m^{(fit)}(B_u)$")
        elif "Bd" in channel:
            ax_mass.set_xlim(5.05-0.1, 5.5+0.1)
            ax_mass.set_xlabel(r"$m^{(fit)}(B_d)$")
        elif "Bs" in channel:
            ax_mass.set_xlim(5.2-0.1, 5.55+0.1)
            ax_mass.set_xlabel(r"$m^{(fit)}(B_s)$")
        plt.tight_layout()
        fig_mass.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_mass_SR.png")
        plt.close(fig_mass)

        fig_pt, ax_pt = plt.subplots()
        if channel == "BuToKMuMu":
            m0 = BU_MASS
        elif channel == "BdToKPiMuMu":
            m0 = BD_MASS
        elif channel == "BsToKKMuMu":
            m0 = BS_MASS
        left_edge = 1.e20
        right_edge = -1.e20
        for xbin in hists[f"{channel}_fit_pt_y_mass"].axis("fit_mass").identifiers():
            if xbin.lo <= m0 - 0.2 and m0 - 0.2 < xbin.hi:
                left_edge = xbin.lo
            elif xbin.lo <= m0 + 0.2 and m0 + 0.2 < xbin.hi:
                right_edge = xbin.hi
        mass_slice = hist.Interval(left_edge, right_edge)
        #mass_slice = slice(left_edge, right_edge)
        hpt = hists[f"{channel}_fit_pt_y_mass"]\
            .integrate("fit_y", y_slice)\
            .integrate("fit_mass", mass_slice)
        hist.plot1d(hpt[(sels),:], ax=ax_pt, overlay="selection")
        if "Bu" in channel:
            ax_pt.set_xlabel(r"$p_{T}^{(fit)}(B_u)$")
        elif "Bd" in channel:
            ax_pt.set_xlabel(r"$p_{T}^{(fit)}(B_d)$")
        elif "Bs" in channel:
            ax_pt.set_xlabel(r"$p_{T}^{(fit)}(B_s)$")
        ax_pt.set_yscale("log")
        ymin = 1.
        ymax = -1.
        for dataset, values_arr in hpt[(sels),:].values().items():
            if np.max(values_arr) > ymax:
                ymax = np.max(values_arr)
        ax_pt.set_ylim(1., ymax * 10.)
        plt.tight_layout()
        fig_pt.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_pt_SR.png")
        plt.close(fig_pt)

        fig_y, ax_y = plt.subplots()
        hy = hists[f"{channel}_fit_pt_y_mass"]\
            .integrate("fit_mass", mass_slice)\
            .integrate("fit_pt", pt_slice)
        hist.plot1d(hy[(sels),:], ax=ax_y, overlay="selection")
        if "Bu" in channel:
            ax_y.set_xlabel(r"$y^{(fit)}(B_u)$")
        elif "Bd" in channel:
            ax_y.set_xlabel(r"$y^{(fit)}(B_d)$")
        elif "Bs" in channel:
            ax_y.set_xlabel(r"$y^{(fit)}(B_s)$")
        ax_y.set_yscale("log")
        ymin = 1.
        ymax = -1.
        for dataset, values_arr in hy[(sels),:].values().items():
            if np.max(values_arr) > ymax:
                ymax = np.max(values_arr)
        ax_y.set_ylim(1., ymax * 10.)
        plt.tight_layout()
        fig_y.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_y_SR.png")
        plt.close(fig_y)

        # 2D plots
        fig_pt_mass, ax_pt_mass = plt.subplots()
        h_pt_mass = hists[f"{channel}_fit_pt_y_mass"].integrate("fit_y", y_slice).integrate("selection", (f"{side}_HLT_Mu9_IP6"))
        hist.plot2d(h_pt_mass, xaxis="fit_mass")
        fig_pt_mass.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_pt_mass_SR.png")
        plt.close(fig_pt_mass)

        fig_pt_y, ax_pt_y = plt.subplots()
        h_pt_y = hists[f"{channel}_fit_pt_y_mass"].integrate("fit_mass").integrate("selection", (f"{side}_HLT_Mu9_IP6"))
        hist.plot2d(h_pt_y, xaxis="fit_y")
        fig_pt_y.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_pt_y_SR.png")
        plt.close(fig_pt_y)

        # pT subslices
        for pt_subslice in [slice(3., 10.), slice(10., 15.), slice(15., 20.), slice(20., 100.)]:
            # Mass plot with phase space cuts
            fig_mass, ax_mass = plt.subplots()
            h1 = hists[f"{channel}_fit_pt_y_mass"]\
                .integrate("fit_y", y_slice)\
                .integrate("fit_pt", pt_subslice)
            sels = [f"{side}_{x}" for x in ["inclusive", "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]]
            hist.plot1d(h1[(sels),:], ax=ax_mass, overlay="selection")
            if "Bu" in channel:
                ax_mass.set_xlim(5.05, 5.5)
                ax_mass.set_xlabel(r"$m^{(fit)}(B_u)$")
            elif "Bd" in channel:
                ax_mass.set_xlim(5.05, 5.5)
                ax_mass.set_xlabel(r"$m^{(fit)}(B_d)$")
            elif "Bs" in channel:
                ax_mass.set_xlim(5.2, 5.55)
                ax_mass.set_xlabel(r"$m^{(fit)}(B_s)$")
            plt.tight_layout()
            fig_mass.savefig(f"{figure_directory}/kinematic/data/{channel}_{side}_mass_ptslice{pt_subslice.start}_{pt_subslice.stop}.png")
            plt.close(fig_mass)

# Bulk kinematic plots
coarse_pt_axis = hist.Bin("pt", r"$p_{T}}$ [GeV]", 50, 0., 50.)
for ptplot in ["BuToKMuMu_fit_pt", "BdToKPiMuMu_fit_pt", "BsToKKMuMu_fit_pt"]:
    hists[ptplot] = hists[ptplot].rebin("fit_pt", coarse_pt_axis)

kinplots = [
    'BuToKMuMu_chi2',
    'BuToKMuMu_fit_cos2D',
    'BuToKMuMu_fit_eta',
    'BuToKMuMu_fit_mass',
    'BuToKMuMu_fit_phi',
    'BuToKMuMu_fit_pt',
    'BuToKMuMu_jpsi_m',
    'BuToKMuMu_l_xy',
    'BuToKMuMu_l_xy_sig',
    'BdToKPiMuMu_chi2',
    'BdToKPiMuMu_fit_barmass',
    'BdToKPiMuMu_fit_best_barmass',
    'BdToKPiMuMu_fit_best_mass',
    'BdToKPiMuMu_fit_cos2D',
    'BdToKPiMuMu_fit_eta',
    'BdToKPiMuMu_fit_mass',
    'BdToKPiMuMu_fit_phi',
    'BdToKPiMuMu_fit_pt',
    'BdToKPiMuMu_jpsi_m',
    'BdToKPiMuMu_kstar_m',
    'BdToKPiMuMu_kstar_m_bar',
    'BdToKPiMuMu_kstar_m_best',
    'BdToKPiMuMu_l_xy',
    'BdToKPiMuMu_l_xy_sig',
    'BsToKKMuMu_chi2',
    'BsToKKMuMu_fit_cos2D',
    'BsToKKMuMu_fit_eta',
    'BsToKKMuMu_fit_mass',
    'BsToKKMuMu_fit_phi',
    'BsToKKMuMu_fit_pt',
    'BsToKKMuMu_jpsi_m',
    'BsToKKMuMu_l_xy',
    'BsToKKMuMu_l_xy_sig',
    'BsToKKMuMu_phi_m',
]
kinfigs = {}
kinaxs = {}
for kinplot in kinplots:
    for side in ["tag", "probe"]:
        sels = [f"{side}_{x}" for x in ["inclusive", "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]]
        kinfigs[kinplot], kinaxs[kinplot] = plt.subplots()
        hist.plot1d(hists[kinplot][(sels),:], overlay="selection", ax=kinaxs[kinplot])
        if "pt" in kinplot:
            kinaxs[kinplot].set_yscale("log")
            kinaxs[kinplot].set_ylim(1., 1.e6)
        if "phi_m" in kinplot:
            kinaxs[kinplot].set_xlim(0.9, 1.1)
        kinfigs[kinplot].savefig(f"{figure_directory}/kinematic/data/{kinplot}_{side}.png")

for channel, cutflow in cutflows.items():
    print(channel)
    keys = sorted(cutflow.keys(), key=lambda x: cutflow[x], reverse=True)
    for key in keys:
        print(f"{key} => {cutflow[key]}")

# N-1 plots
for var in ["phi_m", "jpsi_m", "l_xy_sig", "sv_prob", "cos2D"]:
'NM1_BsToKKMuMu_phi_m', 'NM1_BsToKKMuMu_jpsi_m', 'NM1_BsToKKMuMu_l_xy_sig', 'NM1_BsToKKMuMu_sv_prob', 'NM1_BsToKKMuMu_cos2D'