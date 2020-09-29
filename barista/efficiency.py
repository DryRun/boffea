import os
import sys
import numpy as np 
from coffea import util, hist
import matplotlib.pyplot as plt
import math
from pprint import pprint
import pickle
import copy

import mplhep
plt.style.use(mplhep.style.LHCb)
plt.tight_layout()

figure_directory = "/home/dryu/BFrag/data/efficiency/figures"

triggers = ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]
btypes = ["Bu", "Bs", "Bd"]
btype_longnames = {
	"Bu": "Bu2KJpsi2KMuMu",
	"Bs": "Bs2PhiJpsi2KKMuMu",
	"Bd": "Bd2KsJpsi2KPiMuMu"
}
btype_shortnames = {
	"Bu": "BuToKMuMu", 
	"Bs": "BsToKKMuMu", 
	"Bd": "BdToKPiMuMu"
}

# MC probefilter efficiency
coffea_files = {}
for btype in btypes:
	coffea_files[btype] = util.load(f"{btype_longnames[btype]}/MCEfficiencyHistograms.coffea")

pt_axes = {
	"probe": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([5., 10., 15., 20., 25., 30.])), 
	"tag": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0]))
}

bigh_reco_pt = {}
bigh_truth_pt = {}
for btype in btypes:
	bigh_reco_pt[btype] = {}
	bigh_truth_pt[btype] = {}
	for side in ["probe", "tag"]:
		bigh_reco_pt[btype][side] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt"].rebin("fit_pt", pt_axes[side])
		bigh_truth_pt[btype][side] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt"].rebin("pt", pt_axes[side])

# Probefilter efficency, needed for probe side
eff_probefilter = {}
deff_probefilter = {}
for btype in btypes:
	bigh_probefilter   = bigh_truth_pt[btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	eff_probefilter[btype] = probefilter_counts / inclusive_counts
	deff_probefilter[btype] = np.sqrt(eff_probefilter[btype] * (1. - eff_probefilter[btype]) / inclusive_counts)

avg_eff_probefilter = None 
sum_inclusive_counts = None
first = True
for btype in btypes:
	bigh_probefilter   = bigh_truth_pt[btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	if first:
		first = False
		avg_eff_probefilter = copy.deepcopy(probefilter_counts)
		sum_inclusive_counts = copy.deepcopy(inclusive_counts)
	else:
		avg_eff_probefilter = avg_eff_probefilter + probefilter_counts
		sum_inclusive_counts = sum_inclusive_counts + inclusive_counts
avg_eff_probefilter = avg_eff_probefilter / sum_inclusive_counts
davg_eff_probefilter = np.sqrt(avg_eff_probefilter * (1. - avg_eff_probefilter) / sum_inclusive_counts)


# Single-trigger efficiency
eff = {}
deff = {}

# - For Bd, include both matched and matched-swap
for btype in btypes:
	eff[btype] = {}
	deff[btype] = {}
	for side in ["probe", "tag"]:
		eff[btype][side] = {}
		deff[btype][side] = {}

		for trigger in triggers:
			if side == "probe":
				h_truth_pt = bigh_truth_pt[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			elif side == "tag":
				h_truth_pt = bigh_truth_pt[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			if side == "probe":
				h_reco_pt = bigh_reco_pt[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			elif side == "tag":
				h_reco_pt = bigh_reco_pt[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])

			use_truth_side = True
			if use_truth_side:
				if btype == "Bd":
					h_truth_pt_matched = h_truth_pt.integrate("selection", ([f"matched_{side}_{trigger}", f"matched_swap_{side}_{trigger}"]))
				else:
					h_truth_pt_matched = h_truth_pt.integrate("selection", ([f"matched_{side}_{trigger}"]))
				h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))
				counts_matched = h_truth_pt_matched.values(sumw2=False)[()]
				counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]
			else:
				if btype == "Bd":
					h_reco_pt_matched = h_reco_pt.integrate("selection", ([f"{side}match_{trigger}", f"{side}matchswap_{side}_{trigger}"]))
				else:
					h_reco_pt_matched = h_reco_pt.integrate("selection", ([f"{side}match_{trigger}"]))
				h_reco_pt_inclusive     = h_reco_pt.integrate("selection", (["inclusive"]))
				h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))
				counts_matched = h_reco_pt_matched.values(sumw2=False)[()]
				counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

			eff_recosel = counts_matched / counts_inclusive
			deff_recosel = np.sqrt(eff_recosel * (1. - eff_recosel) / counts_inclusive)

			#eff[btype][side][trigger] = eff_recosel
			#deff[btype][side][trigger] = deff_recosel

			if side == "tag":
				eff[btype][side][trigger] = eff_recosel
				deff[btype][side][trigger] = deff_recosel
			else:
				# Individual probefilters
				#eff[btype][side][trigger] = eff_recosel * eff_probefilter[btype]
				#deff[btype][side][trigger] = eff[btype][side][trigger] * np.sqrt(
				#	(deff_recosel / eff_recosel)**2 + (deff_probefilter[btype] / eff_probefilter[btype])**2
				#	)

				# Average probefilter
				eff[btype][side][trigger] = eff_recosel * avg_eff_probefilter
				deff[btype][side][trigger] = eff[btype][side][trigger] * np.sqrt(
					(deff_recosel / eff_recosel)**2 + (davg_eff_probefilter / avg_eff_probefilter)**2
					)

# Efficiencies for final trigger strategies
total_lumis = {
    "HLT_Mu12_IP6": 34698.771755487,
    "HLT_Mu9_IP6": 33577.176480857,
    "HLT_Mu9_IP5": 20890.692760413,
    "HLT_Mu7_IP4":6939.748065950
}
unique_lumis = {
    "HLT_Mu12_IP6": 34698.771755487 - 33577.176480857,
    "HLT_Mu9_IP6": 33577.176480857 - 20890.692760413,
    "HLT_Mu9_IP5": 20890.692760413 - 6939.748065950,
    "HLT_Mu7_IP4":6939.748065950
}
total_lumi = 34698.771755487
for btype in btypes:
	for side in ["tag", "probe"]:
		eff[btype][side]["HLT_all"] = 0.
		deff[btype][side]["HLT_all"] = 0.
		sumw = 0.
		for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
			weight = unique_lumis[trigger] / total_lumi
			eff[btype][side]["HLT_all"] += eff[btype][side][trigger] * weight
			deff[btype][side]["HLT_all"] += (deff[btype][side][trigger] * weight)**2
			sumw += weight
		eff[btype][side]["HLT_all"] = eff[btype][side]["HLT_all"] / sumw
		deff[btype][side]["HLT_all"] = np.sqrt(deff[btype][side]["HLT_all"]) / sumw

		eff[btype][side]["HLT_Mu9"] = 0.
		deff[btype][side]["HLT_Mu9"] = 0.
		sumw = 0.
		for trigger in ["HLT_Mu9_IP5", "HLT_Mu9_IP6"]:
			if trigger == "HLT_Mu9_IP5":
				weight = total_lumis["HLT_Mu9_IP5"]  / total_lumis["HLT_Mu9_IP6"]
			elif trigger == "HLT_Mu9_IP6":
				weight = (total_lumis["HLT_Mu9_IP6"] - total_lumis["HLT_Mu9_IP5"]) / total_lumis["HLT_Mu9_IP6"]
			else:
				raise ValueError("Error!")
			eff[btype][side]["HLT_Mu9"] += eff[btype][side][trigger] * weight
			deff[btype][side]["HLT_Mu9"] += (deff[btype][side][trigger] * weight)**2
			sumw += weight
		eff[btype][side]["HLT_Mu9"] = eff[btype][side]["HLT_Mu9"] / sumw
		deff[btype][side]["HLT_Mu9"] = np.sqrt(deff[btype][side]["HLT_Mu9"]) / sumw

		eff[btype][side]["HLT_Mu7"] = eff[btype][side]["HLT_Mu7_IP4"]
		deff[btype][side]["HLT_Mu7"] = deff[btype][side]["HLT_Mu7_IP4"]

# Turn into one dict
eff_deff = {}
for btype in eff.keys():
	eff_deff[btype] = {}
	for side in eff[btype].keys():
		eff_deff[btype][side] = {}
		for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"]:
			eff_deff[btype][side][trigger_strategy] = []
			for i in range(len(eff[btype][side][trigger_strategy])):
				eff_deff[btype][side][trigger_strategy].append((eff[btype][side][trigger_strategy][i], deff[btype][side][trigger_strategy][i]))

with open("/home/dryu/BFrag/data/efficiency/efficiency.pkl", "wb") as f:
	pickle.dump(eff_deff, f)


# Plots
for side in ["tag", "probe"]:
	bin_centers = bigh_truth_pt["Bs"][side].axis("pt").centers()
	xerrs = (bigh_truth_pt["Bs"][side].axis("pt").edges()[1:] - bigh_truth_pt["Bs"][side].axis("pt").edges()[:-1]) / 2
	colors = {
		"Bd": "red",
		"Bu": "blue", 
		"Bs": "green"
	}
	labels = {
		"Bd": r"$B^{0}$", 
		"Bu": r"$B^{\pm}$",
		"Bs": r"$B_{s}$"
	}
	for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"]:
		fig, ax = plt.subplots(1, 1, figsize=(10,7))
		for btype in btypes:
			#pprint(eff[btype][side][trigger_strategy])
			ax.errorbar(
				x=bin_centers, 
				y=eff[btype][side][trigger_strategy], 
				xerr=xerrs,
				yerr=deff[btype][side][trigger_strategy], 
				marker=".", 
				markersize=10.,
				color=colors[btype],
				label=labels[btype],
				ls="none",
				ecolor=colors[btype],
				elinewidth=1
				)
		ax.set_xlim(0., 45.)
		if side == "probe":
			ax.set_ylim(1.e-7, 5.e-3)
		elif side == "tag":
			ax.set_ylim(1.e-5, 0.5)
		ax.set_yscale("log")
		ax.set_xlabel(r"$p_{T}$ [GeV]")
		ax.set_ylabel("Total efficiency")
		ax.xaxis.set_ticks_position("both")
		ax.yaxis.set_ticks_position("both")
		ax.tick_params(direction="in")
		ax.legend()
		print(f"{figure_directory}/total_efficiency_{side}_{trigger_strategy}.png")
		fig.savefig(f"{figure_directory}/total_efficiency_{side}_{trigger_strategy}.png")
