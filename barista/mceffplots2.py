import os
import sys
import numpy as np 
from coffea import util, hist
import matplotlib.pyplot as plt
import math
from pprint import pprint

import mplhep
plt.style.use(mplhep.style.LHCb)
plt.tight_layout()

import argparse
parser = argparse.ArgumentParser(description="MC efficiency")
parser.add_argument("--fine", action="store_true", help="Fine binning (1 GeV) instead of 5 GeV")
args = parser.parse_args()

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

if args.fine:
	coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 50, 0., 50.)
	coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 50, 0., 50.)
	coarse_truth_pt_axis = hist.Bin("coarse_truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 50, 0., 50.)
	savetag = "_fine"
else:
	coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 10, 0., 50.)
	coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 10, 0., 50.)
	coarse_truth_pt_axis = hist.Bin("coarse_truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 10, 0., 50.)
	savetag = "_coarse"

bigh_reco_pt = {}
bigh_truth_pt = {}
for btype in btypes:
	bigh_reco_pt[btype] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
	bigh_truth_pt[btype] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt"].rebin("pt", coarse_truth_pt_axis)

eff = {}
deff = {}
# Reconstruction * selection efficiency, probe side
# - For Bd, include both matched and matched-swap
for btype in btypes:
	eff[btype] = {}
	deff[btype] = {}
	for trigger in triggers:
		print(bigh_truth_pt[btype].axis("dataset").identifiers())
		print(f"{btype_longnames[btype]}_probefilter")
		h_truth_pt = bigh_truth_pt[btype].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])

		if btype == "Bd":
			h_truth_pt_matched_probe = h_truth_pt.integrate("selection", ([f"matched_probe_{trigger}", f"matched_swap_probe_{trigger}"]))
		else:
			h_truth_pt_matched_probe = h_truth_pt.integrate("selection", ([f"matched_probe_{trigger}"]))
		h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

		print(h_truth_pt_matched_probe.values(sumw2=False))
		counts_matched_probe = h_truth_pt_matched_probe.values(sumw2=False)[()]
		counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

		eff[btype][f"recosel_probe_{trigger}"] = counts_matched_probe / counts_inclusive
		deff[btype][f"recosel_probe_{trigger}"] = np.sqrt(eff[btype][f"recosel_probe_{trigger}"] * (1. - eff[btype][f"recosel_probe_{trigger}"]) / counts_inclusive)

# MC probefilter efficiency
for btype in btypes:
	bigh_probefilter   = bigh_truth_pt[btype].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	eff[btype]["probefilter"] = probefilter_counts / inclusive_counts
	deff[btype]["probefilter"] = np.sqrt(eff[btype]['probefilter'] * (1. - eff[btype]['probefilter']) / inclusive_counts)

# MC probefilter closure test: probefilter efficiency on probefilter'ed samples
for btype in btypes:
	bigh_probefilter   = bigh_truth_pt[btype].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	eff[btype]["probefilter_closure"] = probefilter_counts / inclusive_counts
	deff[btype]["probefilter_closure"] = np.sqrt(eff[btype]['probefilter_closure'] * (1. - eff[btype]['probefilter_closure']) / inclusive_counts)

# Total probe efficiency
for btype in btypes:
	for trigger in triggers:
		eff[btype][f"probe_{trigger}"] = eff[btype][f"recosel_probe_{trigger}"] * eff[btype]["probefilter"]
		deff[btype][f"probe_{trigger}"] = eff[btype][f"probe_{trigger}"] * np.sqrt(
							(deff[btype][f"recosel_probe_{trigger}"] / eff[btype][f"recosel_probe_{trigger}"])**2 
							+ (deff[btype]["probefilter"] / eff[btype]["probefilter"])**2)


# Reconstruction * selection efficiency, tag side
for btype in btypes:
	for trigger in triggers:
		h_truth_pt = bigh_truth_pt[btype].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
		if btype == "Bd":
			h_truth_pt_matched_tag = h_truth_pt.integrate("selection", ([f"matched_tag_{trigger}", f"matched_swap_tag_{trigger}"]))
		else:
			h_truth_pt_matched_tag = h_truth_pt.integrate("selection", ([f"matched_tag_{trigger}"]))
		h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

		counts_matched_tag = h_truth_pt_matched_tag.values(sumw2=False)[()]
		counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

		eff[btype][f"tag_{trigger}"] = counts_matched_tag / counts_inclusive
		deff[btype][f"tag_{trigger}"] = np.sqrt(eff[btype][f"tag_{trigger}"] * (1. - eff[btype][f"tag_{trigger}"]) / counts_inclusive)

# Compute "inclusive" trigger efficiency
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
		eff[btype][f"{side}_inclusive"] = 0.
		deff[btype][f"{side}_inclusive"] = 0.
		sumw = 0.
		for trigger in triggers:
			weight = unique_lumis[trigger] / total_lumi
			eff[btype][f"{side}_inclusive"] += eff[btype][f"{side}_{trigger}"] * weight
			deff[btype][f"{side}_inclusive"] += (deff[btype][f"{side}_{trigger}"] * weight)**2
			sumw += weight
		eff[btype][f"{side}_inclusive"] = eff[btype][f"{side}_inclusive"] / sumw
		deff[btype][f"{side}_inclusive"] = np.sqrt(deff[btype][f"{side}_inclusive"]) / sumw

# Plots
bin_centers = bigh_truth_pt["Bs"].axis("coarse_truth_pt").centers()
xerrs = (bigh_truth_pt["Bs"].axis("coarse_truth_pt").edges()[1:] - bigh_truth_pt["Bs"].axis("coarse_truth_pt").edges()[:-1]) / 2
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
for trigger in triggers + ["inclusive"]:
	for side in ["tag", "probe"]:
		fig, ax = plt.subplots(1, 1, figsize=(10,7))
		for btype in btypes:
			pprint(eff[btype])
			ax.errorbar(
				x=bin_centers, 
				y=eff[btype][f"{side}_{trigger}"], 
				xerr=xerrs,
				yerr=deff[btype][f"{side}_{trigger}"], 
				marker=".", 
				markersize=10.,
				color=colors[btype],
				label=labels[btype],
				ls="none",
				ecolor=colors[btype],
				elinewidth=1
				)
		ax.set_xlim(0., 30.)
		if side == "probe":
			ax.set_ylim(1.e-7, 5.e-3)
		elif side == "tag":
			ax.set_ylim(1.e-5, 0.5)
		ax.set_yscale("log")
		ax.set_xlabel(r"$p_{T}$ [GeV]")
		ax.set_ylabel("Total efficiency")
		ax.legend()
		print(f"{figure_directory}/total_efficiency_{side}_{trigger}.png")
		fig.savefig(f"{figure_directory}/total_efficiency_{side}_{trigger}{savetag}.png")

# Probefilter efficiency alone
fig_pf, ax_pf = plt.subplots(1, 1, figsize=(10, 7))
for btype in btypes:
	ax_pf.errorbar(
		x=bin_centers, 
		y=eff[btype]["probefilter"], 
		xerr=xerrs,
		yerr=deff[btype]["probefilter"], 
		marker=".", 
		markersize=10.,
		color=colors[btype],
		label=labels[btype],
		ls="none",
		ecolor=colors[btype],
		elinewidth=1
	)
ax_pf.set_xlim(0., 30.)
ax_pf.set_ylim(1.e-3, 0.5)
ax_pf.set_yscale("log")
ax_pf.set_xlabel(r"$p_{T}$ [GeV]")
ax_pf.set_ylabel("probefilter efficiency")
ax_pf.legend()
fig_pf.savefig(f"{figure_directory}/probefilter_efficiency_log{savetag}.png")

ax_pf.set_yscale("linear")
ax_pf.set_ylim(0., 0.04)
fig_pf.savefig(f"{figure_directory}/probefilter_efficiency_linear{savetag}.png")

# Probefilter closure test alone
fig_pf, ax_pf = plt.subplots(1, 1, figsize=(10, 7))
for btype in btypes:
	ax_pf.errorbar(
		x=bin_centers, 
		y=eff[btype]["probefilter_closure"], 
		xerr=xerrs,
		yerr=deff[btype]["probefilter_closure"], 
		marker=".", 
		markersize=10.,
		color=colors[btype],
		label=labels[btype],
		ls="none",
		ecolor=colors[btype],
		elinewidth=1
	)
ax_pf.set_xlim(0., 30.)
ax_pf.set_ylim(0.95, 1.05)
ax_pf.set_xlabel(r"$p_{T}$ [GeV]")
ax_pf.set_ylabel("probefilter closure")
ax_pf.legend()
fig_pf.savefig(f"{figure_directory}/probefilter_closure{savetag}.png")

# Save efficiencies to file
# Turn into one dict
eff_deff = {}
for btype in eff.keys():
	eff_deff[btype] = {}
	for side in eff[btype].keys():
		eff_deff[btype][side] = (eff[btype][f"{side}_inclusive"], deff[btype][f"{side}_inclusive"])

with open("/home/dryu/BFrag/data/efficiency/efficiency{savetag}.pkl", "wb") as f:
	pickle.dump(eff_deff)