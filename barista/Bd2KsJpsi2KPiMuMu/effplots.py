import os
import sys
import numpy as np 
from coffea import util, hist
import matplotlib.pyplot as plt

import mplhep
plt.style.use(mplhep.style.ROOT)
plt.tight_layout()

figure_directory = "/home/dryu/BFrag/data/efficiency/figures"

triggers = ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]
btypes = ["Bu", "Bs", "Bd"]
btype_longnames = {
	"Bu": "Bu2KJpsi2KMuMu",
	"Bs": "Bs2PhiJpsiToKKMuMu",
	"Bd": "BdToKsJpsi2KPiMuMu"
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

coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 10, 0., 50.)
coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 10, 0., 50.)
coarse_truth_pt_axis = hist.Bin("coarse_truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 10, 0., 50.)

bigh_reco_pt = {}
bigh_truth_pt = {}
for btype in btypes:
	bigh_reco_pt = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
	bigh_truth_pt = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt"].rebin("pt", coarse_truth_pt_axis)

eff = {}
deff = {}

# Reconstruction * selection efficiency, probe side
# - For Bd, include both matched and matched-swap
for btype in btypes:
	eff[btype] = {}
	deff[btype] = {}
	for trigger in triggers:
		h_truth_pt = bigh_truth_pt.integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])

		if btype == "Bd"
			h_truth_pt_matched_probe = h_truth_pt.integrate("selection", ([f"matched_probe_{trigger}", f"matched_swap_probe_{trigger}"]))
		else:
			h_truth_pt_matched_probe = h_truth_pt.integrate("selection", ([f"matched_probe_{trigger}"]))
		h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

		counts_matched_probe = h_truth_pt_matched_probe.values(sumw2=False)[()]
		counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

		eff[btype][f"recosel_probe_{trigger}"] = counts_matched_probe / counts_inclusive
		deff[btype][f"recosel_probe_{trigger}"] = np.sqrt(eff[btype][f"recosel_probe_{trigger}"] * (1. - eff[f"recosel_probe_{trigger}"]) / counts_inclusive)

# MC probefilter efficiency
for btype in btypes:
	bigh_probefilter   = bigh_truth_pt.integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	eff[btype]["probefilter"] = probefilter_counts / inclusive_counts
	deff[btype]["probefilter"] = np.sqrt(eff[btype]['probefilter'] * (1. - eff[btype]['probefilter']) / inclusive_counts)

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
		h_truth_pt = bigh_truth_pt.integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
		if btype == "Bd":
			h_truth_pt_matched_tag = h_truth_pt.integrate("selection", ([f"matched_tag_{trigger}", f"matched_swap_tag_{trigger}"]))
		else:
			h_truth_pt_matched_tag = h_truth_pt.integrate("selection", ([f"matched_tag_{trigger}"]))
		h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

		counts_matched_tag = h_truth_pt_matched_tag.values(sumw2=False)[()]
		counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

		eff[btype][f"tag_{trigger}"] = counts_matched_tag / counts_inclusive
		deff[btype][f"tag_{trigger}"] = np.sqrt(eff[btype][f"tag_{trigger}"] * (1. - eff[btype][f"tag_{trigger}"]) / counts_inclusive)


# Plots
bin_centers = bigh_truth_pt.axis("coarse_truth_pt_axis").centers()
for btype in btypes:
	for trigger in triggers:
		for side in ["tag", "probe"]:
			fig, ax = plt.subplots(1, 1, figsize=(10,7))
			ax.errorbar(
				x=bin_centers, 
				y=eff[f"{side}_{trigger}"], 
				yerr=deff[f"{side}_{trigger}"], 
				marker=".", 
				markersize=10.)
			ax.set_xlim(0., 30.)
			if side == "probe":
				ax.set_ylim(1.e-7, 5.e-3)
			elif side == "tag":
				ax.set_ylim(1.e-5, 0.5)
			ax.set_yscale("log")
			ax.set_xlabel(r"$p_{T}$ [GeV]")
			ax.set_ylabel("Total efficiency")
			fig.savefig(f"{figure_directory}/single/total_efficiency_{btype}_{side}_{trigger}.png")
