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

# MC probefilter efficiency
mycoffea = util.load(f"MCEfficiencyHistograms.coffea")

coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 10, 0., 50.)
coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 10, 0., 50.)
coarse_truth_pt_axis = hist.Bin("coarse_truth_pt_axis", r"$p_{T}^{(truth)}$ [GeV]", 10, 0., 50.)

bigh_reco_pt = mycoffea["BsToKKMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
bigh_truth_pt = mycoffea["TruthBsToKKMuMu_pt"].rebin("pt", coarse_truth_pt_axis)

eff = {}
deff = {}

# Reconstruction * selection efficiency, probe side
for trigger in triggers:
	h_truth_pt = bigh_truth_pt.integrate("dataset", ["Bs2PhiJpsi2KKMuMu_probefilter"])

	h_truth_pt_matched_probe = h_truth_pt.integrate("selection", ([f"matched_probe_{trigger}"]))
	h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

	counts_matched_probe = h_truth_pt_matched_probe.values(sumw2=False)[()]
	counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

	eff[f"recosel_probe_{trigger}"] = counts_matched_probe / counts_inclusive
	deff[f"recosel_probe_{trigger}"] = np.sqrt(eff[f"recosel_probe_{trigger}"] * (1. - eff[f"recosel_probe_{trigger}"]) / counts_inclusive)

# MC probefilter efficiency
bigh_probefilter   = bigh_truth_pt.integrate("dataset", ["Bs2PhiJpsi2KKMuMu_inclusive"])
probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
eff["probefilter"] = probefilter_counts / inclusive_counts
deff["probefilter"] = np.sqrt(eff['probefilter'] * (1. - eff['probefilter']) / inclusive_counts)

# Total probe efficiency
for trigger in triggers:
	eff[f"probe_{trigger}"] = eff[f"recosel_probe_{trigger}"] * eff["probefilter"]
	deff[f"probe_{trigger}"] = eff[f"probe_{trigger}"] * np.sqrt(
						(deff[f"recosel_probe_{trigger}"] / eff[f"recosel_probe_{trigger}"])**2 
						+ (deff["probefilter"] / eff["probefilter"])**2)


# Reconstruction * selection efficiency, tag side
for trigger in triggers:
	h_truth_pt = bigh_truth_pt.integrate("dataset", ["Bs2PhiJpsi2KKMuMu_inclusive"])

	h_truth_pt_matched_tag = h_truth_pt.integrate("selection", ([f"matched_tag_{trigger}"]))
	h_truth_pt_inclusive     = h_truth_pt.integrate("selection", (["inclusive"]))

	counts_matched_tag = h_truth_pt_matched_tag.values(sumw2=False)[()]
	counts_inclusive = h_truth_pt_inclusive.values(sumw2=False)[()]

	eff[f"tag_{trigger}"] = counts_matched_tag / counts_inclusive
	deff[f"tag_{trigger}"] = np.sqrt(eff[f"tag_{trigger}"] * (1. - eff[f"tag_{trigger}"]) / counts_inclusive)


# Plots
bin_centers = bigh_truth_pt.axis("coarse_truth_pt_axis").centers()
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
		ax.set_ylim(0.000001, 0.02)
		ax.set_yscale("log")
		ax.set_xlabel(r"$p_{T}$ [GeV]")
		ax.set_ylabel("Total efficiency")
		fig.savefig(f"{figure_directory}/total_efficiency_Bs_{side}_{trigger}.png")
