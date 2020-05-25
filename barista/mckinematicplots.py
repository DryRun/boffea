# Script for making MC efficiency plots. Split into tag and probe side.
import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from pprint import pprint

import mplhep
plt.style.use(mplhep.style.LHCb)
plt.tight_layout()

figure_directory = "/home/dryu/BFrag/data/figures/kinematic"

# Load
histograms = {
	"Bu": util.load(f"Bu2KJpsi2KMuMu/MCEfficiencyHistograms.coffea"),
	"Bs": util.load(f"Bs2PhiJpsiToKKMuMu/MCEfficiencyHistograms.coffea"),
	"Bd": util.load(f"Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms.coffea")
}
#pprint(histograms["Bu"])

# Rebin
coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 50, 0., 50.)
coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 50, 0., 50.)
coarse_truth_pt_axis = hist.Bin("coarse_truth_pt_axis", r"$p_{T}^{(truth)}$ [GeV]", 50, 0., 50.)
histograms["Bu"]["BuToKMuMu_fit_pt"] = histograms["Bu"]["BuToKMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bu"]["TruthBuToKMuMu_pt"] = histograms["Bu"]["TruthBuToKMuMu_pt"].rebin("pt", coarse_truth_pt_axis)
histograms["Bu"]["TruthBuToKMuMu_k_pt"] = histograms["Bu"]["TruthBuToKMuMu_k_p4"].sum("eta", "phi", "mass")
histograms["Bu"]["TruthBuToKMuMu_mup_pt"] = histograms["Bu"]["TruthBuToKMuMu_mup_p4"].sum("eta", "phi", "mass")
histograms["Bu"]["TruthBuToKMuMu_mum_pt"] = histograms["Bu"]["TruthBuToKMuMu_mum_p4"].sum("eta", "phi", "mass")

histograms["Bs"]["BsToKKMuMu_fit_pt"] = histograms["Bs"]["BsToKKMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bs"]["TruthBsToKKMuMu_pt"] = histograms["Bs"]["TruthBsToKKMuMu_pt"].rebin("pt", coarse_truth_pt_axis)
histograms["Bs"]["TruthBsToKKMuMu_kp_pt"] = histograms["Bs"]["TruthBsToKKMuMu_kp_p4"].sum("eta", "phi", "mass")
histograms["Bs"]["TruthBsToKKMuMu_km_pt"] = histograms["Bs"]["TruthBsToKKMuMu_km_p4"].sum("eta", "phi", "mass")
histograms["Bs"]["TruthBsToKKMuMu_mup_pt"] = histograms["Bs"]["TruthBsToKKMuMu_mup_p4"].sum("eta", "phi", "mass")
histograms["Bs"]["TruthBsToKKMuMu_mum_pt"] = histograms["Bs"]["TruthBsToKKMuMu_mum_p4"].sum("eta", "phi", "mass")

histograms["Bd"]["BdToKPiMuMu_fit_pt"] = histograms["Bd"]["BdToKPiMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bd"]["TruthBdToKPiMuMu_pt"] = histograms["Bd"]["TruthBdToKPiMuMu_pt"].rebin("pt", coarse_truth_pt_axis)
histograms["Bd"]["TruthBdToKPiMuMu_k_pt"] = histograms["Bd"]["TruthBdToKPiMuMu_k_p4"].sum("eta", "phi", "mass")
histograms["Bd"]["TruthBdToKPiMuMu_mup_pt"] = histograms["Bd"]["TruthBdToKPiMuMu_mup_p4"].sum("eta", "phi", "mass")
histograms["Bd"]["TruthBdToKPiMuMu_mum_pt"] = histograms["Bd"]["TruthBdToKPiMuMu_mum_p4"].sum("eta", "phi", "mass")



# hname=histogram name from mc_efficiency_processor
# btype=Bu or Bs
# dataset_name=dataset name on "dataset" axis
def kinematic_selection_plot(hname, btype, dataset_name, selection_name, yscale="linear"):
	print(f"Making plot: {hname}, {btype}, {dataset_name}, {selection_name}, {yscale}")
	fig, ax = plt.subplots(1, 1, figsize=(10, 7))
	plt.style.use(mplhep.style.ROOT)
	hist.plot1d(histograms[btype][hname].integrate("dataset", [(dataset_name)]).integrate("selection", [(selection_name)]), 
			error_opts={"marker":".", "linestyle":"none", "markersize":10., "color":"k", "elinewidth":1})
	if "inclusive" in dataset_name:
		dataset_tag = "inclusive"
	elif "probefilter" in dataset_name:
		dataset_tag = "probefilter"
	plt.tight_layout()
	fig.savefig(f"{figure_directory}/{hname}_{btype}_{dataset_tag}_{selection_name}.png")
	plt.close(fig)

# Bu
for dataset_name in ["Bu2KJpsi2KMuMu_probefilter", "Bu2KJpsi2KMuMu_inclusive"]:
	# Reco histograms
	for hname in ["BuToKMuMu_fit_pt", "BuToKMuMu_fit_eta", "BuToKMuMu_fit_phi", "BuToKMuMu_fit_mass", "BToKMuMu_chi2", "BuToKMuMu_fit_cos2D", "BuToKMuMu_l_xy", "BuToKMuMu_l_xy_sig", "BuToKMuMu_jpsi_mass", ]:
		for selection in ["inclusive", "reco", "tag", "probe"]:
			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bu", dataset_name, selection, yscale=yscale)

	# Truth histograms
	for hname in ["TruthBuToKMuMu_k_pt", "TruthBuToKMuMu_mup_pt", "TruthBuToKMuMu_mum_pt"]:
		for selection in ["inclusive", "matched", "matched_tag", "matched_probe"]:
			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bu", dataset_name, selection, yscale=yscale)

# Bs
for dataset_name in ["Bs2PhiJpsi2KKMuMu_probefilter", "Bs2PhiJpsi2KKMuMu_inclusive"]:
	# Reco histograms
	for hname in ["BsToKKMuMu_fit_pt", "BsToKKMuMu_fit_eta", "BsToKKMuMu_fit_phi", "BsToKKMuMu_fit_mass", "BsToKKMuMu_chi2", "BsToKKMuMu_fit_cos2D", "BsToKKMuMu_l_xy", "BsToKKMuMu_l_xy_sig", "BsToKKMuMu_jpsi_mass", "BsToKKMuMu_phi_mass", ]:
		for selection in ["inclusive", "reco", "tag", "probe"]:
			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bs", dataset_name, selection, yscale=yscale)

	# Truth histograms
	for hname in ["TruthBsToKKMuMu_kp_pt", "TruthBsToKKMuMu_km_pt", "TruthBsToKKMuMu_mup_pt", "TruthBsToKKMuMu_mum_pt"]:
		for selection in ["inclusive", "matched", "matched_tag", "matched_probe"]:

			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bs", dataset_name, selection, yscale=yscale)

# Bd
for dataset_name in ["Bd2KstarJpsi2KPiMuMu_probefilter", "Bd2KstarJpsi2KPiMuMu_inclusive"]:
	# Reco histograms
	for hname in ["BdToKPiMuMu_fit_pt", "BdToKPiMuMu_fit_eta", "BdToKPiMuMu_fit_phi", "BdToKPiMuMu_fit_best_mass", "BdToKPiMuMu_chi2", "BdToKPiMuMu_fit_cos2D", "BdToKPiMuMu_l_xy", "BdToKPiMuMu_l_xy_sig", "BdToKPiMuMu_jpsi_mass", "BdToKPiMuMu_fit_best_mkstar", ]:
		for selection in ["inclusive", "reco", "tag", "probe"]:
			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bd", dataset_name, selection, yscale=yscale)

	# Truth histograms
	for hname in ["TruthBdToKPiMuMu_k_pt", "TruthBdToKPiMuMu_mup_pt", "TruthBdToKPiMuMu_mum_pt"]:
		for selection in ["inclusive", "matched", "matched_tag", "matched_probe"]:
			if "pt" in hname:
				yscale = "log"
			else:
				yscale = "linear"
			kinematic_selection_plot(hname, "Bd", dataset_name, selection, yscale=yscale)
