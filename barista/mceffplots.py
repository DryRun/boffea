# Script for making MC efficiency plots. Split into tag and probe side.
import sys
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from pprint import pprint

import mplhep
plt.style.use(mplhep.style.ROOT)
plt.tight_layout()

# Load
histograms = {
	"Bu": util.load(f"Bu2KJpsi2KMuMu/MCEfficiencyHistograms.coffea"),
	"Bs": util.load(f"Bs2PhiJpsiToKKMuMu/MCEfficiencyHistograms.coffea"),
	"Bd": util.load(f"Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms.coffea"),
}
#pprint(histograms["Bu"])

figure_directory = "/home/dryu/BFrag/data/figures/"

# Rebin
coarse_pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 50, 0., 50.)
coarse_reco_pt_axis = hist.Bin("coarse_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 50, 0., 50.)
coarse_truth_pt_axis = hist.Bin("coarse_truth_pt_axis", r"$p_{T}^{(truth)}$ [GeV]", 50, 0., 50.)

histograms["Bu"]["BuToKMuMu_fit_pt"] = histograms["Bu"]["BuToKMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bu"]["TruthBuToKMuMu_pt"] = histograms["Bu"]["TruthBuToKMuMu_pt"].rebin("pt", coarse_truth_pt_axis)

histograms["Bs"]["BsToKKMuMu_fit_pt"] = histograms["Bs"]["BsToKKMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bs"]["TruthBsToKKMuMu_pt"] = histograms["Bs"]["TruthBsToKKMuMu_pt"].rebin("pt", coarse_truth_pt_axis)

histograms["Bd"]["BdToKPiMuMu_fit_pt"] = histograms["Bd"]["BdToKPiMuMu_fit_pt"].rebin("fit_pt", coarse_reco_pt_axis)
histograms["Bd"]["TruthBdToKPiMuMu_pt"] = histograms["Bd"]["TruthBdToKPiMuMu_pt"].rebin("pt", coarse_truth_pt_axis)

"""
Probe filter efficiency
- Plot efficiency of probe filter vs. truth pT
- Actual measurement uses inclusive samples
- For closure test, use dataset_type="probefilter"
"""
def probefilter(dataset_type="inclusive"):
	fig, ax = plt.subplots(2, 1, figsize=(10,12)) #, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
	plt.style.use(mplhep.style.ROOT)

	h_bu_truthpt = histograms["Bu"]["TruthBuToKMuMu_pt"].integrate(f"dataset", (f"Bu2KJpsi2KMuMu_{dataset_type}"))
	h_bu_truthpt.axis("selection").index("inclusive").label = r"$B_u$, inclusive"
	h_bu_truthpt.axis("selection").index("probefilter").label = r"$B_u$, emul. probe filter"
	hist.plot1d(h_bu_truthpt[(["inclusive"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"red", "linestyle":"-"})
	hist.plot1d(h_bu_truthpt[(["probefilter"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"red", "linestyle":"--"})

	h_bs_truthpt = histograms["Bs"]["TruthBsToKKMuMu_pt"].integrate("dataset", (f"Bs2PhiJpsi2KKMuMu_{dataset_type}"))
	h_bs_truthpt.axis("selection").index("inclusive").label = r"$B_s$, inclusive"
	h_bs_truthpt.axis("selection").index("probefilter").label = r"$B_s$, emul. probe filter"
	hist.plot1d(h_bs_truthpt[(["inclusive", "probefilter"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"blue", "linestyle":"-"})
	#hist.plot1d(h_bs_truthpt[(["probefilter"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"blue", "linestyle":"--"})

	h_bd_truthpt = histograms["Bd"]["TruthBdToKPiMuMu_pt"].integrate("dataset", (f"Bd2KstarJpsi2KPiMuMu_{dataset_type}"))
	h_bd_truthpt.axis("selection").index("inclusive").label = r"$B_d$, inclusive"
	h_bd_truthpt.axis("selection").index("probefilter").label = r"$B_d$, emul. probe filter"
	hist.plot1d(h_bd_truthpt[(["inclusive", "probefilter"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"green", "linestyle":"-"})

	h_bu_truthpt_inclusive = h_bu_truthpt.integrate("selection", ("inclusive"))
	h_bu_truthpt_probefilter= h_bu_truthpt.integrate("selection", ("probefilter"))
	h_bu_truthpt_probefilter.label = "Efficiency"
	hist.plotratio(num=h_bu_truthpt_probefilter, denom=h_bu_truthpt_inclusive, ax=ax[1], 
		           unc="clopper-pearson", clear=False, error_opts={'color': 'red', 'marker': '.'},)

	h_bs_truthpt_inclusive = h_bs_truthpt.integrate("selection", ("inclusive"))
	h_bs_truthpt_probefilter= h_bs_truthpt.integrate("selection", ("probefilter"))
	h_bs_truthpt_probefilter.label = "Efficiency"
	hist.plotratio(num=h_bs_truthpt_probefilter, denom=h_bs_truthpt_inclusive, ax=ax[1], 
		           unc="clopper-pearson", clear=False, error_opts={'color': 'blue', 'marker': '.'},)

	h_bd_truthpt_inclusive = h_bd_truthpt.integrate("selection", ("inclusive"))
	h_bd_truthpt_probefilter= h_bd_truthpt.integrate("selection", ("probefilter"))
	h_bd_truthpt_probefilter.label = "Efficiency"
	hist.plotratio(num=h_bd_truthpt_probefilter, denom=h_bd_truthpt_inclusive, ax=ax[1], 
		           unc="clopper-pearson", clear=False, error_opts={'color': 'green', 'marker': '.'},)

	ax[0].set_yscale("log")
	ax[0].set_ylim(1., 1.e6)
	if dataset_type == "inclusive":
		ax[1].set_yscale("log")
		ax[1].set_ylim(0.001, 0.2)

	ax[0].legend(fontsize=14)
	plt.tight_layout()
	fig.savefig(f"{figure_directory}/probefilter_efficiency_{dataset_type}.png")

"""
Reconstruction * selection efficiencies vs truth pT
- Probe side: use probe filter sample
"""
def recoseleff_probe():
	for btype in ["Bu", "Bs", "Bd"]:
		fig, ax = plt.subplots(2, 1, figsize=(10,12))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
		plt.style.use(mplhep.style.ROOT)
		if btype == "Bu":
			h_truthpt = histograms["Bu"]["TruthBuToKMuMu_pt"].integrate("dataset", ["Bu2KJpsi2KMuMu_probefilter"])
		elif btype == "Bs":
			h_truthpt = histograms["Bs"]["TruthBsToKKMuMu_pt"].integrate("dataset", ["Bs2PhiJpsi2KKMuMu_probefilter"])
		elif btype == "Bd":
			h_truthpt = histograms["Bd"]["TruthBdToKPiMuMu_pt"].integrate("dataset", ["Bd2KstarJpsi2KPiMuMu_probefilter"])
		else:
			sys.exit(1)

		# Legend entries
		h_truthpt.axis("selection").index("inclusive").label = "Inclusive"
		h_truthpt.axis("selection").index("matched").label = "Reco matched"
		h_truthpt.axis("selection").index("matched_sel").label = "Reco matched * selection"
		h_truthpt.axis("selection").index("matched_probe").label = "Reco matched * selection * probe"

		# Top plot
		hist.plot1d(h_truthpt[(["inclusive"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"black"})
		hist.plot1d(h_truthpt[(["matched"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"red"})
		hist.plot1d(h_truthpt[(["matched_sel"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"green"})
		hist.plot1d(h_truthpt[(["matched_probe"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"blue"})
		ax[0].set_ylim(1., 1.e6)
		ax[0].set_yscale("log")

		# Bottom plot
		h_truthpt_inclusive     = h_truthpt.integrate("selection", (["inclusive"]))
		h_truthpt_matched       = h_truthpt.integrate("selection", (["matched"]))
		h_truthpt_matched_sel   = h_truthpt.integrate("selection", (["matched_sel"]))
		h_truthpt_matched_probe = h_truthpt.integrate("selection", (["matched_probe"]))
		h_truthpt_matched.label = "Efficiency"
		h_truthpt_matched.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=True, error_opts={'color': 'red', 'marker': '.'},)
		h_truthpt_matched_sel.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched_sel, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=False, error_opts={'color': 'green', 'marker': '.'},)
		h_truthpt_matched_probe.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched_probe, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=False, error_opts={'color': 'blue', 'marker': '.'},)
		ax[1].set_ylim(0., 0.3)

		ax[0].legend(fontsize=14)
		plt.tight_layout()
		fig.savefig(f"{figure_directory}/recosel_efficiency_probe_{btype}.png")

"""
Total efficiency for probe events: e(probefilter) * e(reco * selection * probe | probefilter)
"""
def totaleff_probe():
	for btype in ["Bu", "Bs", "Bd"]:
		# Total probe efficiency, i.e. including probefilter
		 
		# Reconstruction * selection efficiency (from the previous plot)
		if btype == "Bu":
			h_truthpt = histograms["Bu"]["TruthBuToKMuMu_pt"].integrate("dataset", ["Bu2KJpsi2KMuMu_probefilter"])
		elif btype == "Bs":
			h_truthpt = histograms["Bs"]["TruthBsToKKMuMu_pt"].integrate("dataset", ["Bs2PhiJpsi2KKMuMu_probefilter"])
		elif btype == "Bd":
			h_truthpt = histograms["Bd"]["TruthBdToKPiMuMu_pt"].integrate("dataset", ["Bd2KstarJpsi2KPiMuMu_probefilter"])
		h_truthpt_matched_probe = h_truthpt.integrate("selection", (["matched_probe"]))
		h_truthpt_inclusive     = h_truthpt.integrate("selection", (["inclusive"]))
		counts_matched_probe_probefilter = h_truthpt_matched_probe.values(sumw2=False)[()]
		counts_inclusive_probefilter = h_truthpt_inclusive.values(sumw2=False)[()]
		reco_sel_eff = counts_matched_probe_probefilter / counts_inclusive_probefilter
		reco_sel_eff_err = np.sqrt(reco_sel_eff * (1. - reco_sel_eff) / counts_inclusive_probefilter)
		print("Reco * selection efficiency:")
		print(reco_sel_eff)

		if btype == "Bu":
			h_probefilter = histograms["Bu"]["TruthBuToKMuMu_pt"].integrate("dataset", ["Bu2KJpsi2KMuMu_inclusive"])
		elif btype == "Bs":
			h_probefilter = histograms["Bs"]["TruthBsToKKMuMu_pt"].integrate("dataset", ["Bs2PhiJpsi2KKMuMu_inclusive"])
		elif btype == "Bs":
			h_probefilter = histograms["Bs"]["TruthBdToKPiMuMu_pt"].integrate("dataset", ["Bd2KstarJpsi2KPiMuMu_inclusive"])
		probefilter_counts = h_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
		print("Probe filter counts")
		print(probefilter_counts)
		inclusive_counts = h_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
		probefilter_efficiency = probefilter_counts / inclusive_counts
		probefilter_efficiency_err = np.sqrt(probefilter_efficiency * (1. - probefilter_efficiency) / inclusive_counts)
		print("Probe filter efficiency:")
		print(probefilter_efficiency)
		total_efficiency = reco_sel_eff * probefilter_efficiency
		total_efficiency_err = total_efficiency * np.sqrt((reco_sel_eff_err / reco_sel_eff)**2 + (probefilter_efficiency_err / probefilter_efficiency)**2)
		print("Total efficiency:")
		print(total_efficiency)

		bin_centers = h_truthpt.axis("coarse_truth_pt_axis").centers()

		# Plot
		fig, ax = plt.subplots(1, 1, figsize=(10,7))
		ax.errorbar(x=bin_centers, y=total_efficiency, yerr=total_efficiency_err, marker=".", markersize=10.)
		ax.set_xlim(0., 30.)
		ax.set_ylim(0.000001, 0.002)
		ax.set_yscale("log")
		fig.savefig(f"{figure_directory}/total_efficiency_{btype}.png")

def recoseleff_tag():
	for btype in ["Bu", "Bs", "Bd"]:
		fig, ax = plt.subplots(2, 1, figsize=(10,12))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
		plt.style.use(mplhep.style.ROOT)
		if btype == "Bu":
			h_truthpt = histograms["Bu"]["TruthBuToKMuMu_pt"].integrate("dataset", ["Bu2KJpsi2KMuMu_inclusive"])
		elif btype == "Bs":
			h_truthpt = histograms["Bs"]["TruthBsToKKMuMu_pt"].integrate("dataset", ["Bs2PhiJpsi2KKMuMu_inclusive"])
		elif btype == "Bd":
			h_truthpt = histograms["Bd"]["TruthBdToKPiMuMu_pt"].integrate("dataset", ["Bd2KstarJpsi2KPiMuMu_inclusive"])
		else:
			sys.exit(1)

		# Legend entries
		h_truthpt.axis("selection").index("inclusive").label = "Inclusive"
		h_truthpt.axis("selection").index("matched").label = "Reco matched"
		h_truthpt.axis("selection").index("matched_sel").label = "Reco matched * selection"
		h_truthpt.axis("selection").index("matched_tag").label = "Reco matched * selection * tag"

		# Top plot
		hist.plot1d(h_truthpt[(["inclusive"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"black"})
		hist.plot1d(h_truthpt[(["matched"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"red"})
		hist.plot1d(h_truthpt[(["matched_sel"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"green"})
		hist.plot1d(h_truthpt[(["matched_tag"]),:], ax=ax[0], clear=False, overlay="selection", line_opts={"color":"blue"})
		ax[0].set_ylim(1., 1.e6)
		ax[0].set_yscale("log")

		# Bottom plot
		h_truthpt_inclusive     = h_truthpt.integrate("selection", (["inclusive"]))
		h_truthpt_matched       = h_truthpt.integrate("selection", (["matched"]))
		h_truthpt_matched_sel  = h_truthpt.integrate("selection", (["matched_sel"]))
		h_truthpt_matched_probe = h_truthpt.integrate("selection", (["matched_tag"]))
		h_truthpt_matched.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=True, error_opts={'color': 'red', 'marker': '.'},)
		h_truthpt_matched_sel.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched_sel, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=False, error_opts={'color': 'green', 'marker': '.'},)
		h_truthpt_matched_probe.label = "Efficiency"
		hist.plotratio(num=h_truthpt_matched_probe, denom=h_truthpt_inclusive, ax=ax[1], 
		               unc="clopper-pearson", clear=False, error_opts={'color': 'blue', 'marker': '.'},)

		ax[0].legend(fontsize=14)
		ax[1].set_ylim(0., 0.4)
		plt.tight_layout()
		fig.savefig(f"{figure_directory}/recosel_efficiency_tag_{btype}.png")

"""
Truth vs reco response
"""
def responsematrix():
	for btype in ["Bu", "Bs", "Bd"]:
		if btype == "Bu":
			dataset_bname = "Bu2KJpsi2KMuMu"
			histogram_bname = "BuToKMuMu"
			axis_bname = "BToKMuMu"
		elif btype == "Bs":
			dataset_bname = "Bs2PhiJpsi2KKMuMu"
			histogram_bname = "BsToKKMuMu"
			axis_bname = "BsToKKMuMu"
		elif btype == "Bd":
			dataset_bname = "Bd2KstarJpsi2KPiMuMu"
			histogram_bname = "BdToKPiMuMu"
			axis_bname = "BdToKPiMuMu"
		for selection in ["tag", "probe", "inclusive"]:
			if selection == "inclusive":
				hist2d = histograms[btype][f"Truth{histogram_bname}_truthpt_recopt"].integrate("dataset", ([f"{dataset_bname}_inclusive"])).integrate("selection", "inclusive")
			elif selection == "tag":
				hist2d = histograms[btype][f"Truth{histogram_bname}_truthpt_recopt"].integrate("dataset", ([f"{dataset_bname}_inclusive"])).integrate("selection", "matched_tag")
			elif selection == "probe":
				hist2d = histograms[btype][f"Truth{histogram_bname}_truthpt_recopt"].integrate("dataset", ([f"{dataset_bname}_probefilter"])).integrate("selection", "matched_probe")
			fig, ax = plt.subplots(1, 1, figsize=(10,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
			hist2d = hist2d.rebin(f"reco_pt", coarse_reco_pt_axis)
			hist2d = hist2d.rebin(f"truth_pt", coarse_truth_pt_axis)
			hist.plot2d(hist2d, xaxis="coarse_reco_pt", ax=ax)
			plt.tight_layout()
			fig.savefig(f"{figure_directory}/responsematrix_{btype}_{selection}.png")


def truth_nMuon():
	fig, ax = plt.subplots(1, 1, figsize=(10,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
	plt.style.use(mplhep.style.ROOT)

	# Normalize histograms
	for btype in ["Bu", "Bs"]:
		normalizations = histograms[btype]["nTruthMuon"].integrate("nTruthMuon")
		sfs = {}
		for k, v in normalizations.values().items():
			sfs[k[0]] = 1. / v
		#pprint(sfs)
		histograms[btype]["nTruthMuon"].scale(sfs, axis="dataset")

	for dataset_type in ["inclusive", "probefilter"]:
		print(dataset_type)
		for btype in ["Bu", "Bs"]:
			print(btype)
			if btype == "Bu":
				bname = "Bu2KJpsi2KMuMu"
			elif btype == "Bs":
				bname = "Bs2PhiJpsi2KKMuMu"
			dataset_name = f"{bname}_{dataset_type}"
			#h_truth_nMuon = histograms[btype]["nTruthMuon"].integrate("dataset", [f"{bname}_{dataset_type}"])
			#print(dataset_name)
			#print(histograms[btype]["nTruthMuon"].axis("dataset").identifiers())
			histograms[btype]["nTruthMuon"].axis("dataset").index(dataset_name).label = f"{dataset_type}, {btype}"
			#print(histograms[btype]["nTruthMuon"][([dataset_name]),:])
			hist.plot1d(histograms[btype]["nTruthMuon"][([dataset_name]),:], ax=ax, clear=False, overlay="dataset")
	plt.tight_layout()
	fig.savefig(f"{figure_directory}/truth_nMuon.png")

if __name__ == "__main__":
	probefilter()
	probefilter(dataset_type="probefilter")
	recoseleff_probe()
	recoseleff_tag()
	truth_nMuon()
	responsematrix()
	totaleff_probe()

	print("Done.")