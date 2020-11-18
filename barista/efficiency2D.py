import os
import sys
import numpy as np 
from coffea import util, hist
import matplotlib.pyplot as plt
import math
from pprint import pprint
import pickle
import copy
import ROOT
from brazil.seaborn_colors import create_hls_palette_fast

#import mplhep
#plt.style.use(mplhep.style.LHCb)
#plt.tight_layout()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

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

axes = {}
# pt: probe=5 bins, tag=13 bins
axes["pt"] = {
	"probe": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([8., 13., 18., 23., 28., 33., 100.])), # [5., 10., 15., 20., 25., 30., 100.0]
	"tag": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0, 100.0]))
}
axes["y"] = {
	"probe": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.5+0.5, 0.5))), 
	"tag": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.5+0.5, 0.5)))
}

bigh_reco = {}
bigh_truth = {}

for btype in btypes:
	bigh_reco[btype] = {}
	bigh_truth[btype] = {}
	for side in ["probe", "tag"]:
		bigh_reco[btype][side] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt_absy_mass"]\
			.integrate("fit_mass")\
			.rebin("fit_absy", axes["y"][side])\
			.rebin("fit_pt", axes["pt"][side])
		bigh_truth[btype][side] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt_absy_mass"]\
			.integrate("mass")\
			.rebin("absy", axes["y"][side])\
			.rebin("pt", axes["pt"][side])

# Probefilter efficency, needed for probe side
eff_probefilter = {}
deff_probefilter = {}
avg_eff_probefilter = {}
davg_eff_probefilter = {}
this_avg_eff_probefilter = None 
this_sum_inclusive_counts = None
first = True

for btype in btypes:
	bigh_probefilter        = bigh_truth[btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts      = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts        = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	eff_probefilter[btype]  = probefilter_counts / inclusive_counts
	deff_probefilter[btype] = np.sqrt(eff_probefilter[btype] * (1. - eff_probefilter[btype]) / inclusive_counts)

	bigh_probefilter   = bigh_truth[btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
	probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
	inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
	if first:
		this_avg_eff_probefilter = copy.deepcopy(probefilter_counts)
		sum_inclusive_counts = copy.deepcopy(inclusive_counts)
	else:
		this_avg_eff_probefilter = this_avg_eff_probefilter + probefilter_counts
		sum_inclusive_counts = sum_inclusive_counts + inclusive_counts
	first = False
avg_eff_probefilter = this_avg_eff_probefilter / sum_inclusive_counts
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
		if side == "probe":
			eff[btype]["probe_total"] = {}
			deff[btype]["probe_total"] = {}

		for trigger in triggers:
			if side == "probe":
				h_truth_var = bigh_truth[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			elif side == "tag":
				h_truth_var = bigh_truth[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			if side == "probe":
				h_reco_var = bigh_reco[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
			elif side == "tag":
				h_reco_var = bigh_reco[btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])

			use_truth_side = True
			if use_truth_side:
				if btype == "Bd":
					h_truth_var_matched = h_truth_var.integrate("selection", ([f"matched_{side}_{trigger}", f"matched_swap_{side}_{trigger}"]))
				else:
					h_truth_var_matched = h_truth_var.integrate("selection", ([f"matched_{side}_{trigger}"]))
				h_truth_var_inclusive     = h_truth_var.integrate("selection", (["inclusive"]))
				counts_matched = h_truth_var_matched.values(sumw2=False)[()]
				counts_inclusive = h_truth_var_inclusive.values(sumw2=False)[()]
			else:
				if btype == "Bd":
					h_reco_var_matched = h_reco_var.integrate("selection", ([f"{side}match_{trigger}", f"{side}matchswap_{side}_{trigger}"]))
				else:
					h_reco_var_matched = h_reco_var.integrate("selection", ([f"{side}match_{trigger}"]))
				h_reco_var_inclusive     = h_reco_var.integrate("selection", (["inclusive"]))
				h_truth_var_inclusive     = h_truth_var.integrate("selection", (["inclusive"]))
				counts_matched = h_reco_var_matched.values(sumw2=False)[()]
				counts_inclusive = h_truth_var_inclusive.values(sumw2=False)[()]

			eff_recosel = counts_matched / counts_inclusive
			deff_recosel = np.sqrt(eff_recosel * (1. - eff_recosel) / counts_inclusive)

			eff[btype][side][trigger] = eff_recosel
			deff[btype][side][trigger] = deff_recosel

			if side == "probe":
				eff[btype]["probe_total"][trigger] = eff_recosel * avg_eff_probefilter
				deff[btype]["probe_total"][trigger] = eff[btype]["probe_total"][trigger] * np.sqrt(
					(deff_recosel / eff_recosel)**2 + (davg_eff_probefilter / avg_eff_probefilter)**2
					)


# Efficiencies for final trigger strategies
total_lumis = {
    "HLT_Mu12_IP6": 34698.771755487,
    "HLT_Mu9_IP6": 33577.176480857,
    "HLT_Mu9_IP5": 20890.692760413,
    "HLT_Mu7_IP4": 6939.748065950
}
unique_lumis = {
    "HLT_Mu12_IP6": 34698.771755487 - 33577.176480857,
    "HLT_Mu9_IP6": 33577.176480857 - 20890.692760413,
    "HLT_Mu9_IP5": 20890.692760413 - 6939.748065950,
    "HLT_Mu7_IP4": 6939.748065950
}
total_lumi = 34698.771755487
f_eff = ROOT.TFile("/home/dryu/BFrag/data/efficiency/efficiency2D.root", "RECREATE")
hist_eff = {}
for btype in btypes:
	hist_eff[btype] = {}
	for side in ["tag", "probe", "probe_total"]:
		hist_eff[btype][side] = {}

		# HLT_all strategy
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

		# HLT_Mu9 combined
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

		# Convert to TH2Ds, for use with coffea
		for trigger_strategy in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9", "HLT_all"]:
			pt_edges = axes["pt"][side.replace("_total", "")].edges()
			y_edges = axes["y"][side.replace("_total", "")].edges()
			hist_eff[btype][side][trigger_strategy] = ROOT.TH2D("h_trigeff2D_{}_{}_{}".format(btype, side, trigger_strategy),
										"h_trigeff2D_{}_{}_{}".format(btype, side, trigger_strategy),
										len(pt_edges), pt_edges, 
										len(y_edges), y_edges)
			for ipt in range(axes["pt"][side.replace("_total", "")].size-3):
				for iy in range(axes["y"][side.replace("_total", "")].size-3):
					#print("(ipt, iy) = ({}, {})".format(ipt, iy))
					hist_eff[btype][side][trigger_strategy].SetBinContent(ipt + 1, iy + 1, eff[btype][side][trigger_strategy][ipt][iy])
					hist_eff[btype][side][trigger_strategy].SetBinError(ipt + 1, iy + 1, deff[btype][side][trigger_strategy][ipt][iy])

			f_eff.cd()
			hist_eff[btype][side][trigger_strategy].Write()

# Plots
ROOT.gStyle.SetPaintTextFormat("4.3f");

canvases = {}
for btype in btypes:
	for side in ["tag", "probe", "probe_total"]:
		for trigger_strategy in ["HLT_Mu7_IP4", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_all"]:
			# 2D color plots of efficiency
			cname = f"eff2D_{btype}_{side}_{trigger_strategy}"
			canvases[cname] = ROOT.TCanvas(cname, cname, 800, 600)
			canvases[cname].SetLogz()
			canvases[cname].SetRightMargin(0.25)
			if side == "probe" or side == "tag":
				hist_eff[btype][side][trigger_strategy].SetMinimum(1.e-5)
				hist_eff[btype][side][trigger_strategy].SetMaximum(0.5)
			else:
				hist_eff[btype][side][trigger_strategy].SetMinimum(1.e-6)
				hist_eff[btype][side][trigger_strategy].SetMaximum(5.e-3)

			hist_eff[btype][side][trigger_strategy].GetXaxis().SetTitle("p_{T} [GeV]")
			hist_eff[btype][side][trigger_strategy].GetYaxis().SetTitle("|y|")
			hist_eff[btype][side][trigger_strategy].Draw("colz text89")

			print(f"{figure_directory}/{cname}.png")
			canvases[cname].SaveAs(f"{figure_directory}/{cname}.png")

			# 1D overlaid lines eff vs. y, one line per pT bin
			cname = f"eff1D_ptslices_{btype}_{side}_{trigger_strategy}"
			canvases[cname] = ROOT.TCanvas(cname, cname, 800, 600)
			canvases[cname].SetLogy()
			effhists_y = {}
			frame = ROOT.TH1F("frame", "frame", 10, 0., 3.5)
			frame.SetMinimum(1.e-6)
			frame.SetMaximum(0.5)
			frame.GetXaxis().SetTitle("|y|")
			frame.GetYaxis().SetTitle("Efficiency")
			frame.Draw()
			legend = ROOT.TLegend(0.7, 0.3, 0.85, 0.8)
			legend.SetBorderSize(0)
			legend.SetFillColor(0)
			legend.SetFillStyle(0)
			legend.SetHeader("p_{T} range")
			pt_palette = create_hls_palette_fast(n_colors=hist_eff[btype][side][trigger_strategy].GetXaxis().GetNbins())
			for ptbin in range(1, hist_eff[btype][side][trigger_strategy].GetXaxis().GetNbins()):
				effhists_y[ptbin] = hist_eff[btype][side][trigger_strategy].ProjectionY("_ptbin{}".format(ptbin), ptbin, ptbin)
				effhists_y[ptbin].SetMarkerStyle(24)
				effhists_y[ptbin].SetMarkerColor(pt_palette[ptbin])
				effhists_y[ptbin].SetLineColor(pt_palette[ptbin])
				effhists_y[ptbin].Draw("same")
				legend.AddEntry(effhists_y[ptbin], "{}-{}".format(hist_eff[btype][side][trigger_strategy].GetXaxis().GetBinLowEdge(ptbin), hist_eff[btype][side][trigger_strategy].GetXaxis().GetBinUpEdge(ptbin)))
			legend.Draw()
			canvases[cname].SaveAs(f"{figure_directory}/{cname}.png")

			# 1D overlaid lines eff vs. pt, one line per y bin
			cname = f"eff1D_yslices_{btype}_{side}_{trigger_strategy}"
			canvases[cname] = ROOT.TCanvas(cname, cname, 800, 600)
			canvases[cname].SetLogy()
			effhists_pt = {}
			if side == "tag":
				ptmax = 50.
			elif side == "probe" or side == "probe_total":
				ptmax = 30.

			frame = ROOT.TH1F("frame", "frame", 10, 0., ptmax)
			frame.SetMinimum(1.e-6)
			frame.SetMaximum(0.5)
			frame.GetXaxis().SetTitle("p_{T} [GeV]")
			frame.GetYaxis().SetTitle("Efficiency")
			frame.Draw()
			legend = ROOT.TLegend(0.6, 0.25, 0.95, 0.6)
			legend.SetBorderSize(0)
			legend.SetFillColor(0)
			legend.SetFillStyle(0)
			legend.SetHeader("|y| range")
			y_palette = create_hls_palette_fast(n_colors=hist_eff[btype][side][trigger_strategy].GetYaxis().GetNbins())
			for ybin in range(1, hist_eff[btype][side][trigger_strategy].GetYaxis().GetNbins()):
				effhists_pt[ybin] = hist_eff[btype][side][trigger_strategy].ProjectionX("_ybin{}".format(ybin), ybin, ybin)
				effhists_pt[ybin].SetMarkerStyle(24)
				effhists_pt[ybin].SetMarkerColor(y_palette[ybin])
				effhists_pt[ybin].SetLineColor(y_palette[ybin])
				effhists_pt[ybin].Draw("same")
				legend.AddEntry(effhists_pt[ybin], "{}-{}".format(hist_eff[btype][side][trigger_strategy].GetYaxis().GetBinLowEdge(ybin), hist_eff[btype][side][trigger_strategy].GetYaxis().GetBinUpEdge(ybin)))
			legend.Draw()
			canvases[cname].SaveAs(f"{figure_directory}/{cname}.png")

			cname = f"effRelUnc_{btype}_{side}_{trigger_strategy}"
			canvases[cname] = ROOT.TCanvas(cname, cname, 800, 600)
			canvases[cname].SetRightMargin(0.25)

			h_relunc = hist_eff[btype][side][trigger_strategy].Clone()
			for xbin in range(1, hist_eff[btype][side][trigger_strategy].GetXaxis().GetNbins() + 1):
				for ybin in range(1, hist_eff[btype][side][trigger_strategy].GetYaxis().GetNbins() + 1):
					if hist_eff[btype][side][trigger_strategy].GetBinContent(xbin, ybin) > 0:
						h_relunc.SetBinContent(xbin, ybin, hist_eff[btype][side][trigger_strategy].GetBinError(xbin, ybin) / hist_eff[btype][side][trigger_strategy].GetBinContent(xbin, ybin))
					else:
						h_relunc.SetBinContent(xbin, ybin, -0.1)
			h_relunc.Draw("colz text89")
			canvases[cname].SaveAs(f"{figure_directory}/{cname}.png")


f_eff.Close()
