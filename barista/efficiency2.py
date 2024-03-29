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

vars = ["pt", "y"]

# MC probefilter efficiency
coffea_files = {}
for btype in btypes:
	coffea_files[btype] = util.load(f"{btype_longnames[btype]}/MCEfficiencyHistograms.coffea")

axes = {}
axes["pt"] = {
	"probe": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([8., 13., 18., 23., 28., 33.])), 
	"tag": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0]))
}
axes["y"] = {
	"probe": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25))), 
	"tag": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25)))
}

bigh_reco = {}
bigh_truth = {}

for var in vars:
	bigh_reco[var] = {}
	bigh_truth[var] = {}
for btype in btypes:
	bigh_reco["pt"][btype] = {}
	bigh_truth["pt"][btype] = {}
	for side in ["probe", "tag"]:
		bigh_reco["pt"][btype][side] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt_absy_mass"].integrate("fit_mass").integrate("fit_absy", slice(0., 2.25)).rebin("fit_pt", axes["pt"][side])
		bigh_truth["pt"][btype][side] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt_absy_mass"].integrate("mass").integrate("absy", slice(0., 2.25)).rebin("pt", axes["pt"][side])

	bigh_reco["y"][btype] = {}
	bigh_truth["y"][btype] = {}
	for side in ["probe", "tag"]:
		bigh_reco["y"][btype][side] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt_absy_mass"].integrate("fit_mass").integrate("fit_pt", slice(10.0, 30.0)).rebin("fit_absy", axes["y"][side])
		bigh_truth["y"][btype][side] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt_absy_mass"].integrate("mass").integrate("pt", slice(10.0, 30.0)).rebin("absy", axes["y"][side])


# Probefilter efficency, needed for probe side
eff_probefilter = {}
deff_probefilter = {}
avg_eff_probefilter = {}
davg_eff_probefilter = {}
for var in vars:
	eff_probefilter[var] = {}
	deff_probefilter[var] = {}
	this_avg_eff_probefilter = None 
	this_sum_inclusive_counts = None
	first = True

	for btype in btypes:
		bigh_probefilter   = bigh_truth[var][btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
		probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
		inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
		eff_probefilter[var][btype] = probefilter_counts / inclusive_counts
		deff_probefilter[var][btype] = np.sqrt(eff_probefilter[var][btype] * (1. - eff_probefilter[var][btype]) / inclusive_counts)

		bigh_probefilter   = bigh_truth[var][btype]["probe"].integrate("dataset", [f"{btype_longnames[btype]}_inclusive"])
		probefilter_counts = bigh_probefilter.integrate("selection", (["probefilter"])).values(sumw2=False)[()]
		inclusive_counts   = bigh_probefilter.integrate("selection", (["inclusive"])).values(sumw2=False)[()]
		if first:
			this_avg_eff_probefilter = copy.deepcopy(probefilter_counts)
			sum_inclusive_counts = copy.deepcopy(inclusive_counts)
		else:
			this_avg_eff_probefilter = this_avg_eff_probefilter + probefilter_counts
			sum_inclusive_counts = sum_inclusive_counts + inclusive_counts
		first = False
	avg_eff_probefilter[var] = this_avg_eff_probefilter / sum_inclusive_counts
	davg_eff_probefilter[var] = np.sqrt(avg_eff_probefilter[var] * (1. - avg_eff_probefilter[var]) / sum_inclusive_counts)


# Single-trigger efficiency
eff = {}
deff = {}

# - For Bd, include both matched and matched-swap
for var in vars:
	eff[var] = {}
	deff[var] = {}
	for btype in btypes:
		eff[var][btype] = {}
		deff[var][btype] = {}
		for side in ["probe", "tag"]:
			eff[var][btype][side] = {}
			deff[var][btype][side] = {}
			if side == "probe":
				eff[var][btype]["probe_total"] = {}
				deff[var][btype]["probe_total"] = {}

			for trigger in triggers:
				if side == "probe":
					h_truth_var = bigh_truth[var][btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
					h_reco_var = bigh_reco[var][btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
				elif side == "tag":
					h_truth_var = bigh_truth[var][btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])
					h_reco_var = bigh_reco[var][btype][side].integrate("dataset", [f"{btype_longnames[btype]}_probefilter"])

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

				eff[var][btype][side][trigger] = eff_recosel
				deff[var][btype][side][trigger] = deff_recosel

				if side == "probe":
					eff[var][btype]["probe_total"][trigger] = eff_recosel * avg_eff_probefilter[var]
					deff[var][btype]["probe_total"][trigger] = eff[var][btype]["probe_total"][trigger] * np.sqrt(
						(deff_recosel / eff_recosel)**2 + (davg_eff_probefilter[var] / avg_eff_probefilter[var])**2
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
for var in vars:
	for btype in btypes:
		for side in ["tag", "probe", "probe_total"]:
			# HLT_all strategy
			eff[var][btype][side]["HLT_all"] = 0.
			deff[var][btype][side]["HLT_all"] = 0.
			sumw = 0.
			for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
				weight = unique_lumis[trigger] / total_lumi
				eff[var][btype][side]["HLT_all"] += eff[var][btype][side][trigger] * weight
				deff[var][btype][side]["HLT_all"] += (deff[var][btype][side][trigger] * weight)**2
				sumw += weight
			eff[var][btype][side]["HLT_all"] = eff[var][btype][side]["HLT_all"] / sumw
			deff[var][btype][side]["HLT_all"] = np.sqrt(deff[var][btype][side]["HLT_all"]) / sumw

			# HLT_Mu9 strategy
			eff[var][btype][side]["HLT_Mu9"] = 0.
			deff[var][btype][side]["HLT_Mu9"] = 0.
			sumw = 0.
			for trigger in ["HLT_Mu9_IP5", "HLT_Mu9_IP6"]:
				if trigger == "HLT_Mu9_IP5":
					weight = total_lumis["HLT_Mu9_IP5"]  / total_lumis["HLT_Mu9_IP6"]
				elif trigger == "HLT_Mu9_IP6":
					weight = (total_lumis["HLT_Mu9_IP6"] - total_lumis["HLT_Mu9_IP5"]) / total_lumis["HLT_Mu9_IP6"]
				else:
					raise ValueError("Error!")
				eff[var][btype][side]["HLT_Mu9"] += eff[var][btype][side][trigger] * weight
				deff[var][btype][side]["HLT_Mu9"] += (deff[var][btype][side][trigger] * weight)**2
				sumw += weight
			eff[var][btype][side]["HLT_Mu9"] = eff[var][btype][side]["HLT_Mu9"] / sumw
			deff[var][btype][side]["HLT_Mu9"] = np.sqrt(deff[var][btype][side]["HLT_Mu9"]) / sumw

			# HLT_Mu7 strategy
			eff[var][btype][side]["HLT_Mu7"] = eff[var][btype][side]["HLT_Mu7_IP4"]
			deff[var][btype][side]["HLT_Mu7"] = deff[var][btype][side]["HLT_Mu7_IP4"]

# Turn into one dict
eff_deff = {}
for var in vars:
	eff_deff[var] = {}
	for btype in eff[var].keys():
		eff_deff[var][btype] = {}
		for side in eff[var][btype].keys():
			eff_deff[var][btype][side] = {}
			for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"] + ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
				eff_deff[var][btype][side][trigger_strategy] = []
				for i in range(len(eff[var][btype][side][trigger_strategy])):
					eff_deff[var][btype][side][trigger_strategy].append((eff[var][btype][side][trigger_strategy][i], deff[var][btype][side][trigger_strategy][i]))

with open("/home/dryu/BFrag/data/efficiency/efficiency2.pkl", "wb") as f:
	pickle.dump(eff_deff, f)


# Plots
for var in vars:
	for side in ["tag", "probe", "probe_total"]:
		bin_centers = bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).centers()
		xerrs = (bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).edges()[1:] - bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).edges()[:-1]) / 2
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
		for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7", "HLT_Mu9_IP5", "HLT_Mu9_IP6"]:
			fig, ax = plt.subplots(1, 1, figsize=(10,7))
			for btype in btypes:
				#pprint(eff[btype][side][trigger_strategy])
				ax.errorbar(
					x=bin_centers, 
					y=eff[var][btype][side][trigger_strategy], 
					xerr=xerrs,
					yerr=deff[var][btype][side][trigger_strategy], 
					marker=".", 
					markersize=10.,
					color=colors[btype],
					label=labels[btype],
					ls="none",
					ecolor=colors[btype],
					elinewidth=1
					)
			if side == "probe_total":
				ax.set_ylim(1.e-6, 5.e-3)
			elif side == "tag" or side == "probe":
				ax.set_ylim(1.e-5, 0.5)
			ax.set_yscale("log")
			if var == "pt":
				ax.set_xlim(0., 45.)
				ax.set_xlabel(r"$p_{T}$ [GeV]")
			elif var == "y":
				ax.set_xlim(0., 2.5)
				ax.set_xlabel(r"$|y|$")
			ax.set_ylabel("Total efficiency")
			ax.xaxis.set_ticks_position("both")
			ax.yaxis.set_ticks_position("both")
			ax.tick_params(direction="in")
			ax.legend()
			print(f"{figure_directory}/total_efficiency_{side}_{trigger_strategy}_{var}.png")
			fig.savefig(f"{figure_directory}/total_efficiency_{side}_{trigger_strategy}_{var}.png")

# Probefilter plot
for var in vars:
	fig, ax = plt.subplots(1, 1, figsize=(10,7))
	bin_centers = bigh_truth[var]["Bs"]["probe"].axis(var).centers()
	xerrs = (bigh_truth[var]["Bs"]["probe"].axis(var).edges()[1:] - bigh_truth[var]["Bs"]["probe"].axis(var).edges()[:-1]) / 2
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
	for btype in btypes:
		#pprint(eff[btype][side][trigger_strategy])
		ax.errorbar(
			x=bin_centers, 
			y=eff_probefilter[var][btype], 
			xerr=xerrs,
			yerr=deff_probefilter[var][btype], 
			marker=".", 
			markersize=10.,
			color=colors[btype],
			label=labels[btype],
			ls="none",
			ecolor=colors[btype],
			elinewidth=1,
			)
	ax.set_ylim(1.e-3, 1.e-1)
	ax.set_yscale("log")
	if var == "pt":
		ax.set_xlim(0., 45.)
		ax.set_xlabel(r"$p_{T}$ [GeV]")
	elif var == "y":
		ax.set_xlim(0., 2.5)
		ax.set_xlabel(r"$|y|$")
	ax.set_ylabel("probefilter efficiency")
	ax.xaxis.set_ticks_position("both")
	ax.yaxis.set_ticks_position("both")
	ax.tick_params(direction="in")
	ax.legend()
	print(f"{figure_directory}/probefilter_efficiency_{var}.png")
	fig.savefig(f"{figure_directory}/probefilter_efficiency_{var}.png")


# Trigger ratios
for var in vars:
	for side in ["tag", "probe"]:
		bin_centers = bigh_truth[var]["Bs"][side].axis(var).centers()
		xerrs = (bigh_truth[var]["Bs"][side].axis(var).edges()[1:] - bigh_truth[var]["Bs"][side].axis(var).edges()[:-1]) / 2
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
		plt.tight_layout()
		fig, ax = plt.subplots(2, 1, figsize=(10,12))
		plt.tight_layout()
		for trigger_strategy in ["HLT_Mu7_IP4", "HLT_Mu9_IP5"]:
			for btype in btypes:
				if trigger_strategy == "HLT_Mu7_IP4":
					marker_type = "o"
				else:
					marker_type = "s"
				#pprint(eff[btype][side][trigger_strategy])
				ax[0].errorbar(
					x=bin_centers, 
					y=eff[var][btype][side][trigger_strategy], 
					xerr=xerrs,
					yerr=deff[var][btype][side][trigger_strategy], 
					marker=marker_type, 
					markersize=10.,
					color=colors[btype],
					label="{}, {}".format(labels[btype], trigger_strategy),
					ls="none",
					ecolor=colors[btype],
					elinewidth=1
					)
		if side == "probe":
			ax[0].set_ylim(1.e-6, 5.e-3)
		elif side == "tag":
			ax[0].set_ylim(1.e-5, 0.5)
		ax[0].set_yscale("log")
		if var == "pt":
			ax[0].set_xlim(0., 45.)
			#ax[0].set_xlabel(r"$p_{T}$ [GeV]")
		elif var == "y":
			ax[0].set_xlim(0., 2.5)
			#ax[0].set_xlabel(r"$|y|$")
		ax[0].set_ylabel("Total efficiency")
		ax[0].xaxis.set_ticks_position("both")
		ax[0].yaxis.set_ticks_position("both")
		ax[0].tick_params(direction="in")
		ax[0].legend(prop={'size': 10})

		for btype in btypes:
			ax[1].errorbar(
					x=bin_centers, 
					y=eff[var][btype][side]["HLT_Mu9_IP5"] / eff[var][btype][side]["HLT_Mu7_IP4"], 
					xerr=xerrs,
					#yerr=deff[var][btype][side][trigger_strategy], 
					marker=".", 
					markersize=10.,
					color=colors[btype],
					#label="{}, {}".format(labels[btype]),
					ls="none",
					ecolor=colors[btype],
					elinewidth=1

			)
		ax[1].set_ylim(0., 1.0)
		ax[1].set_yscale("linear")
		if var == "pt":
			ax[1].set_xlim(0., 45.)
			ax[1].set_xlabel(r"$p_{T}$ [GeV]")
		elif var == "y":
			ax[1].set_xlim(0., 2.5)
			ax[1].set_xlabel(r"$|y|$")
		ax[1].set_ylabel("HLT_Mu9_IP5 / HLT_Mu7_IP4")
		ax[1].xaxis.set_ticks_position("both")
		ax[1].yaxis.set_ticks_position("both")
		ax[1].tick_params(direction="in")

		print(f"{figure_directory}/total_eff_trigratio_{side}_{var}.png")
		plt.tight_layout()
		fig.savefig(f"{figure_directory}/total_eff_trigratio_{side}_{var}.png")


# Trigger comparison
for var in vars:
	for side in ["tag", "probe", "probe_total"]:
		bin_centers = bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).centers()
		xerrs = (bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).edges()[1:] - bigh_truth[var]["Bs"][side.replace("probe_total", "probe")].axis(var).edges()[:-1]) / 2
		colors = {
			"HLT_Mu7_IP4": "green", 
			"HLT_Mu9_IP5": "purple", 
			"HLT_Mu9_IP6": "blue", 
			"HLT_Mu12_IP6": "red"
		}
		labels = {
			"Bd": r"$B^{0}$", 
			"Bu": r"$B^{\pm}$",
			"Bs": r"$B_{s}$"
		}
		markers = {
			"HLT_Mu7_IP4": "o", 
			"HLT_Mu9_IP5": "s", 
			"HLT_Mu9_IP6": "D", 
			"HLT_Mu12_IP6": "P",
			"HLT_all": "XYZ"
		}
		for btype in btypes:
			fig, ax = plt.subplots(2, 1, figsize=(10,12))
			for trigger_strategy in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
				#pprint(eff[btype][side][trigger_strategy])
				ax[0].errorbar(
					x=bin_centers, 
					y=eff[var][btype][side][trigger_strategy], 
					xerr=xerrs,
					yerr=deff[var][btype][side][trigger_strategy], 
					marker=markers[trigger_strategy], 
					markersize=10.,
					color=colors[trigger_strategy],
					label=trigger_strategy,
					ls="none",
					ecolor=colors[trigger_strategy],
					elinewidth=1
					)
			if side == "probe_total":
				ax[0].set_ylim(1.e-6, 5.e-3)
			elif side == "tag" or side == "probe":
				ax[0].set_ylim(1.e-5, 0.5)
			ax[0].set_yscale("log")
			if var == "pt":
				ax[0].set_xlim(0., 45.)
				ax[0].set_xlabel(r"$p_{T}$ [GeV]")
			elif var == "y":
				ax[0].set_xlim(0., 2.5)
				ax[0].set_xlabel(r"$|y|$")
			ax[0].set_ylabel("Total efficiency")
			ax[0].xaxis.set_ticks_position("both")
			ax[0].yaxis.set_ticks_position("both")
			ax[0].tick_params(direction="in")
			ax[0].legend()

			# Ratio
			for trigger_strategy in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
				ax[1].errorbar(
						x=bin_centers, 
						y=eff[var][btype][side][trigger_strategy] / eff[var][btype][side]["HLT_Mu7_IP4"], 
						xerr=xerrs,
						#yerr=deff[var][btype][side][trigger_strategy], 
						marker=markers[trigger_strategy], 
						markersize=10.,
						color=colors[trigger_strategy],
						#label="{}, {}".format(labels[btype]),
						ls="none",
						ecolor=colors[trigger_strategy],
						elinewidth=1

				)
			ax[1].set_ylim(0., 1.1)
			ax[1].set_yscale("linear")
			if var == "pt":
				ax[1].set_xlim(0., 45.)
				ax[1].set_xlabel(r"$p_{T}$ [GeV]")
			elif var == "y":
				ax[1].set_xlim(0., 2.5)
				ax[1].set_xlabel(r"$|y|$")
			ax[1].set_ylabel("Ratio to HLT_Mu7_IP4")
			ax[1].xaxis.set_ticks_position("both")
			ax[1].yaxis.set_ticks_position("both")
			ax[1].tick_params(direction="in")

			print(f"{figure_directory}/eff_trigcompare_{side}_{btype}_{var}.png")
			fig.savefig(f"{figure_directory}/eff_trigcompare_{side}_{btype}_{var}.png")
