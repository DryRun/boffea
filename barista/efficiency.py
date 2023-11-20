import os
import sys
import numpy as np 
from coffea import util, hist
import matplotlib.pyplot as plt
import math
from pprint import pprint
import pickle
import copy
from brazil.reweighting import reweight_trkpt, reweight_y, w_rap
from brazil.reweighting import reweight_had_pt, reweight_had_y, w_pt_had, w_rap_had

# Rebinning
axes = {}
axes["pt"] = {
	"probe": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([8., 13., 18., 23., 28., 33., 50.])), 
	"probe2": hist.Bin("pt", r"$p_{T}$ [GeV]", np.arange(10.0, 50.0+5.0, 5.0)), 
	"tag": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0, 60.0])), # 10.0, 
	#"tag": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([10.0, 13.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0])),
	#"probeMaxPt": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([8., 13., 18., 23., 28., 33.])), 
	#"tagMaxPt": hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0]))
}
axes["y"] = {
	"probe": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25))), 
	"probe2": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25))), 
	"tag": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 1.75+0.25, 0.25))),
	#"probeMaxPt": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25))), 
	#"tagMaxPt": hist.Bin("y", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.25)))
}
axes["pt"]["tagx"] = axes["pt"]["tag"]
axes["y"]["tagx"] = axes["y"]["tag"]

opposite_axis_cuts = {
	"pt": {
		"probe": [0., 2.25], 
		"probe2": [0., 2.25], 
		"tag": [0., 1.5],
	}, 
	"y": {
		"probe": [13.0, 50.0], 
		"probe2": [10.0, 50.0], 
		"tag": [13.0, 50.0],
	}
}
opposite_axis_cuts["pt"]["tagx"] = opposite_axis_cuts["pt"]["tag"]
opposite_axis_cuts["y"]["tagx"] = opposite_axis_cuts["y"]["tag"]


if __name__ == "__main__":
	import mplhep
	plt.style.use(mplhep.style.LHCb)
	plt.tight_layout()

	import argparse
	parser = argparse.ArgumentParser(description="Compute efficiencies")
	parser.add_argument("--selection", "-s", type=str, default="nominal", choices=["nominal", "HiTrkPt", "badlumi", "VarMuonPt", "MediumMuonPt", "HiMuonPt", "MediumMuonID"], help="Selection name (nominal or HiTrkPt)")
	parser.add_argument("--reweight", type=str, default="nominal", choices=["nominal", "pythia"], help="Reweight option: pythia, nominal")
	args = parser.parse_args()

	# reweight=nominal corresponds to:
	# - tag side: reweight using FONLL, implemented at the histogram level as *rwgt_pt or *rwgt_absy
	# - probe side: reweighting using hadronic analysis data/MC ratios

	if args.selection != "nominal" and args.reweight != "nominal":
		raise ValueError("Both --selection and --reweight are not nominal")

	# Configure I/O
	btypes = ["Bs", "Bd", "Bu"]
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
	coffea_files = {}
	for btype in btypes:
		coffea_files[btype] = util.load(f"{btype_longnames[btype]}/MCEfficiencyHistograms_{btype}.coffea")

	figure_directory = os.path.expandvars(f"$BDATA/efficiency/figures/{args.selection}")
	os.system(f"mkdir -pv {figure_directory}")

	efficiency_filename = os.path.expandvars(f"$BDATA/efficiency/efficiency_{args.selection}_{args.reweight}.pkl")
	#efficiency_filename = os.path.expandvars(f"$BDATA/efficiency/efficiency_{args.selection}.pkl")

	# Configure loops
	triggers = ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]
	vars = ["pt", "y"]

	if args.selection == "nominal" or args.selection == "badlumi": # , "rwgt_pt", "rwgt_absy"
		selection_str = ""
	elif args.selection == "HiTrkPt":
		selection_str = "HiTrkPt"
	elif "MuonPt" in args.selection:
		selection_str = args.selection
	elif args.selection == "MediumMuonID":
		selection_str = args.selection
	elif args.selection in ["VarMuonPt", "HiMuonPt", "MediumMuonPt"]:
		selection_str = args.selection
	else:
		raise ValueError("Unsupported --selection: {}".format(args.selection))

	# Start computation
	counts_reco = {}
	counts_truth = {}

	sides = ["tag", "probe", "tagx"]
	if args.selection in ["MediumMuonPt", "HiMuonPt", "MediumMuonID"]:
		sides = ["tag", "tagx"]

	for btype in btypes:
		counts_reco[btype] = {}
		counts_truth[btype] = {}
		for side in sides: # , "probeMaxPt", "tagMaxPt"
			counts_reco[btype][side] = {}
			counts_truth[btype][side] = {}

			# Reco counts
			for var in ["pt", "y"]:
				counts_reco[btype][side][var] = {}
				if side == "tag":
					if var == "pt":
						hname = f"{btype_shortnames[btype]}_fit_pt_absy_mass_rwgt_absy" # 
					elif var == "y":
						hname = f"{btype_shortnames[btype]}_fit_pt_absy_mass_rwgt_pt" # 
				else:
					hname = f"{btype_shortnames[btype]}_fit_pt_absy_mass"
				bigh_reco_2d = coffea_files[btype][hname].integrate("fit_mass")

				# Rebin primary axis
				if var == "pt":
					bigh_reco_2d = bigh_reco_2d.rebin("fit_pt", axes["pt"][side])
				elif var == "y":
					bigh_reco_2d = bigh_reco_2d.rebin("fit_absy", axes["y"][side])

				# Rebin secondary axis (mostly for reweighting)
				if var == "pt": 
					# 0.125 --> 0.25 bin width
					bigh_reco_2d = bigh_reco_2d.rebin("fit_absy", 2)
				elif var == "y":
					# For probe, match Taeun's binning for weights
					if side == "probe":
						bigh_reco_2d = bigh_reco_2d.rebin("fit_pt", hist.Bin("fit_pt", r"$p_{T}$ [GeV]", np.array([13., 18., 23., 28., 33., 50.])))

				for sparse_key in bigh_reco_2d.values(sumw2=False).keys():

					# Secondary axis: apply weights and cuts
					sum_axis = 1 if var == "pt" else 0

					counts_reco_2D = bigh_reco_2d.values(sumw2=False)[sparse_key]
					if side == "probe" and args.reweight == "nominal":
						if var == "pt":
							counts_reco_2D = counts_reco_2D * w_rap_had[btype]
						elif var == "y":
							counts_reco_2D = np.transpose(np.transpose(counts_reco_2D) * w_pt_had[btype][1:])

					# Cut
					secondary_axis = bigh_reco_2d.axis("fit_pt" if var == "y" else "fit_absy")
					secondary_edges = secondary_axis.edges()
					low_idx = np.argwhere(secondary_edges == opposite_axis_cuts[var][side][0])[0][0]
					high_idx = np.argwhere(secondary_edges == opposite_axis_cuts[var][side][1])[0][0]
					counts_reco_2D = counts_reco_2D.take(indices=range(low_idx, high_idx), axis=sum_axis)

					# Project out secondary axis
					counts_reco[btype][side][var][sparse_key] = np.sum(counts_reco_2D, axis=sum_axis)

					# Tack on the overflow bin for pt
					if var == "pt":
						overflow_count = bigh_reco_2d.values(sumw2=False, overflow='over')[sparse_key].take(indices=0, axis=sum_axis)[-1]
						counts_reco[btype][side][var][sparse_key] = np.append(counts_reco[btype][side][var][sparse_key], overflow_count)
						#np.sum(np.take(bigh_reco_2d.values(sumw2=False, overflow='over')[sparse_key], axis=sum_axis)

			# Truth counts
			for var in ["pt", "y"]:
				counts_truth[btype][side][var] = {}
				if side == "tag":
					if var == "pt":
						hname = f"Truth{btype_shortnames[btype]}_pt_absy_mass_rwgt_absy" # 
					elif var == "y":
						hname = f"Truth{btype_shortnames[btype]}_pt_absy_mass_rwgt_pt" # 
				else:
					hname = f"Truth{btype_shortnames[btype]}_pt_absy_mass"
				bigh_truth_2d = coffea_files[btype][hname].integrate("mass")

				# Rebin primary axis
				if var == "pt":
					bigh_truth_2d = bigh_truth_2d.rebin("pt", axes["pt"][side])
				elif var == "y":
					bigh_truth_2d = bigh_truth_2d.rebin("absy", axes["y"][side])

				# Rebin secondary axis (mostly for reweighting)
				if var == "pt": 
					# 0.125 --> 0.25 bin width
					bigh_truth_2d = bigh_truth_2d.rebin("absy", 2)
				elif var == "y":
					# For probe, match Taeun's binning for weights
					if side == "probe":
						bigh_truth_2d = bigh_truth_2d.rebin("pt", hist.Bin("pt", r"$p_{T}$ [GeV]", np.array([13., 18., 23., 28., 33., 50.])))

				for sparse_key in bigh_truth_2d.values(sumw2=False).keys():
					# Secondary axis: apply weights and cuts
					sum_axis = 1 if var == "pt" else 0

					counts_truth_2D = bigh_truth_2d.values(sumw2=False)[sparse_key]
					if side == "probe" and args.reweight == "nominal":
						if var == "pt":
							counts_truth_2D = counts_truth_2D * w_rap_had[btype]
						elif var == "y":
							#print("xyz counts_truth_2D shae:")
							#print(counts_truth_2D.shape)
							counts_truth_2D = np.transpose(np.transpose(counts_truth_2D) * w_pt_had[btype][1:])
							#print("xyz counts_truth_2D after reweighting:")
							#print(counts_truth_2D)

					# Cut secondary axis
					secondary_axis = bigh_truth_2d.axis("pt" if var == "y" else "absy")
					secondary_edges = secondary_axis.edges()
					low_idx = np.argwhere(secondary_edges == opposite_axis_cuts[var][side][0])[0][0]
					high_idx = np.argwhere(secondary_edges == opposite_axis_cuts[var][side][1])[0][0]

					#print("xyz Cutting secondary axis")
					#print("Original axis edges:")
					#print(secondary_edges)
					#print(f"Low index = {low_idx} / high index = {high_idx}")

					# Low/high indices refer to the edges array; the highest bin you want is high_idx-1 (since it's a right edge)
					counts_truth_2D = counts_truth_2D.take(indices=range(low_idx, high_idx), axis=sum_axis)

					counts_truth[btype][side][var][sparse_key] = np.sum(counts_truth_2D, axis=sum_axis)

					# Tack on the overflow bin for pt
					if var == "pt":
						counts_truth_2D_overpt = bigh_truth_2d.values(sumw2=False, overflow='over')[sparse_key]
						counts_truth_2D_overpt = counts_truth_2D_overpt[:,:-1]
						if args.reweight == "nominal": 
							counts_truth_2D_overpt = counts_truth_2D_overpt * w_rap_had[btype]
						counts_truth_2D_overpt = counts_truth_2D_overpt.take(indices=range(low_idx, high_idx), axis=sum_axis)
						overflow_count = np.sum(counts_truth_2D_overpt, axis=sum_axis)[-1]

						counts_truth[btype][side][var][sparse_key] = np.append(counts_truth[btype][side][var][sparse_key], overflow_count)

						# Consistency check
						#print("asdf consistency check")
						#print(np.sum(bigh_truth_2d.values(sumw2=False)[sparse_key], axis=sum_axis))
						#print(bigh_truth_2d.values(sumw2=False, overflow='over')[sparse_key].take(indices=0, axis=sum_axis))

		#bigh_reco["y"][btype] = {}
		#bigh_truth_2d["y"][btype] = {}
		#for side in ["probe", "tag"]:
		#	bigh_reco["y"][btype][side] = coffea_files[btype][f"{btype_shortnames[btype]}_fit_pt_absy_mass"].integrate("fit_mass").integrate("fit_pt", slice(10.0, 30.0)).rebin("fit_absy", axes["y"][side])
		#	bigh_truth_2d["y"][btype][side] = coffea_files[btype][f"Truth{btype_shortnames[btype]}_pt_absy_mass"].integrate("mass").integrate("pt", slice(10.0, 30.0)).rebin("absy", axes["y"][side])


	# Probefilter efficency, needed for probe side
	eff_probefilter = {}
	deff_probefilter = {}
	avg_eff_probefilter = {}
	davg_eff_probefilter = {}

	if "probe" in sides:
		for var in vars:
			eff_probefilter[var] = {}
			deff_probefilter[var] = {}
			this_avg_eff_probefilter = None 
			this_sum_inclusive_counts = None
			first = True

			for btype in btypes:
				#print(counts_truth[btype]["probe"].keys())
				probefilter_counts = counts_truth[btype]["probe"][var][(f"{btype_longnames[btype]}_inclusive", "probefilter_fiducial")]
				inclusive_counts = counts_truth[btype]["probe"][var][(f"{btype_longnames[btype]}_inclusive", "fiducial")]

				eff_probefilter[var][btype] = probefilter_counts / inclusive_counts
				deff_probefilter[var][btype] = np.sqrt(eff_probefilter[var][btype] * (1. - eff_probefilter[var][btype]) / inclusive_counts)

				if first:
					this_avg_eff_probefilter = copy.deepcopy(probefilter_counts)
					sum_inclusive_counts = copy.deepcopy(inclusive_counts)
				else:
					this_avg_eff_probefilter = this_avg_eff_probefilter + probefilter_counts
					sum_inclusive_counts = sum_inclusive_counts + inclusive_counts
				first = False
			avg_eff_probefilter[var] = this_avg_eff_probefilter / sum_inclusive_counts
			davg_eff_probefilter[var] = np.sqrt(avg_eff_probefilter[var] * (1. - avg_eff_probefilter[var]) / sum_inclusive_counts)


	# MuFilter efficency, needed for tag side
	eff_mufilter = {}
	deff_mufilter = {}
	avg_eff_mufilter = {}
	davg_eff_mufilter = {}
	for var in vars:
		eff_mufilter[var] = {}
		deff_mufilter[var] = {}
		this_avg_eff_mufilter = None 
		this_sum_inclusive_counts = None
		first = True

		for btype in btypes:
			#print(counts_truth[btype]["tag"].keys())
			mufilter_counts = counts_truth[btype]["tag"][var][(f"{btype_longnames[btype]}_inclusive", "mufilter_fiducial")]
			inclusive_counts = counts_truth[btype]["tag"][var][(f"{btype_longnames[btype]}_inclusive", "fiducial")]

			eff_mufilter[var][btype] = mufilter_counts / inclusive_counts
			deff_mufilter[var][btype] = np.sqrt(eff_mufilter[var][btype] * (1. - eff_mufilter[var][btype]) / inclusive_counts)

			if first:
				this_avg_eff_mufilter = copy.deepcopy(mufilter_counts)
				sum_inclusive_counts = copy.deepcopy(inclusive_counts)
			else:
				this_avg_eff_mufilter = this_avg_eff_mufilter + mufilter_counts
				sum_inclusive_counts = sum_inclusive_counts + inclusive_counts
			first = False
		avg_eff_mufilter[var] = this_avg_eff_mufilter / sum_inclusive_counts
		davg_eff_mufilter[var] = np.sqrt(avg_eff_mufilter[var] * (1. - avg_eff_mufilter[var]) / sum_inclusive_counts)


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
			for side in sides: # , "probeMaxPt", "tagMaxPt"
				eff[var][btype][side] = {}
				deff[var][btype][side] = {}

				eff[var][btype][f"{side}_total"] = {}
				deff[var][btype][f"{side}_total"] = {}

				for trigger in triggers:
					use_truth_side = True
					if use_truth_side:
						#print("asdf")
						#print(counts_truth[btype][side].keys())

						# Set the "sparse keys" for looking up numerator and denominator counts
						# Remember, the key is (dataset_tag, selection)
						# See mc_efficiency_processor.py for more info
						if side == "probe": 
							matched_key = (f"{btype_longnames[btype]}_probefilter", f"matched_fid_{side}{selection_str}_{trigger}")
							matched_swap_key = (f"{btype_longnames[btype]}_probefilter", f"matched_swap_fid_{side}{selection_str}_{trigger}")
							inclusive_key = (f"{btype_longnames[btype]}_probefilter", "fiducial")
						elif side == "tag" or side == "tagx":
							matched_key = (f"{btype_longnames[btype]}_mufilter", f"matched_fid_{side}{selection_str}_{trigger}")
							matched_swap_key = (f"{btype_longnames[btype]}_mufilter", f"matched_swap_fid_{side}{selection_str}_{trigger}")
							inclusive_key = (f"{btype_longnames[btype]}_mufilter", "fiducial")

						#print(counts_truth[btype][side.replace("MaxPt", "")].keys())
						#print("DEBUG : Available keys:")
						#print(counts_truth[btype][side.replace("MaxPt", "")][var].keys())
						counts_matched = counts_truth[btype][side.replace("MaxPt", "")][var][matched_key]
						if btype == "Bd":
							counts_matched += counts_truth[btype][side.replace("MaxPt", "")][var][matched_swap_key]
						counts_inclusive = counts_truth[btype][side.replace("MaxPt", "")][var][inclusive_key]
					else:
						print("Using reco side doesn't work for fiducial efficiency")
						sys.exit(1)
						# Set the "sparse keys" for looking up numerator and denominator counts
						# Remember, the key is (dataset_tag, selection)
						# See mc_efficiency_processor.py for more info
						'''
						if side == "probe": 
							matched_key = (f"{btype_longnames[btype]}_probefilter", f"{side}match_{trigger}")
							matched_swap_key = (f"{btype_longnames[btype]}_probefilter", f"{side}matchswap_{trigger}")
							inclusive_key = (f"{btype_longnames[btype]}_probefilter", "inclusive")
						elif side == "tag":
							matched_key = (f"{btype_longnames[btype]}_mufilter", f"{side}match_{trigger}")
							matched_swap_key = (f"{btype_longnames[btype]}_mufilter", f"{side}matchswap_{trigger}")
							inclusive_key = (f"{btype_longnames[btype]}_mufilter", "inclusive")

						counts_matched = counts_reco[btype][side][var][matched_key]
						if btype == "Bd":
							counts_matched += counts_reco[btype][side][var][matched_swap_key]
						counts_inclusive = counts_reco[btype][side][var][inclusive_key]
						'''

					eff_recosel = counts_matched / counts_inclusive
					deff_recosel = np.sqrt(eff_recosel * (1. - eff_recosel) / counts_inclusive)

					eff[var][btype][side][trigger] = copy.deepcopy(eff_recosel)
					deff[var][btype][side][trigger] = copy.deepcopy(deff_recosel)

					if side == "probe":
						eff[var][btype]["probe_total"][trigger] = eff_recosel * avg_eff_probefilter[var]
						deff[var][btype]["probe_total"][trigger] = eff[var][btype]["probe_total"][trigger] * np.sqrt(
							(deff_recosel / eff_recosel)**2 + (davg_eff_probefilter[var] / avg_eff_probefilter[var])**2
							)
					elif side == "tag":
						eff[var][btype]["tag_total"][trigger] = eff_recosel * eff_mufilter[var][btype]
						deff[var][btype]["tag_total"][trigger] = eff[var][btype]["tag_total"][trigger] * np.sqrt(
							(deff_recosel / eff_recosel)**2 + (deff_mufilter[var][btype] / eff_mufilter[var][btype])**2
							)

						# If eff=0, set deff=0 too, instead of nan
						for ii in range(len(deff[var][btype]["tag_total"][trigger])):
							if eff[var][btype]["tag_total"][trigger][ii] == 0:
								deff[var][btype]["tag_total"][trigger][ii] = 0.0
					elif side == "tagx":
						# For tagx, you can't use the mufilter efficiency. None of your samples is appropriate for vetoing a probe muon 
						# So, hack: just use 1.0, and remember you can't use the total...
						eff[var][btype]["tagx_total"][trigger] = eff_recosel
						deff[var][btype]["tagx_total"][trigger] = deff_recosel

						# If eff=0, set deff=0 too, instead of nan
						for ii in range(len(deff[var][btype]["tagx_total"][trigger])):
							if eff[var][btype]["tagx_total"][trigger][ii] == 0:
								deff[var][btype]["tagx_total"][trigger][ii] = 0.0


	# Efficiencies for final trigger strategies
	if args.selection == "badlumi":
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
	else:
		unique_lumis = {
			'HLT_Mu7_IP4' : 6939.748065235031,
			'HLT_Mu9_IP5' : 13950.9446944551,
			'HLT_Mu9_IP6' : 12686.483720985076,
			'HLT_Mu12_IP6': 8023.183781160029,
		}
		total_lumis = {
			'HLT_Mu7_IP4' : 6939.748065235031,
			'HLT_Mu9_IP5' : 20890.692759689857,
			'HLT_Mu9_IP6' : 33577.17648067498,
			'HLT_Mu12_IP6': 34698.77175553492,
		}
		total_lumi = 41600.36026183524
	
	for var in vars:
		for btype in btypes:
			for side in sides + [f"{x}_total" for x in sides]: # ["probe", "tag", "probe_total", "tag_total", "tagx", "tagx_total", "probe2"]: # , "probeMaxPt", "tagMaxPt"
				# HLT_all strategy
				eff[var][btype][side]["HLT_all"] = 0.
				deff[var][btype][side]["HLT_all"] = 0.
				sumw = 0.
				for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
					weight = unique_lumis[trigger] / total_lumi
					#print(f"{var} {btype}  {trigger} {side}")
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
						# Lumi where Mu9_IP5 is active = unique Mu9_IP5 plus unique Mu7_IP4, since these two were always nested
						weight = (unique_lumis["HLT_Mu9_IP5"] + unique_lumis["HLT_Mu7_IP4"]) / (unique_lumis["HLT_Mu9_IP6"] + unique_lumis["HLT_Mu9_IP5"] + unique_lumis["HLT_Mu7_IP4"])
					else:
						weight = (unique_lumis["HLT_Mu9_IP6"]) / (unique_lumis["HLT_Mu9_IP6"] + unique_lumis["HLT_Mu9_IP5"] + unique_lumis["HLT_Mu7_IP4"])
					#if trigger == "HLT_Mu9_IP5":
					#	weight = total_lumis["HLT_Mu9_IP5"]  / total_lumis["HLT_Mu9_IP6"]
					#elif trigger == "HLT_Mu9_IP6":
					#	weight = (total_lumis["HLT_Mu9_IP6"] - total_lumis["HLT_Mu9_IP5"]) / total_lumis["HLT_Mu9_IP6"]
					#else:
					#	raise ValueError("Error!")
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
				if side == "probe2_total":
					continue
				eff_deff[var][btype][side] = {}
				for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"] + ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
					eff_deff[var][btype][side][trigger_strategy] = []
					print(f"{var} / {btype} / {side} / {trigger_strategy}")
					for i in range(len(eff[var][btype][side][trigger_strategy])):
						eff_deff[var][btype][side][trigger_strategy].append((eff[var][btype][side][trigger_strategy][i], deff[var][btype][side][trigger_strategy][i]))

	with open(efficiency_filename, "wb") as f:
		pickle.dump(eff_deff, f)

	# Plots
	for var in vars:
		for side in sides + [f"{x}_total" for x in sides]:# ["tag", "probe", "tagx", "probe_total", "tag_total", "tagx_total"]:# "tagMaxPt", "probeMaxPt"
			# Get bin info from histograms
			#bigh_truth_2d = coffea_files["Bs"][f"Truth{btype_shortnames["Bs"]}_pt_absy_mass_rwgt"].integrate("mass")
			this_xaxis = axes[var][side.replace("_total", "")]
			bin_centers = this_xaxis.centers()
			xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
			if var == "pt":
				# Put overflow on plot
				bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
				xerrs = np.append(xerrs, 1.0)
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
				elif side == "tag" or side == "probe" or side == "tag_total" or side == "tagx" or side == "tagx_total":
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
				print(f"{figure_directory}/total_efficiency_{side}_{args.selection}_{trigger_strategy}_{var}.png")
				fig.savefig(f"{figure_directory}/total_efficiency_{side}_{args.selection}_{trigger_strategy}_{var}.png")
				fig.savefig(f"{figure_directory}/total_efficiency_{side}_{args.selection}_{trigger_strategy}_{var}.pdf")
				plt.close(fig)
				#fig.close()

	# Probefilter plot
	if "probe" in sides:
		print("DEBUG : making probefilter plot jkl")
		for var in vars:
			fig, ax = plt.subplots(1, 1, figsize=(10,7))
			this_xaxis = axes[var]["probe"]
			bin_centers = this_xaxis.centers()
			xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
			if var == "pt":
				# Put overflow on plot
				bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
				xerrs = np.append(xerrs, 1.0)
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
			fig.savefig(f"{figure_directory}/probefilter_efficiency_{var}.pdf")
			plt.close(fig)
			#fig.close()

	# MuFilter plot
	for var in vars:
		fig, ax = plt.subplots(1, 1, figsize=(10,7))
		this_xaxis = axes[var]["tag"]
		bin_centers = this_xaxis.centers()
		xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
		if var == "pt":
			# Put overflow on plot
			bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
			xerrs = np.append(xerrs, 1.0)
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
				y=eff_mufilter[var][btype], 
				xerr=xerrs,
				yerr=deff_mufilter[var][btype], 
				marker=".", 
				markersize=10.,
				color=colors[btype],
				label=labels[btype],
				ls="none",
				ecolor=colors[btype],
				elinewidth=1,
				)
		ax.set_ylim(0., 1.)
		#ax.set_yscale("log")
		if var == "pt":
			ax.set_xlim(0., 45.)
			ax.set_xlabel(r"$p_{T}$ [GeV]")
		elif var == "y":
			ax.set_xlim(0., 2.5)
			ax.set_xlabel(r"$|y|$")
		ax.set_ylabel("MuFilter efficiency")
		ax.xaxis.set_ticks_position("both")
		ax.yaxis.set_ticks_position("both")
		ax.tick_params(direction="in")
		ax.legend()
		print(f"{figure_directory}/mufilter_efficiency_{var}.png")
		fig.savefig(f"{figure_directory}/mufilter_efficiency_{var}.png")
		fig.savefig(f"{figure_directory}/mufilter_efficiency_{var}.pdf")
		plt.close(fig)
		#fig.close()


	# Trigger ratios
	for var in vars:
		for side in sides:
			this_xaxis = axes[var][side]
			bin_centers = this_xaxis.centers()
			xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
			if var == "pt":
				# Put overflow on plot
				bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
				xerrs = np.append(xerrs, 1.0)
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
			elif side == "tag" or side == "tagx":
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

			print(f"{figure_directory}/total_eff_trigratio_{side}_{args.selection}_{var}.png")
			plt.tight_layout()
			fig.savefig(f"{figure_directory}/total_eff_trigratio_{side}_{args.selection}_{var}.png")
			fig.savefig(f"{figure_directory}/total_eff_trigratio_{side}_{args.selection}_{var}.pdf")
			plt.close(fig)
			#fig.close()


	# Trigger comparison
	for var in vars:
		for side in sides + [f"{x}_total" for x in sides]:# ["tag", "probe", "tagx", "probe_total", "tag_total", "tagx_total"]:
			this_xaxis = axes[var][side.replace("_total", "")]
			bin_centers = this_xaxis.centers()
			xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
			if var == "pt":
				# Put overflow on plot
				bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
				xerrs = np.append(xerrs, 1.0)
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
				elif side == "tag" or side == "probe" or side == "tagx":
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

				print(f"{figure_directory}/eff_trigcompare_{side}_{args.selection}_{btype}_{var}.png")
				fig.savefig(f"{figure_directory}/eff_trigcompare_{side}_{args.selection}_{btype}_{var}.png")
				fig.savefig(f"{figure_directory}/eff_trigcompare_{side}_{args.selection}_{btype}_{var}.pdf")
				plt.close(fig)


	# Channel ratios
	print("xyz making channel ratio plots")
	for var in vars:
		for side in sides: #["tag", "probe", "tagx"]:
			this_xaxis = axes[var][side]
			bin_centers = this_xaxis.centers()
			xerrs = (this_xaxis.edges()[1:] - this_xaxis.edges()[:-1]) / 2
			if var == "pt":
				# Put overflow on plot
				bin_centers = np.append(bin_centers, this_xaxis.edges()[-1] + 1.0)
				xerrs = np.append(xerrs, 1.0)
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
			for trigger_strategy in ["HLT_Mu7_IP4", "HLT_all"]:
				plt.tight_layout()
				fig, ax = plt.subplots(2, 1, figsize=(10,12))
				plt.tight_layout()
				# Top: plot all btypes
				for btype in btypes:
					ax[0].errorbar(
						x=bin_centers, 
						y=eff[var][btype][side][trigger_strategy], 
						xerr=xerrs,
						yerr=deff[var][btype][side][trigger_strategy], 
						marker="o", 
						markersize=10.,
						color=colors[btype],
						label="{}, {}".format(labels[btype], trigger_strategy),
						ls="none",
						ecolor=colors[btype],
						elinewidth=1
						)
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
				plt.tight_layout()

				# Bottom: ratios between btypes
				ratios = {}
				dratios = {}
				colors_ratio = {(0, 1): "purple", (0, 2): "orange", (1, 2): "olive"}
				for ii in range(len(btypes)):
					for jj in range(ii+1, len(btypes)):
						btype1 = btypes[ii]
						btype2 = btypes[jj]
						ratios[(ii, jj)] = eff[var][btype1][side][trigger_strategy] / eff[var][btype2][side][trigger_strategy]
						dratios[(ii, jj)] = ratios[(ii, jj)] * ((deff[var][btype1][side][trigger_strategy] / eff[var][btype1][side][trigger_strategy])**2 + (deff[var][btype2][side][trigger_strategy] / eff[var][btype2][side][trigger_strategy])**2)**0.5 

						ax[1].errorbar(
							x=bin_centers, 
							y=ratios[(ii, jj)], 
							xerr=xerrs,
							yerr=dratios[(ii, jj)],
							marker="s", 
							markersize=10.,
							color=colors_ratio[(ii, jj)],
							label="{}/{}".format(labels[btype1], labels[btype2]),
							ls="none",
							ecolor=colors_ratio[(ii, jj)],
							elinewidth=1
							)
				ax[1].set_ylim(0., 2.0)
				ax[1].set_yscale("linear")
				if var == "pt":
					ax[1].set_xlim(0., 45.)
					ax[1].set_xlabel(r"$p_{T}$ [GeV]")
				elif var == "y":
					ax[1].set_xlim(0., 2.5)
					ax[1].set_xlabel(r"$|y|$")
				ax[1].set_ylabel("Eff. ratio")
				ax[1].xaxis.set_ticks_position("both")
				ax[1].yaxis.set_ticks_position("both")
				ax[1].tick_params(direction="in")
				ax[1].legend(prop={'size': 10})
				plt.tight_layout()

			print(f"{figure_directory}/channelratios_{side}_{args.selection}_{var}.png")
			plt.tight_layout()
			fig.savefig(f"{figure_directory}/channelratios_{side}_{args.selection}_{var}.png")
			fig.savefig(f"{figure_directory}/channelratios_{side}_{args.selection}_{var}.pdf")
			plt.close(fig)
			#fig.close()
