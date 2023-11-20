import os
import sys
import re
import math
import pickle
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats as stats
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
from brazil.fonll import *

from brazil.seaborn_colors import SeabornColors
seaborn_colors = SeabornColors()
palette_dir = os.path.expandvars("/home/dyu7/BFrag/boffea/brazil/seaborn_palettes")
seaborn_colors.load_palette("Blues_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Reds_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Oranges_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Greens_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Purples_d", palette_dir=palette_dir)
seaborn_colors.load_palette("RdPu_r", palette_dir=palette_dir)
seaborn_colors.load_palette("hls", palette_dir=palette_dir)
seaborn_colors.load_palette("hls_light", palette_dir=palette_dir)

from brazil.aguapreta import BR_phiKK, BR_KstarKpi, BR_BuKJpsi, BR_BdKstarJpsi, BR_BsPhiJpsi
#BR_phiKK = 0.492
#BR_KstarKpi = 0.665
#BR_BuKJpsi = 1.020e-3
#BR_BdKstarJpsi = 1.27e-3
#BR_BsPhiJpsi = 1.08e-3

sys.path.append("/home/dyu7/BFrag/boffitting/barista/fitting")
from fit_settings import *

legend_entries = {
	"Bu": "B^{#pm}#rightarrow K^{#pm} J/#psi(#mu^{+}#mu^{-})",
	"Bd": "B^{0}#rightarrow K^{*}J/#psi#rightarrow K^{#pm}#pi^{#mp}#mu^{+}#mu^{-}",
	"Bs": "B^{0}_{s}#rightarrow #phi J/#psi #rightarrow K^{+}K^{-}#mu^{+}#mu^{-}",
}
style = {
	"Bu":{
		"marker_style": 20, 
		"marker_color": seaborn_colors.get_root_color("Blues_d", 2),
		"line_color": seaborn_colors.get_root_color("Blues_d", 3),
		"line_width": 1,
	},
	"Bd":{
		"marker_style": 20, 
		"marker_color": seaborn_colors.get_root_color("Greens_d", 2),
		"line_color": seaborn_colors.get_root_color("Greens_d", 3),
		"line_width": 1,
	},
	"Bs":{
		"marker_style": 20, 
		"marker_color": seaborn_colors.get_root_color("Oranges_d", 2),
		"line_color": seaborn_colors.get_root_color("Oranges_d", 3),
		"line_width": 1,
	}
}
vars_pretty = {
	"pt": "$\\pt$", 
	"y": "$|y|$", 
	"absy": "$|y|$",
}

def style_graph(graph, btype):
	graph.SetMarkerStyle(style[btype]["marker_style"])
	graph.SetMarkerColor(style[btype]["marker_color"])
	graph.SetLineColor(style[btype]["line_color"])
	graph.SetLineWidth(style[btype]["line_width"])

def render_val_err(val, err, digits=2):
	if (val <= 0.) or np.isinf(val) or np.isnan(val):
		return f"${val:.{digits}e} \\pm {err:.{digits}e}$"

	else:
		exponent = int(math.floor(math.log(val) / math.log(10)))
		val_mantissa = val / 10**exponent
		err_mantissa = err / 10**exponent
		return f"$({val_mantissa:.{digits}f} \\pm {err_mantissa:.{digits}f}) \\times 10^{{{exponent}}}$"

class FFRData:
	def __init__(self, axis="pt", side="probe", trigger_strategy="HLT_all", binned=False, fitfunc="", selection=""):
		if not fitfunc in ["johnson", "3gauss", "poly"]:
			raise ValueError("fitfunc != johnson or 3gauss or poly")
		if not selection in ["nominal", "HiTrkPt", "HiMuonPt", "MediumMuonPt", "HugeMuonPt", "VarMuonPt", "MediumMuonID"]:
			raise ValueError("selection != nominal or HiTrkPt or HiMuonPt etc")

		self._axis = axis
		self._side = side
		self._binned = binned
		self._trigger_strategy = trigger_strategy
		self._fitfunc = fitfunc
		self._selection = selection
		self._other_ffrs = None
		self._yield_uncertainties = None

	def load_yields(self):
		self._yields = {}
		self._dyields = {}
		yield_files = {
			"Bs": {
				"binned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bs/yields_{self._fitfunc}_{self._selection}_binned.pkl",
				"unbinned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bs/yields_{self._fitfunc}_{self._selection}.pkl",
			},
			"Bu": {
				"binned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bu/yields_{self._fitfunc}_{self._selection}_binned.pkl",
				"unbinned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bu/yields_{self._fitfunc}_{self._selection}.pkl",
			},
			"Bd": {
				"binned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bd/yields_{self._fitfunc}_{self._selection}_binned.pkl",
				"unbinned": f"/home/dyu7/BFrag/boffitting/barista/fitting/Bd/yields_{self._fitfunc}_{self._selection}.pkl",
			},
		}

		yields_tmp = {}
		cuts_tmp = []
		for btype in ["Bu", "Bd", "Bs"]:
			if self._binned:
				yield_file = yield_files[btype]["binned"]
			else:
				yield_file = yield_files[btype]["unbinned"]
			print(f"Loading yields from {yield_file}")
			with open(yield_file, "rb") as f:
				print(yield_file)
				stuff = pickle.load(f)
				print(stuff[self._side].keys())
				yields_tmp[btype] = stuff[self._side][self._trigger_strategy]
			cuts_tmp.extend(yields_tmp[btype].keys())
		cuts_tmp = list(set(cuts_tmp))
		if self._axis == "pt":
			cuts_tmp = [x for x in cuts_tmp if "ptbin" in x] # Remove overflow
		elif self._axis == "y":
			cuts_tmp = [x for x in cuts_tmp if "ybin" in x]
		print(cuts_tmp)
		print(cut_xvals)
		cuts_tmp.sort(key=lambda x: cut_xvals[x][0])
		print("cuts_tmp:")
		print(cuts_tmp)
		# Make numpy arrays: x bin boundaries, y=yields
		self._xbins = np.array(sorted(list(set([cut_xvals[x][0] for x in cuts_tmp] + [cut_xvals[x][1] for x in cuts_tmp]))))
		if self._axis == "pt" and self._xbins[-1] > 61.0:
			# Set overflow max by hand, just for plotting
			if self._side == "tag" or self._side == "tagx":
				self._xbins[-1] = 75.0
			elif self._side == "probe":
				self._xbins[-1] = 70.0

		# DEBUG
		for btype in ["Bu", "Bd", "Bs"]:
			print(btype)
			print(yields_tmp[btype].keys())
		for btype in ["Bu", "Bd", "Bs"]:
			self._yields[btype] = np.zeros(len(self._xbins)-1)
			self._dyields[btype] = np.zeros(len(self._xbins)-1)
			for i, cut_name in enumerate(cuts_tmp):
				self._yields[btype][i]  = yields_tmp[btype][cut_name][0]
				self._dyields[btype][i] = yields_tmp[btype][cut_name][1]

		#print("xbins:")
		#print(self._xbins)

		#print("Yields:")
		#pprint(self._yields)

	def load_efficiencies(self):
		with open(f"/home/dyu7/BFrag/data/efficiency/efficiency_{self._selection}_nominal.pkl", "rb") as f:
			eff_tmp = pickle.load(f) # eff_deff[btype][side][trigger_strategy]
		self._efficiencies = {}
		self._defficiencies = {}

		self._total_efficiencies = {}
		self._dtotal_efficiencies = {}

		for btype in ["Bu", "Bd", "Bs"]:
			self._efficiencies[btype] = np.array([x[0] for x in eff_tmp[self._axis][btype][self._side][self._trigger_strategy]])
			self._defficiencies[btype] = np.array([x[1] for x in eff_tmp[self._axis][btype][self._side][self._trigger_strategy]])
			self._total_efficiencies[btype] = np.array([x[0] for x in eff_tmp[self._axis][btype][f"{self._side}_total"][self._trigger_strategy]])
			self._dtotal_efficiencies[btype] = np.array([x[1] for x in eff_tmp[self._axis][btype][f"{self._side}_total"][self._trigger_strategy]])
		#print("Efficiencies:")
		#pprint(self._efficiencies)

	def finalize(self):
		# Check that uncertainties were added only as other_ffrs or yield_uncertainties
		if self._other_ffrs and self._yield_uncertainties:
			raise ValueError("Uncertainties should be provide as either other_ffrs or yield_uncertainties, not both.")
		#if not (self._other_ffrs or self._yield_uncertainties):
		#	raise ValueError("Uncertainties should be provide as either other_ffrs or yield_uncertainties.")


		self._n = {} # Yields corrected up to MC filters (nominal default)
		self._dn = {} # Total uncertainty
		self._total_n = {} # Yields corrected up to inclusive MC (not nominal, MC filters cancel in ratios)
		self._dtotal_n = {}
		
		self._dn_src = {"stat": {}, "mcstat": {}} 
		if self._yield_uncertainties:
			self._dn_src = {"stat": {}, "mcstat": {}}
			self._dtotal_n_src = {"stat": {}, "mcstat": {}}
			self._yield_unc_btype_corr["stat"] = False
			self._yield_unc_btype_corr["mcstat"] = False

			for unc_name in self._yield_uncertainties:
				self._dn_src[unc_name] = {}
				self._dtotal_n_src[unc_name] = {}

		elif self._other_ffrs:
			self._dn_src = {"stat": {}, "mcstat": {}} # Uncertainties broken down by component
			self._dn_src["fitmodel"] = {}
			self._dtotal_n_src = {"stat": {}, "mcstat": {}}
			self._dtotal_n_src["fitmodel"] = {}

		else:
			# Only statistical uncertainties
			self._dn_src = {"stat": {}, "mcstat": {}} # Uncertainties broken down by component
			self._dtotal_n_src = {"stat": {}, "mcstat": {}}
			self._yield_unc_btype_corr = {}
			self._yield_unc_btype_corr["stat"] = False
			self._yield_unc_btype_corr["mcstat"] = False

		for btype in ["Bu", "Bd", "Bs"]:
			print(btype)
			self._n[btype] = self._yields[btype] / self._efficiencies[btype]
			self._dn_src["stat"][btype] = self._dyields[btype] / self._efficiencies[btype]
			self._dn_src["mcstat"][btype] = self._defficiencies[btype] * self._n[btype] / self._efficiencies[btype]

			# Add uncertainties provided as entire other FFRs
			if self._other_ffrs:
				# Compute dYield values
				npt = len(self._xbins)-1
				self._dn_src["fitmodel"][btype] = np.zeros(npt)

				# Could probably numpy this
				for ipt in range(npt):
					self._dn_src["fitmodel"][btype][ipt] = max([abs(self.Yield(btype)[ipt] - other.Yield(btype)[ipt]) for other in self._other_ffrs]) / self._efficiencies[btype][ipt]

			elif self._yield_uncertainties:
				for unc_name in self._yield_uncertainties:
					if unc_name == "mcstat":
						raise ValueError("Don't include mcstat in yield_uncertainties. This is added automatically.")

					print(f"xbins: {len(self._xbins)}")
					print(self._xbins)
					print(unc_name)
					print(btype)
					self._dn_src[unc_name][btype] = self._n[btype] * self._yield_uncertainties[unc_name][btype]


			# Total uncertainty
			dn2 = np.zeros_like(self._n[btype])
			for unc_src in self._dn_src.keys():
				dn2 += np.power(self._dn_src[unc_src][btype], 2)
			self._dn[btype] = np.power(dn2, 0.5)


			# Total N (i.e. correct for MC filter efficiencies)
			self._total_n[btype] = self._yields[btype] / self._total_efficiencies[btype]
			self._dtotal_n_src["stat"][btype] = self._dyields[btype] / self._total_efficiencies[btype]
			self._dtotal_n_src["mcstat"][btype] = self._dtotal_efficiencies[btype] * self._total_n[btype] / self._total_efficiencies[btype]

			if self._other_ffrs:
				self._dtotal_n_src["fitmodel"] = {}
				# Compute dYield values
				npt = len(self._xbins)-1
				self._dtotal_n_src["fitmodel"][btype] = np.zeros(npt)

				# Could probably numpy this
				for ipt in range(npt):
					self._dtotal_n_src["fitmodel"][btype][ipt] = max([abs(self.Yield(btype)[ipt] - other.Yield(btype)[ipt]) for other in self._other_ffrs]) / self._total_efficiencies[btype][ipt]
			elif self._yield_uncertainties:
				for unc_name in self._yield_uncertainties:
					if unc_name == "mcstat":
						raise ValueError("Don't include mcstat in yield_uncertainties. This is added automatically.")
					self._dtotal_n_src[unc_name][btype] = self._total_n[btype] * self._yield_uncertainties[unc_name][btype]


			dtotal_n2 = np.zeros_like(self._total_n[btype])
			for unc_src in self._dtotal_n_src.keys():
				dtotal_n2 += np.power(self._dtotal_n_src[unc_src][btype], 2)
			self._dtotal_n[btype] = np.power(dtotal_n2, 0.5)

			#self._total_n[btype] * np.sqrt(
			#	(self._dyields[btype] / self._yields[btype])**2
			#	+ (self._dtotal_efficiencies[btype] / self._total_efficiencies[btype])**2)

		#BR_phiKK = 0.492
		#BR_KstarKpi = 0.665
		#BR_BuKJpsi = 1.020e-3
		#BR_BdKstarJpsi = 1.27e-3
		#BR_BsPhiJpsi = 1.08e-3

		self._use_total_n = False #(self._side == "tag")

		if self._use_total_n:
			self._Rsu = self._total_n["Bs"] / self._total_n["Bu"] # Old: divided by (/ BR_phiKK / 2.0), which is wrong!
			self._dRsu = self._Rsu * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)
			self._dRsu_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRsu_src[unc_src] = self._Rsu * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dRsu_src[unc_src] = self._Rsu * np.sqrt((self._dtotal_n_src[unc_src]["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n_src[unc_src]["Bu"] / self._total_n["Bu"])**2)

			self._Rsd = self._total_n["Bs"] / self._total_n["Bd"] #/ BR_phiKK * BR_KstarKpi / 2.0
			self._dRsd = self._Rsd * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bd"] / self._total_n["Bd"])**2)
			self._dRsd_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRsd_src[unc_src] = self._Rsd * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bd"]) - 1.0)
				else:
					self._dRsd_src[unc_src] = self._Rsd * np.sqrt((self._dtotal_n_src[unc_src]["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n_src[unc_src]["Bd"] / self._total_n["Bd"])**2)

			self._Rdu = (self._total_n["Bd"]) / self._total_n["Bu"] #/ BR_KstarKpi / 2.0
			self._dRdu = self._Rdu * np.sqrt((self._dtotal_n["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)
			self._dRdu_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRdu_src[unc_src] = self._Rdu * ((1.0 + self._yield_uncertainties[unc_src]["Bd"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dRdu_src[unc_src] = self._Rdu * np.sqrt((self._dtotal_n_src[unc_src]["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n_src[unc_src]["Bu"] / self._total_n["Bu"])**2)

			self._fsfu = self._total_n["Bs"] / self._total_n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
			self._dfsfu = self._fsfu * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)
			self._dfsfu_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfsfu_src[unc_src] = self._fsfu * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dfsfu_src[unc_src] = self._fsfu * np.sqrt((self._dtotal_n_src[unc_src]["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n_src[unc_src]["Bu"] / self._total_n["Bu"])**2)

			self._fsfd = self._total_n["Bs"] / self._total_n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
			self._dfsfd = self._fsfd * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bd"] / self._total_n["Bd"])**2)
			self._dfsfd_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfsfd_src[unc_src] = self._fsfd * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bd"]) - 1.0)
				else:
					self._dfsfd_src[unc_src] = self._fsfd * np.sqrt((self._dtotal_n_src[unc_src]["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n_src[unc_src]["Bd"] / self._total_n["Bd"])**2)

			self._fdfu = self._total_n["Bd"] / self._total_n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
			self._dfdfu = self._fdfu * np.sqrt((self._dtotal_n["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)
			self._dfdfu_src = {}
			for unc_src in self._dtotal_n_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfdfu_src[unc_src] = self._fdfu * ((1.0 + self._yield_uncertainties[unc_src]["Bd"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dfdfu_src[unc_src] = self._fdfu * np.sqrt((self._dtotal_n_src[unc_src]["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n_src[unc_src]["Bu"] / self._total_n["Bu"])**2)

			# Compute averages
			self._sum_n = {}
			self._dsum_n = {}
			for btype in ["Bu", "Bd", "Bs"]:
				self._sum_n[btype] = self._total_n[btype].sum()
				self._dsum_n[btype] = math.sqrt((self._dn[btype] * self._dn[btype]).sum())

			self._avg_Rsu = (self._sum_n["Bs"]) / self._sum_n["Bu"] #/ BR_phiKK / 2.0
			self._avg_dRsu = self._avg_Rsu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_Rsd = (self._sum_n["Bs"]) / (self._sum_n["Bd"]) #/ BR_phiKK * BR_KstarKpi / 2.0
			self._avg_dRsd = self._avg_Rsd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2)

			self._avg_Rdu = (self._sum_n["Bd"]) / self._sum_n["Bu"] #/ BR_KstarKpi / 2.0
			self._avg_dRdu = self._avg_Rdu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_fsfu = self._sum_n["Bs"] / self._sum_n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
			self._avg_dfsfu = self._avg_fsfu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_fsfd = self._sum_n["Bs"] / self._sum_n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
			self._avg_dfsfd = self._avg_fsfd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2)

			self._avg_fdfu = self._sum_n["Bd"] / self._sum_n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
			self._avg_dfdfu = self._avg_fdfu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)
		else:
			self._Rsu = (self._n["Bs"]) / self._n["Bu"] #/ BR_phiKK / 2.0
			self._dRsu = self._Rsu * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)
			self._dRsu_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRsu_src[unc_src] = self._Rsu * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dRsu_src[unc_src] = self._Rsu * np.sqrt((self._dn_src[unc_src]["Bs"] / self._n["Bs"])**2 + (self._dn_src[unc_src]["Bu"] / self._n["Bu"])**2)

			self._Rsd = (self._n["Bs"]) / (self._n["Bd"]) #/ BR_phiKK * BR_KstarKpi / 2.0
			self._dRsd = self._Rsd * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bd"] / self._n["Bd"])**2)
			self._dRsd_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRsd_src[unc_src] = self._Rsd * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bd"]) - 1.0)
				else:
					self._dRsd_src[unc_src] = self._Rsd * np.sqrt((self._dn_src[unc_src]["Bs"] / self._n["Bs"])**2 + (self._dn_src[unc_src]["Bd"] / self._n["Bd"])**2)

			self._Rdu = (self._n["Bd"]) / self._n["Bu"]# / BR_KstarKpi / 2.0
			self._dRdu = self._Rdu * np.sqrt((self._dn["Bd"] / self._n["Bd"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)
			self._dRdu_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dRdu_src[unc_src] = self._Rdu * ((1.0 + self._yield_uncertainties[unc_src]["Bd"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dRdu_src[unc_src] = self._Rdu * np.sqrt((self._dn_src[unc_src]["Bd"] / self._n["Bd"])**2 + (self._dn_src[unc_src]["Bu"] / self._n["Bu"])**2)

			self._fsfu = self._n["Bs"] / self._n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
			self._dfsfu = self._fsfu * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)
			self._dfsfu_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfsfu_src[unc_src] = self._fsfu * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dfsfu_src[unc_src] = self._fsfu * np.sqrt((self._dn_src[unc_src]["Bs"] / self._n["Bs"])**2 + (self._dn_src[unc_src]["Bu"] / self._n["Bu"])**2)

			self._fsfd = self._n["Bs"] / self._n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
			self._dfsfd = self._fsfd * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bd"] / self._n["Bd"])**2)
			self._dfsfd_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfsfd_src[unc_src] = self._fsfd * ((1.0 + self._yield_uncertainties[unc_src]["Bs"]) / (1.0 + self._yield_uncertainties[unc_src]["Bd"]) - 1.0)
				else:
					self._dfsfd_src[unc_src] = self._fsfd * np.sqrt((self._dn_src[unc_src]["Bs"] / self._n["Bs"])**2 + (self._dn_src[unc_src]["Bd"] / self._n["Bd"])**2)

			self._fdfu = self._n["Bd"] / self._n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
			self._dfdfu = self._fdfu * np.sqrt((self._dn["Bd"] / self._n["Bd"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)
			self._dfdfu_src = {}
			for unc_src in self._dn_src.keys():
				if self._yield_unc_btype_corr[unc_src]:
					self._dfdfu_src[unc_src] = self._fdfu * ((1.0 + self._yield_uncertainties[unc_src]["Bd"]) / (1.0 + self._yield_uncertainties[unc_src]["Bu"]) - 1.0)
				else:
					self._dfdfu_src[unc_src] = self._fdfu * np.sqrt((self._dn_src[unc_src]["Bd"] / self._n["Bd"])**2 + (self._dn_src[unc_src]["Bu"] / self._n["Bu"])**2)

			# Compute averages
			self._sum_n = {}
			self._dsum_n = {}
			for btype in ["Bu", "Bd", "Bs"]:
				self._sum_n[btype] = self._n[btype].sum()
				self._dsum_n[btype] = math.sqrt((self._dn[btype] * self._dn[btype]).sum())

			self._avg_Rsu = (self._sum_n["Bs"]) / self._sum_n["Bu"]
			self._avg_dRsu = self._avg_Rsu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_Rsd = (self._sum_n["Bs"]) / (self._sum_n["Bd"])
			self._avg_dRsd = self._avg_Rsd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2)

			self._avg_Rdu = (self._sum_n["Bd"]) / self._sum_n["Bu"]
			self._avg_dRdu = self._avg_Rdu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_fsfu = self._sum_n["Bs"] / self._sum_n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
			self._avg_dfsfu = self._avg_fsfu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			self._avg_fsfd = self._sum_n["Bs"] / self._sum_n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
			self._avg_dfsfd = self._avg_fsfd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2)

			self._avg_fdfu = self._sum_n["Bd"] / self._sum_n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
			self._avg_dfdfu = self._avg_fdfu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

			print("qwert")
			print(self._dRsu_src)

	def print_yield_table_2flav(self, what):
		if what == "fdfu":
			left_btype = "Bd"
			right_btype = "Bu"
			what_value = self._fdfu
			what_dvalue = self._dfdfu
		elif what == "fsfu":
			left_btype = "Bs"
			right_btype = "Bu"
			what_value = self._fsfu
			what_dvalue = self._dfsfu
		elif what == "fsfd":
			left_btype = "Bs"
			right_btype = "Bd"
			what_value = self._fsfd
			what_dvalue = self._dfsfd
		if what == "Rdu":
			left_btype = "Bd"
			right_btype = "Bu"
			what_value = self._Rdu
			what_dvalue = self._dRdu
		elif what == "Rsu":
			left_btype = "Bs"
			right_btype = "Bu"
			what_value = self._Rsu
			what_dvalue = self._dRsu
		elif what == "Rsd":
			left_btype = "Bs"
			right_btype = "Bd"
			what_value = self._Rsd
			what_dvalue = self._dRsd

		table = f"""
\\begin{{table}}\n
	\\begin{{tabular}}{{|c|c|c|c|c|c|}}\n
		\\hline\n
		Bin & $\\{left_btype}$ raw yield & $\\{left_btype}$ eff. & $\\{left_btype}$ total yield & \\{right_btype}$ raw yield & $\\{right_btype}$ eff. & $\\{right_btype}$ total yield & {what} \\\\\n 
		\\hline\n"""

		for iline in range(len(self._xbins)-1):
			table += f"""\t\t{self._xbins[iline]:.1f}--{self._xbins[iline+1]:.1f} \
& ${self._yields[left_btype][iline]:.2f} \\pm {self._dyields[left_btype][iline]:.2f}$ \
& ${self._total_efficiencies[left_btype][iline]:.2e} \\pm {self._dtotal_efficiencies[left_btype][iline]:.2e}$ \
& ${self._total_n[left_btype][iline]:.2f} \\pm {self._dtotal_n[left_btype][iline]:.2f}$ \
& ${self._yields[right_btype][iline]:.2f} \\pm {self._dyields[right_btype][iline]:.2f}$ \
& ${self._total_efficiencies[right_btype][iline]:.2e} \\pm {self._dtotal_efficiencies[right_btype][iline]:.2e}$ \
& ${self._total_n[right_btype][iline]:.2f} \\pm {self._dtotal_n[right_btype][iline]:.2f}$ \
& ${what_value[iline]:.2f} \\pm {what_dvalue[iline]:.2f}$ \\\\\n\
		\\hline"""

		table += f"""
		\\caption{{Raw yields, efficiencies from MC, resulting total yields, and {what} values.}}
	\\end{{tabular}}\n
\\end{{table}}"""

		return table




	def print_yield_table(self):
		table = f"""
\\begin{{table}}
	\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|}}
		\\hline
		Bin & $\\Bu$ raw yield & $\\Bu$ eff. & $\\Bu$ total yield & $\\Bd$ raw yield & $\\Bd$ eff. & $\\Bd$ total yield & $\\Bs$ raw yield & $\\Bs$ eff. & $\\Bs$ total yield \\\\
		\\hline\n"""

		for iline in range(len(self._xbins)-1):
			table += f"""\t\t{self._xbins[iline]:.1f}--{self._xbins[iline+1]:.1f} \
& {render_val_err(self._yields["Bu"][iline], self._dyields["Bu"][iline], 2)} \
& {render_val_err(self._total_efficiencies["Bu"][iline], self._dtotal_efficiencies["Bu"][iline], 2)} \
& {render_val_err(self._total_n["Bu"][iline], self._dtotal_n["Bu"][iline], 2)} \
& {render_val_err(self._yields["Bd"][iline], self._dyields["Bd"][iline], 2)} \
& {render_val_err(self._total_efficiencies["Bd"][iline], self._dtotal_efficiencies["Bd"][iline], 2)} \
& {render_val_err(self._total_n["Bd"][iline], self._dtotal_n["Bd"][iline], 2)} \
& {render_val_err(self._yields["Bs"][iline], self._dyields["Bs"][iline], 2)} \
& {render_val_err(self._total_efficiencies["Bs"][iline], self._dtotal_efficiencies["Bs"][iline], 2)} \
& {render_val_err(self._total_n["Bs"][iline], self._dtotal_n["Bs"][iline], 2)} \\\\
		\\hline\
"""

		table += f"""
	\\end{{tabular}}
	\\caption{{Raw and corrected yields versus versus {vars_pretty[self._axis]} for {self._side}-side events.}}
	\\label{{table:yields-{self._side}-{self._axis}}}
\\end{{table}}\n"""

		return table

	def print_unc_table(self, btype):
		unc_names = ["stat"] + [x for x in self._dn_src.keys() if x != "stat"]
		print(f"DEBUG : unc_names = ")
		print(unc_names)
		unc_names_pretty_dict = {
			"stat": "Stat.", 
			"mcstat": "MC stat.", 
			"fitmodel": "Fit model", 
			"kinrwgt": "Kin. modeling", 
			"tracking": "Tracking eff.",
		}
		unc_names_pretty = []
		for unc_name in unc_names:
			if unc_name in unc_names_pretty_dict:
				unc_names_pretty.append(unc_names_pretty_dict[unc_name])
			else:
				unc_names_pretty.append(unc_name)
			if self._yield_unc_btype_corr[unc_name]:
				unc_names_pretty[-1] = unc_names_pretty[-1] + " $^{\\dagger}$"
		unc_names_pretty.append("Total")

		tabby = "\t&\t"
		ncolumns = len(unc_names) + 1
		column_string = "{" + "|c" * ncolumns + "|}"
		table = f"""\\begin{{table}}
	\\centering
	\\begin{{tabular}}{column_string}
		\\hline
		Bin & {tabby.join(unc_names_pretty)} \\\\
		\\hline\n"""

		for iline in range(len(self._xbins)-1):
			if self._axis == "pt":
				table += f"\t\t{self._xbins[iline]:.0f}--{self._xbins[iline+1]:.0f} "

			else:
				table += f"\t\t{self._xbins[iline]:.2f}--{self._xbins[iline+1]:.2f} "
			for unc_name in unc_names:
				table += f"{tabby}{100. * self._dn_src[unc_name][btype][iline] / self._n[btype][iline]:.2f}"
			table += f"{tabby}{100. * self._dn[btype][iline] / self._n[btype][iline]:.2f}"
			table += "\\\\\n\t\t\\hline\n"
		table += f"""	\\end{{tabular}}
	\\caption{{
		The impact in percent of each source of uncertainty on the efficiency-corrected $\\{btype}$ event yields, versus {vars_pretty[self._axis]} for {self._side}-side events.
		The $^{{\\dagger}}$ indicates sources that are correlated between $\\PB$ hadron flavors, and thus cancel in ratios to some degree.
	}}
	\\label{{table:syst-unc-{self._axis}-{btype}-{self._axis}-{self._side}}}
\\end{{table}}
		"""
		return table

	def save(self):
		output_path = f"/home/dyu7/BFrag/data/ffrs/ffrs_{self._axis}_{self._side}_{self._trigger_strategy}_{'binned' if self._binned else 'unbinned'}_{self._fitfunc}_{self._selection}.pkl"
		with open(output_path, "wb") as f:
			pickle.dump(self, f)

	def axis(self):
		return self._axis

	def side(self):
		return self._side

	def binned(self):
		return self._binned

	def trigger_strategy(self):
		return self._trigger_strategy

	def fitfunc(self):
		return self._fitfunc

	def selection(self):
		return self._selection

	def xbins(self):
		return self._xbins

	def Rsu(self):
		return self._Rsu

	def Rsd(self):
		return self._Rsd

	def Rdu(self):
		return self._Rdu

	def dRsu(self):
		return self._dRsu

	def dRsd(self):
		return self._dRsd

	def dRdu(self):
		return self._dRdu

	def dRsu_src(self, unc_src):
		return self._dRsu_src[unc_src]

	def dRsd_src(self, unc_src):
		return self._dRsd_src[unc_src]

	def dRdu_src(self, unc_src):
		return self._dRdu_src[unc_src]

	def dRsu_total(self):
		#pprint(self._dRsu_src)
		indiv_uncs = list(self._dRsu_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def dRsd_total(self):
		indiv_uncs = list(self._dRsd_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def dRdu_total(self):
		indiv_uncs = list(self._dRdu_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def fsfu(self):
		return self._fsfu

	def fsfd(self):
		return self._fsfd

	def fdfu(self):
		return self._fdfu

	def dfsfu(self):
		return self._dfsfu

	def dfsfd(self):
		return self._dfsfd

	def dfdfu(self):
		return self._dfdfu

	def dfsfu_src(self, unc_src):
		return self._dfsfu_src[unc_src]

	def dfsfd_src(self, unc_src):
		return self._dfsfd_src[unc_src]

	def dfdfu_src(self, unc_src):
		return self._dfdfu_src[unc_src]

	def dfsfu_total(self):
		indiv_uncs = list(self._dfsfu_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def dfsfd_total(self):
		indiv_uncs = list(self._dfsfd_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def dfdfu_total(self):
		indiv_uncs = list(self._dfdfu_src.values())
		total_unc = np.zeros_like(indiv_uncs[0])
		for indiv_unc in indiv_uncs:
			total_unc += np.array(indiv_unc)**2
		total_unc = total_unc**0.5
		return total_unc

	def __getitem__(self, key):
		return getattr(self, f"_{key}")

	def N(self, btype):
		return self._n[btype]

	def dN(self, btype):
		return self._dn[btype]

	def TotalN(self, btype):
		return self._total_n[btype]

	def dTotalN(self, btype):
		return self._dtotal_n[btype]

	def Yield(self, btype):
		return self._yields[btype]

	def dYield(self, btype):
		return self._dyields[btype]

	def Eff(self, btype):
		return self._efficiencies[btype]

	def dEff(self, btype):
		return self._defficiencies[btype]

	def TotalEff(self, btype):
		return self._total_efficiencies[btype]

	def dTotalEff(self, btype):
		return self._dtotal_efficiencies[btype]

	# Averages
	def Rsu_avg(self):
		return self._avg_Rsu

	def Rsd_avg(self):
		return self._avg_Rsd

	def Rdu_avg(self):
		return self._avg_Rdu

	def dRsu_avg(self):
		return self._avg_dRsu

	def dRsd_avg(self):
		return self._avg_dRsd

	def dRdu_avg(self):
		return self._avg_dRdu

	def fsfu_avg(self):
		return self._avg_fsfu

	def fsfd_avg(self):
		return self._avg_fsfd

	def fdfu_avg(self):
		return self._avg_fdfu

	def dfsfu_avg(self):
		return self._avg_dfsfu

	def dfsfd_avg(self):
		return self._avg_dfsfd

	def dfdfu_avg(self):
		return self._avg_dfdfu

	def N_avg(self, btype):
		return self._sum_n[btype]

	def dN_avg(self, btype):
		return self._dsum_n[btype]

	# Add uncertainty as symmetric envelope of other FFRs
	# - Uncertainty is computed first for yields, for each alternative fit model
	# - Yield uncertainties are propagated to ratios individually, and added in quadrature.
	# - Computation is done in finalize()
	def add_fitmodel_uncertainty(self, other_ffrs):
		self._other_ffrs = other_ffrs

	# Add uncertainty on yields
	# - Format: {btype: [frac_unc, frac_unc, ...]}
	# - btype_corr controls if the uncertainty is correlated between btypes (=>partially cancel in ratios. Or uncancel if unlucky)
	def add_yield_uncertainty(self, unc_name, unc_values, btype_corr=False):
		if not self._yield_uncertainties:
			self._yield_uncertainties = {}
			self._yield_unc_btype_corr = {}
		self._yield_uncertainties[unc_name] = unc_values
		self._yield_unc_btype_corr[unc_name] = btype_corr


	def __repr__(self):
		from pprint import pformat
		print("fd/fu:")
		pprint(self._fdfu)
		print("fs/fu:")
		pprint(self._fsfu)
		print("fs/fd:")
		pprint(self._fsfd)
		return f"""
fd/fu:
{pformat(self._fdfu, indent=4, width=1)}
fs/fu:
{pformat(self._fsfu, indent=4, width=1)}
fs/fd:
{pformat(self._fsfd, indent=4, width=1)}
"""

class FFRFits:
	def __init__(self, ffr, statonly=False):
		self._whats = ["Rsu", "Rsd", "Rdu", "fsfu", "fsfd", "fdfu"]
		self._axis             = ffr.axis()
		self._side             = ffr.side()
		self._binned           = ffr.binned()
		self._trigger_strategy = ffr.trigger_strategy()
		self._fitfunc          = ffr.fitfunc()
		self._selection        = ffr.selection()

		self._ffr = ffr
		self._tgraphs = {}
		self._fits = {}
		self._fit_results = {}
		self._chi2 = {}
		self._ndf = {}
		self._statonly = statonly

	def make_tgraphs(self):
		# Make TGraphs
		self._lo_edges = self._ffr.xbins()[:-1]
		self._hi_edges = self._ffr.xbins()[1:]
		if self._axis == "pt":
			self._bin_centers = fonll_pt_barycenter_v(lo=self._lo_edges, hi=self._hi_edges)
		elif self._axis == "y":
			self._bin_centers = fonll_y_barycenter_v(lo=self._lo_edges, hi=self._hi_edges)
		bin_xerrs = [self._bin_centers - self._lo_edges, self._hi_edges - self._bin_centers]

		for what in self._whats:
			if self._statonly:
				dy = getattr(self._ffr, f"d{what}")()
			else:
				dy = getattr(self._ffr, f"d{what}_total")()
			self._tgraphs[what] = ROOT.TGraphAsymmErrors(
										len(self._ffr.xbins()), 
										self._bin_centers, 
										getattr(self._ffr, what)(),
										np.zeros(len(self._bin_centers)), # dx low
										np.zeros(len(self._bin_centers)), # dx high
										dy, # dy low
										dy  # dy high
									)
			self._tgraphs[what].SetName(f"ffr_{what}")

	def fit(self, poly_n):
		if poly_n in self._fits:
			raise ValueError(f"In FFRFits::fit(), poly_n {poly_n} already exists.")
		self._fits[poly_n] = {}
		self._fit_results[poly_n] = {}

		poly_string = "+".join([f"[{i}] * x**{i}" for i in range(poly_n+1)])
		for what in self._whats:
			fit_name = f"fit_{what}_{poly_n}"
			#print(f"DEBUG : Fitting {what} {poly_n}")
			self._fits[poly_n][what] = ROOT.TF1(fit_name, poly_string, np.min(self._ffr.xbins()), np.max(self._ffr.xbins()))
			for i in range(poly_n):
				if i == 0:
					self._fits[poly_n][what].SetParameter(i, getattr(self._ffr, f"{what}_avg")())
				else:
					self._fits[poly_n][what].SetParameter(i, 0.)
			self._fit_results[poly_n][what] = self._tgraphs[what].Fit(self._fits[poly_n][what], "QR0S")

	def ftest(self, what, n1, n2):
		if not n1 < n2:
			raise ValueError("In ftest(n1, n2), n1 must be less than n2")

		# Compute chi2
		chi2_1 = self.chi2(what, n1)
		chi2_2 = self.chi2(what, n2)

		nll1 = 0.5 * chi2_1
		nll2 = 0.5 * chi2_2 
		npt = len(self._ffr[what])
		fval = (nll1 - nll2) / nll2 * (npt - n2) / (n2 - n1)
		p = 1. - stats.f.cdf(fval, n2 - n1, npt - n2)
		return fval, p

	def chi2(self, what, poly_n):
		chi2 = 0.0
		for ipt in range(len(getattr(self._ffr, what)())):
			this_x = self._bin_centers[ipt]
			if self._statonly:
				chi2 += (self._fits[poly_n][what].Eval(this_x) - self._ffr[what][ipt])**2 / getattr(self._ffr, f"d{what}_src")("stat")[ipt]**2

			else:
				chi2 += (self._fits[poly_n][what].Eval(this_x) - self._ffr[what][ipt])**2 / getattr(self._ffr, f"d{what}_total")()[ipt]**2
		return chi2

	def ndf(self, what, poly_n):
		return len(self._ffr[what]) - (poly_n + 1) #self._ndf[poly_n][what]

	def get_fit(self, what, poly_n):
		if not poly_n in self._fits:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fits[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fits[poly_n][what]

	def get_fit_result(self, what, poly_n):
		if not poly_n in self._fit_results:
			raise ValueError(f"In FFRFits.get_fit_result(), poly_n={poly_n} not found")
		if not what in self._fits[poly_n]:
			raise ValueError(f"In FFRFits.get_fit_result(), what={what} not found")

		return self._fit_results[poly_n][what]

	def get_fitparam(self, what, poly_n, ipar):
		if not poly_n in self._fits:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fits[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fits[poly_n][what].GetParameter(ipar)

	def get_fitparamerr(self, what, poly_n, ipar):
		if not poly_n in self._fits:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fits[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fits[poly_n][what].GetParError(ipar)

	def get_chi2(self, what, poly_n):
		if not poly_n in self._fit_results:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fit_results[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fit_results[poly_n][what].Chi2()

	def get_ndf(self, what, poly_n):
		if not poly_n in self._fit_results:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fit_results[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fit_results[poly_n][what].Ndf()

	def get_prob(self, what, poly_n):
		if not poly_n in self._fit_results:
			raise ValueError(f"In FFRFits.get_fit(), poly_n={poly_n} not found")
		if not what in self._fit_results[poly_n]:
			raise ValueError(f"In FFRFits.get_fit(), what={what} not found")

		return self._fit_results[poly_n][what].Prob()

	def save(self):
		output_path = f"/home/dyu7/BFrag/data/ffrs/fits_{self._axis}_{self._side}_{self._trigger_strategy}_{'binned' if self._binned else 'unbinned'}_{self._fitfunc}_{self._selection}.root"
		output_file = ROOT.TFile(output_path, "RECREATE")
		for what in self._whats:
			self._tgraphs[what].Write()
		for poly_n, dict2 in self._fits.items():
			for what, fit in dict2.items():
				self._fits[poly_n][what].Write()
		output_file.Close()

	def load(self):
		re_tgraph = re.compile("ffr_(?P<what>[a-zA-Z0-9]+)")
		re_fit = re.compile("fit_(?P<what>[a-zA-Z0-9]+)_(?P<poly_n>\d+)")

		output_path = f"/home/dyu7/BFrag/data/ffrs/fits_{self._axis}_{self._side}_{self._trigger_strategy}_{'binned' if self._binned else 'unbinned'}_{self._fitfunc}_{self._selection}.root"
		output_file = ROOT.TFile(output_path, "READ")
		for key in output_file.GetListOfKeys():
			keyname = key.GetName()
			obj = output_file.Get(keyname)
			if "TGraph" in obj.ClassName():
				match_tgraph = re_tgraph.search(keyname)
				this_what = match_tgraph.group("what")
				self._tgraphs[this_what] = obj 
				#self._tgraphs[this_what].SetDirectory(0)
			elif "TF1" in obj.ClassName():
				match_fit = re_fit.search(keyname)
				this_what = match_fit.group("what")
				this_poly_n = match_fit.group("poly_n")
				if not this_what in self._whats:
					self._whats.append(this_what)
				if not this_poly_n in self._fits:
					self._fits[this_poly_n] = {}
				self._fits[this_poly_n][this_what] = obj 
				#self._fits[this_poly_n][this_what].SetDirectory(0)

		output_file.Close()

	def print(self):
		print("\nPrinting FFR fit results for:")
		print(f"\taxis = {self._axis}")
		print(f"\tside = {self._side}")
		print(f"\tbinned = {self._binned}")
		print(f"\ttrigger_strategy = {self._trigger_strategy}")
		print(f"\tfitfunc = {self._fitfunc}")
		print(f"\tselection = {self._selection}")
		for what in self._whats:
			for poly_n in self._fits.keys():
				print(f"\t{what}, degree {poly_n}:")
				print(self._fits[poly_n][what])
				for i in range(poly_n+1):
					print("\tp{} = {:.3f} \\pm {:.3f}".format(i, self._fits[poly_n][what].GetParameter(i), self._fits[poly_n][what].GetParError(i)))
					print(f"\tQuality: {self._fit_results[poly_n][what].Chi2()} / {self._fit_results[poly_n][what].Ndf()} = {self._fit_results[poly_n][what].Prob()}")

class FFRPlot:
	def __init__(self, ffrs, 
		fits=None, 
		legend_entries=None, 
		save_tag="test",
		xlabel="",
		xlim=[0., 30.],
		ylim=[0., 0.4], 
		var=""):
		print(f"Making FFR plot for save_tag={save_tag}")
		if not var in ["pt", "y"]:
			raise ValueError("In FFRPlot(), argument 'var' must be 'pt' or 'y'")
		self._save_tag = save_tag
		self._ffrs = ffrs
		self._fits = fits
		self._legend_entries = legend_entries
		self._save_tag = save_tag
		self._xlabel = xlabel
		self._xlim = xlim
		self._ylim = ylim

		#colorwheel = [seaborn_colors.get_root_color("Blues_d", 2), 
		#				seaborn_colors.get_root_color("Reds_d", 3),
		#				seaborn_colors.get_root_color("Greens_d", 3),
		#				seaborn_colors.get_root_color("Oranges_d", 4),
		#				seaborn_colors.get_root_color("Purples_d", 4),
		#				seaborn_colors.get_root_color("Blues_d", 5)]
		colorwheel = ["blue", "red", "green", "orange", "purple"]

		fig_su, ax_su = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			#bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			#bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			lo_edges = ffr.xbins()[:-1]
			hi_edges = ffr.xbins()[1:]
			if var == "pt":
				bin_centers = fonll_pt_barycenter_v(lo=lo_edges, hi=hi_edges)
			elif var == "y":
				bin_centers = fonll_y_barycenter_v(lo=lo_edges, hi=hi_edges)
			bin_xerrs = [bin_centers - lo_edges, hi_edges - bin_centers]
			
			#print(bin_centers)
			#print(bin_widths)
			#print(ffr.Rsu())
			#print(ffr.dRsu())
			ax_su.errorbar(
				x=bin_centers, 
				y=ffr.Rsu(),
				xerr=bin_xerrs,
				yerr=ffr.dRsu(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)

			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="Rsu", poly_n=0).Eval(x) for x in xfit])
				p0 = fits[i].get_fitparam(what="Rsu", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="Rsu", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="Rsu", poly_n=0)
				prob_p0 = fits[i].get_prob(what="Rsu", poly_n=0)
				ax_su.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="Rsu", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="Rsu", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="Rsu", poly_n=1)
				prob_p1 = fits[i].get_prob(what="Rsu", poly_n=1)
				ax_su.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")

		ax_su.set_xlabel(self._xlabel, fontsize=18)
		ax_su.set_xlim(self._xlim)
		ax_su.set_ylim([y/2 for y in self._ylim])
		ax_su.set_ylabel(r"$\mathcal{R}_{su}$", fontsize=18)
		ax_su.xaxis.set_ticks_position("both")
		ax_su.yaxis.set_ticks_position("both")
		ax_su.tick_params(direction="in")
		ax_su.legend()
		fig_su.savefig(f"/home/dyu7/BFrag/data/ffrs/Rsu_{save_tag}.png")
		fig_su.savefig(f"/home/dyu7/BFrag/data/ffrs/Rsu_{save_tag}.pdf")		
		plt.close(fig_su)

		fig_sd, ax_sd = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			ax_sd.errorbar(
				x=bin_centers, 
				y=ffr.Rsd(),
				xerr=bin_widths,
				yerr=ffr.dRsd(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)

			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="Rsd", poly_n=0).Eval(x) for x in xfit])
				p0 = fits[i].get_fitparam(what="Rsd", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="Rsd", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="Rsd", poly_n=0)
				prob_p0 = fits[i].get_prob(what="Rsd", poly_n=0)
				ax_sd.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="Rsd", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="Rsd", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="Rsd", poly_n=1)
				prob_p1 = fits[i].get_prob(what="Rsd", poly_n=1)
				ax_sd.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")
		ax_sd.set_xlabel(self._xlabel)
		ax_sd.set_xlim(self._xlim)
		ax_sd.set_ylim([y/2 for y in self._ylim])
		ax_sd.set_ylabel(r"$\mathcal{R}_{sd}$")
		ax_sd.xaxis.set_ticks_position("both")
		ax_sd.yaxis.set_ticks_position("both")
		ax_sd.tick_params(direction="in")
		ax_sd.legend()
		fig_sd.savefig(f"/home/dyu7/BFrag/data/ffrs/Rsd_{save_tag}.png")
		fig_sd.savefig(f"/home/dyu7/BFrag/data/ffrs/Rsd_{save_tag}.pdf")		
		plt.close(fig_sd)

		fig_du, ax_du = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			ax_du.errorbar(
				x=bin_centers, 
				y=ffr.Rdu(),
				xerr=bin_widths,
				yerr=ffr.dRdu(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)

			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="Rdu", poly_n=0).Eval(x) for x in xfit])
				p0 = fits[i].get_fitparam(what="Rdu", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="Rdu", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="Rdu", poly_n=0)
				prob_p0 = fits[i].get_prob(what="Rdu", poly_n=0)
				ax_du.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="Rdu", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="Rdu", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="Rdu", poly_n=1)
				prob_p1 = fits[i].get_prob(what="Rdu", poly_n=1)
				ax_du.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")
		ax_du.set_xlabel(self._xlabel)
		ax_du.set_xlim(self._xlim)
		ax_du.set_ylim([0., 1.0])
		ax_du.set_ylabel(r"$\mathcal{R}_{du}$")
		ax_du.xaxis.set_ticks_position("both")
		ax_du.yaxis.set_ticks_position("both")
		ax_du.tick_params(direction="in")
		ax_du.legend()
		fig_du.savefig(f"/home/dyu7/BFrag/data/ffrs/Rdu_{save_tag}.png")
		fig_du.savefig(f"/home/dyu7/BFrag/data/ffrs/Rdu_{save_tag}.pdf")		
		plt.close(fig_du)


		fig_fsfu, ax_fsfu = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			ax_fsfu.errorbar(
				x=bin_centers, 
				y=ffr.fsfu(),
				xerr=bin_widths,
				yerr=ffr.dfsfu(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)
			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="fsfu", poly_n=0).Eval(x) for x in xfit])
				p0 = fits[i].get_fitparam(what="fsfu", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="fsfu", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="fsfu", poly_n=0)
				prob_p0 = fits[i].get_prob(what="fsfu", poly_n=0)
				ax_fsfu.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="fsfu", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="fsfu", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="fsfu", poly_n=1)
				prob_p1 = fits[i].get_prob(what="fsfu", poly_n=1)
				ax_fsfu.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")
		ax_fsfu.set_xlabel(self._xlabel)
		ax_fsfu.set_xlim(self._xlim)
		ax_fsfu.set_ylim(self._ylim)
		ax_fsfu.set_ylabel(r"$f_{s}/f_{u}$")
		ax_fsfu.xaxis.set_ticks_position("both")
		ax_fsfu.yaxis.set_ticks_position("both")
		ax_fsfu.tick_params(direction="in")
		ax_fsfu.legend()
		fig_fsfu.savefig(f"/home/dyu7/BFrag/data/ffrs/fsfu_{save_tag}.png")
		fig_fsfu.savefig(f"/home/dyu7/BFrag/data/ffrs/fsfu_{save_tag}.pdf")		
		plt.close(fig_fsfu)

		fig_fsfd, ax_fsfd = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			ax_fsfd.errorbar(
				x=bin_centers, 
				y=ffr.fsfd(),
				xerr=bin_widths,
				yerr=ffr.dfsfd(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)
			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="fsfd", poly_n=0).Eval(x) for x in xfit])
				p0 = fits[i].get_fitparam(what="fsfd", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="fsfd", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="fsfd", poly_n=0)
				prob_p0 = fits[i].get_prob(what="fsfd", poly_n=0)
				ax_fsfd.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="fsfd", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="fsfd", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="fsfd", poly_n=1)
				prob_p1 = fits[i].get_prob(what="fsfd", poly_n=1)
				ax_fsfd.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")
		ax_fsfd.set_xlabel(self._xlabel)
		ax_fsfd.set_xlim(self._xlim)
		ax_fsfd.set_ylim(self._ylim)
		ax_fsfd.set_ylabel(r"$f_{s}/f_{d}$")
		ax_fsfd.xaxis.set_ticks_position("both")
		ax_fsfd.yaxis.set_ticks_position("both")
		ax_fsfd.tick_params(direction="in")
		ax_fsfd.legend()
		fig_fsfd.savefig(f"/home/dyu7/BFrag/data/ffrs/fsfd_{save_tag}.png")
		fig_fsfd.savefig(f"/home/dyu7/BFrag/data/ffrs/fsfd_{save_tag}.pdf")		
		plt.close(fig_fsfd)

		fig_fdfu, ax_fdfu = plt.subplots(1,1)
		for i, ffr in enumerate(ffrs):
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			ax_fdfu.errorbar(
				x=bin_centers, 
				y=ffr.fdfu(),
				xerr=bin_widths,
				yerr=ffr.dfdfu(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)
			if fits:
				xfit    = np.linspace(np.min(ffr.xbins()), np.max(ffr.xbins()), 100)
				yfit_p0 = np.array([fits[i].get_fit(what="fdfu", poly_n=0).Eval(x) for x in xfit])
				p0      = fits[i].get_fitparam(what="fdfu", poly_n=0, ipar=0)
				chi2_p0 = fits[i].get_chi2(what="fdfu", poly_n=0)
				ndf_p0  = fits[i].get_ndf(what="fdfu", poly_n=0)
				prob_p0 = fits[i].get_prob(what="fdfu", poly_n=0)
				ax_fdfu.plot(
					xfit, 
					yfit_p0, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} const fit",)

				yfit_p1 = np.array([fits[i].get_fit(what="fdfu", poly_n=1).Eval(x) for x in xfit])
				chi2_p1 = fits[i].get_chi2(what="fdfu", poly_n=1)
				ndf_p1  = fits[i].get_ndf(what="fdfu", poly_n=1)
				prob_p1 = fits[i].get_prob(what="fdfu", poly_n=1)
				ax_fdfu.plot(
					xfit, 
					yfit_p1, 
					color=colorwheel[i],
					label=f"{legend_entries[i]} linear fit")
		ax_fdfu.set_xlabel(self._xlabel)
		ax_fdfu.set_xlim(self._xlim)
		ax_fdfu.set_ylim([0.5, 1.5])
		ax_fdfu.set_ylabel(r"$f_{d}/f_{u}$")
		ax_fdfu.xaxis.set_ticks_position("both")
		ax_fdfu.yaxis.set_ticks_position("both")
		ax_fdfu.tick_params(direction="in")
		ax_fdfu.legend()
		fig_fdfu.savefig(f"/home/dyu7/BFrag/data/ffrs/fdfu_{save_tag}.png")
		fig_fdfu.savefig(f"/home/dyu7/BFrag/data/ffrs/fdfu_{save_tag}.pdf")		
		plt.close(fig_fdfu)

	def corrected_yield_plot(self):
		fig_yield, ax_yield = plt.subplots(1,1)
		colors = {
			"Bu": "blue", 
			"Bd": "green", 
			"Bs": "orange"
		}
		markers = [".", "s", "D"]
		marker_sizes = [10, 7, 10]
		maxy = -1.0
		for btype in ["Bu", "Bd", "Bs"]:
			for i, ffr in enumerate(self._ffrs):
				bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
				bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
				ax_yield.errorbar(
					x=bin_centers, 
					y=ffr.TotalN(btype) / (2*bin_widths),
					xerr=bin_widths,
					yerr=ffr.dTotalN(btype) / (2*bin_widths), 
					marker=markers[i], 
					markersize=marker_sizes[i],
					#color=colors[btype],
					mec=colors[btype],
					mfc=colors[btype] if i == 0 else "none",
					mew=1,
					label=f"{btype}, {self._legend_entries[i]}",
					ls="none",
					ecolor=colors[btype],
					elinewidth=1,
					)
				#print(ffr.N(btype))
				maxy = max(maxy, np.nanmax(ffr.TotalN(btype)[ffr.TotalN(btype) != np.inf]))
		ax_yield.set_xlabel(self._xlabel)
		ax_yield.set_xlim(self._xlim)
		ax_yield.set_yscale("log")
		print(ffr.axis())
		if ffr.axis() == "pt":
			ax_yield.set_ylabel(r"Corrected yield / GeV")
			ax_yield.set_ylim([1.e3, 1.e8])
		else:
			ax_yield.set_ylabel(r"Corrected yield")
			ax_yield.set_ylim([3.e4, 1.e8])
		ax_yield.xaxis.set_ticks_position("both")
		ax_yield.yaxis.set_ticks_position("both")
		ax_yield.tick_params(direction="in")
		ax_yield.legend()
		fig_yield.savefig(f"/home/dyu7/BFrag/data/yields/corr_yields_{self._save_tag}.png")
		plt.close(fig_yield)



def format_N(N, dN):
	if np.isfinite(N) and (N > 0):
		power10 = math.floor(math.log(N) / math.log(10.))
		return "$({:.2f} \\pm {:.2f}) \\times 10^{{{:d}}}$".format(N / 10**power10, dN / 10**power10, power10)
	else:
		return "{}".format(N)

def format_R(R, dR):
	return "${:.3f} \\pm {:.3f}$".format(R, dR)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Compute efficiencies")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="Fit function (johnson or hypatia)")
	parser.add_argument("--selection", "-s", type=str, default="nominal", help="Selection name (nominal or HiTrkPt or HiMuonPt)")
	parser.add_argument("--all", action="store_true", help="Run everything")
	parser.add_argument("--ffrs", action="store_true", help="Run FFR calculation")
	parser.add_argument("--fits", action="store_true", help="Run FFR fits")
	parser.add_argument("--plots", action="store_true", help="Make plots and tables")
	args = parser.parse_args()

	do_ffr = args.all or args.ffrs
	do_fits = args.all or args.fits 
	do_plots = args.all or args.plots

	if not (do_ffr or do_fits or do_plots):
		raise ValueError("You didn't specify anything to do! --all, --ffrs, --fits, --plots")

	sides_to_run = ["tag", "probe", "tagx"]
	if args.selection in ["MediumMuonPt", "HiMuonPt", "MediumMuonID"]:
		sides_to_run = ["tag"]

	ffrs = {}
	for axis in ["pt", "y"]:
		ffrs[axis] = {}
		for side in sides_to_run: # , "tagMaxPt", "probeMaxPt"
			ffrs[axis][side] = {}
			for binned in [True]:
				ffrs[axis][side][binned] = {}
				for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7", "HLT_Mu9_IP5"]: # HLT_Mu9_IP6
					if do_ffr:
						print(f"Computing FFRs for {axis} {side} binned={binned} {trigger_strategy}")
						ffrs[axis][side][binned][trigger_strategy] = FFRData(axis=axis,
																		side=side,
																		binned=binned,
																		trigger_strategy=trigger_strategy,
																		selection=args.selection,
																		fitfunc=args.fitfunc)
						ffrs[axis][side][binned][trigger_strategy].load_yields()
						ffrs[axis][side][binned][trigger_strategy].load_efficiencies()

						# If central value, add uncertainties
						if (args.selection in ["nominal"]) and args.fitfunc == "johnson":
							'''
							with open(f"/home/dyu7/BFrag/data/ffrs/ffrs_{axis}_{side}_{trigger_strategy}_{'binned' if binned else 'unbinned'}_poly_{args.selection}.pkl", "rb") as f:
								ffr_poly = pickle.load(f)
							with open(f"/home/dyu7/BFrag/data/ffrs/ffrs_{axis}_{side}_{trigger_strategy}_{'binned' if binned else 'unbinned'}_3gauss_{args.selection}.pkl", "rb") as f:
								ffr_3gauss = pickle.load(f)
							ffrs[axis][side][binned][trigger_strategy].add_fitmodel_uncertainty([ffr_poly, ffr_3gauss])
							'''
							syst_uncs = {}
							with open(f"/home/dyu7/BFrag/data/systematics/fitmodel.pkl", "rb") as f:
								unc_dict = pickle.load(f) # fracunc[(var, btype, side)]
								syst_uncs["fitmodel"] = {}
								for btype in ["Bu", "Bs", "Bd"]:
									hack_side = side
									if hack_side == "tagx":
										hack_side = "tag"
									hack_axis = axis
									if hack_axis == "y":
										hack_axis = "absy"
									syst_uncs["fitmodel"][btype] = unc_dict[(hack_axis, btype, hack_side)]
							ffrs[axis][side][binned][trigger_strategy].add_yield_uncertainty("fitmodel", syst_uncs["fitmodel"], btype_corr=False)

							# Kinematic reweighting
							# Note: source is different for tag and probe sides
							if side == "tag" or side == "tagx":
								with open(f"/home/dyu7/BFrag/data/systematics/fonll.pkl", "rb") as f:
									unc_dict = pickle.load(f) # fracunc[(var, btype, side)]
									syst_uncs["kinrwgt"] = {}
									for btype in ["Bu", "Bs", "Bd"]:
										hack_side = side
										if hack_side == "tagx":
											hack_side = "tag"
										hack_axis = axis
										if hack_axis == "y":
											hack_axis = "absy"
										syst_uncs["kinrwgt"][btype] = unc_dict[(hack_axis, btype, hack_side)]
								ffrs[axis][side][binned][trigger_strategy].add_yield_uncertainty("kinrwgt", syst_uncs["kinrwgt"], btype_corr=True)
							elif side == "probe":
								with open("/home/dyu7/BFrag/data/systematics/rwgt_probe.pkl", "rb") as f:
									unc_dict = pickle.load(f)
									syst_uncs["kinrwgt"] = {}
									for btype in ["Bu", "Bs", "Bd"]:
										if axis == "y":
											hackaxis = "absy"
										else:
											hackaxis = axis
										syst_uncs["kinrwgt"][btype] = unc_dict[(hackaxis, btype, side)]
								ffrs[axis][side][binned][trigger_strategy].add_yield_uncertainty("kinrwgt", syst_uncs["kinrwgt"], btype_corr=True)

							# Tracking efficiency: flat 2.3% per track
							# - Magic of numpy: you can supply a single float, and it'll be broadcast to the whole array!
							ffrs[axis][side][binned][trigger_strategy].add_yield_uncertainty("tracking", {"Bu": 0.023, "Bd": 0.046529, "Bs": 0.046529}, btype_corr=True)

						print(f"{axis} {side} {binned} {trigger_strategy} {args.fitfunc} {args.selection}")
						ffrs[axis][side][binned][trigger_strategy].finalize()
						ffrs[axis][side][binned][trigger_strategy].save()
					else:
						save_path = f"/home/dyu7/BFrag/data/ffrs/ffrs_{axis}_{side}_{trigger_strategy}_{'binned' if binned else 'unbinned'}_{args.fitfunc}_{args.selection}.pkl"
						with open(save_path, "rb") as f:
							ffrs[axis][side][binned][trigger_strategy] = pickle.load(f)
					print(f"\nPrint Rsu for {axis} {side} binned={binned} {trigger_strategy}")
					if trigger_strategy == "HLT_all":
						table = ffrs[axis][side][binned][trigger_strategy].print_yield_table_2flav("Rsu")


	fits = {}
	for axis in ["pt", "y"]:
		fits[axis] = {}
		for side in sides_to_run: # , "tagMaxPt", "probeMaxPt"
			fits[axis][side] = {}
			for binned in [True]:
				fits[axis][side][binned] = {}
				for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7", "HLT_Mu9_IP5"]: # HLT_Mu9_IP6
					if do_fits:
						print(f"Fitting FFRs for {axis} {side} binned={binned} {trigger_strategy}")
						fits[axis][side][binned][trigger_strategy] = FFRFits(ffrs[axis][side][binned][trigger_strategy])
						fits[axis][side][binned][trigger_strategy].make_tgraphs()
						fits[axis][side][binned][trigger_strategy].fit(poly_n=0)
						fits[axis][side][binned][trigger_strategy].fit(poly_n=1)
						fits[axis][side][binned][trigger_strategy].save()
						fits[axis][side][binned][trigger_strategy].print()
					else:
						fits[axis][side][binned][trigger_strategy] = FFRFits(ffrs[axis][side][binned][trigger_strategy])
						fits[axis][side][binned][trigger_strategy].load()


	# FFR plot comparing tag and probe
	if do_plots:
		xlims = {"pt": [0, 50], "y": [0, 2.5]}
		xlabels = {"pt": r"$p_{T}$ [GeV]", "y": r"$|y|$"}
		for axis in ["pt", "y"]:#, "y"]:
			for binned in [True]:
				if binned:
					binned_str = "binned"
				else:
					binned_str = "unbinned"
				for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7", "HLT_Mu9_IP5"]: # HLT_Mu9_IP6
					if "tagx" in sides_to_run and "probe" in sides_to_run:
						print(f"Plotting FFRs for {axis} binned={binned} {trigger_strategy}")
						ffrplot = FFRPlot([ffrs[axis]["tag"][binned][trigger_strategy], ffrs[axis]["probe"][binned][trigger_strategy]], 
									fits=None,#[fits[axis]["tag"][binned][trigger_strategy], fits[axis]["probe"][binned][trigger_strategy]], 								
									legend_entries=["Tag", "Probe"],
									xlabel=xlabels[axis],
									xlim=xlims[axis],
									ylim=[0., 0.4],
									save_tag=f"{axis}_{binned_str}_{trigger_strategy}_{args.selection}_{args.fitfunc}",
									var=axis
							)
						ffrplot.corrected_yield_plot()
						for side in ["tag", "probe"]:
							print(f"Relative {side} {trigger_strategy} uncertainties:")
							print("Rdu = ")
							print(ffrs[axis][side][binned][trigger_strategy].Rdu())
							print("dRdu = ")
							print(ffrs[axis][side][binned][trigger_strategy].dRdu())
							print("dRdu / Rdu = ")
							print(ffrs[axis][side][binned][trigger_strategy].dRdu() / ffrs[axis][side][binned][trigger_strategy].Rdu())
							print("TotalN(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy].TotalN("Bd"))
							print("dTotalN(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy].dTotalN("Bd"))
							print("Yields(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy]._yields["Bd"])
							print("dYields(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy]._dyields["Bd"])
							print("Total eff(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy]._total_efficiencies["Bd"])
							print("dTotal eff(Bd) = ")
							print(ffrs[axis][side][binned][trigger_strategy]._dtotal_efficiencies["Bd"])
					else:
						# Just make one plot per side. No corrected yield plot. 
						for side in sides_to_run:
							ffrplot = FFRPlot([ffrs[axis][side][binned][trigger_strategy]], 
										fits=None,#[fits[axis]["tag"][binned][trigger_strategy], fits[axis]["probe"][binned][trigger_strategy]], 								
										legend_entries=[side],
										xlabel=xlabels[axis],
										xlim=xlims[axis],
										ylim=[0., 0.4],
										save_tag=f"{axis}_{binned_str}_{trigger_strategy}_{args.selection}_{args.fitfunc}_{side}",
										var=axis
								)


				# Final plot: HLT_Mu7 for tag, HLT_all for probe
				if "tagx" in sides_to_run and "probe" in sides_to_run:
					print(f"Plotting FFRs for {axis} binned={binned}, final trigger strategy")
					ffrplot = FFRPlot([ffrs[axis]["tagx"][binned]["HLT_all"], ffrs[axis]["probe"][binned]["HLT_all"]], 
								fits=None,#[fits[axis]["tag"][binned]["HLT_Mu7"], fits[axis]["probe"][binned]["HLT_all"]], 
								legend_entries=["Tag", "Probe"],
								xlabel=xlabels[axis],
								xlim=xlims[axis],
								ylim=[0., 0.4],
								save_tag=f"{axis}_{binned_str}_{args.selection}_{args.fitfunc}_final", 
								var=axis
						)
					ffrplot.corrected_yield_plot()

		# Uncertainty table
		for axis in ["pt", "y"]:
			for side in sides_to_run:
				for btype in ["Bu", "Bs", "Bd"]:
					unc_table = ffrs[axis][side][True]["HLT_all"].print_unc_table(btype)
					with open(f"/home/dyu7/BFrag/data/systematics/syst_table_{axis}_{side}_{btype}.tex", "w") as f:
						f.write(unc_table)

		# Make latex table of raw and corrected yields
		for axis in ["pt", "y"]:
			for side in sides_to_run: # , "tagMaxPt", "probeMaxPt"
				for binned in [True]:
					for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7", "HLT_Mu9_IP5"]: # HLT_Mu9_IP6
						# Yield table
						yield_table_path = f"/home/dyu7/BFrag/data/yields/yields_{axis}_{side}_{trigger_strategy}_{'binned' if binned else 'unbinned'}_{args.fitfunc}_{args.selection}.tex"
						with open(yield_table_path, "w") as f:
							f.write(ffrs[axis][side][binned][trigger_strategy].print_yield_table())

	for btype in ["Bu", "Bd", "Bs"]:
		for side in sides_to_run:
			print("\n")
			print(f"Corrected yields (to filter) {side} {btype} = {ffrs['pt'][side][True]['HLT_all'].N(btype)} +/- {ffrs['pt'][side][True]['HLT_all'].dN(btype)}")

			print(f"Eff {side} {btype} = {ffrs['pt'][side][True]['HLT_all'].Eff(btype)} +/- {ffrs['pt'][side][True]['HLT_all'].dEff(btype)}")
			print("")
			print(f"Corrected yields (to inclusive) {side} {btype} = {ffrs['pt'][side][True]['HLT_all'].TotalN(btype)} +/- {ffrs['pt'][side][True]['HLT_all'].dTotalN(btype)}")
			print(f"Total Eff {side} {btype} = {ffrs['pt'][side][True]['HLT_all'].TotalEff(btype)} +/- {ffrs['pt'][side][True]['HLT_all'].dTotalEff(btype)}")



	# Make latex table of results
	with open(f"/home/dyu7/BFrag/data/ffrs/Rtable_{args.selection}.tex", 'w') as ftable:
		for side in sides_to_run:
			for axis in ["pt", "y"]:
				ftable.write("""
\n
\\begin{table}[htbp]
\t\\centering
\t\\resizebox{\\textwidth}{!}{
\t\\begin{tabular}{|l|c|c|c|c|c|c|}
\t\t\\hline
\t\tBin & $N^{\\mathrm{corr}}_{\\Bs}$ & $N^{\\mathrm{corr}}_{\\Bd}$ & $N^{\\mathrm{corr}}_{\\Bu}$ & $\\Rsu$ & $\\Rsd$ & $\\Rdu$ \\\\
\t\t\\hline""")
				if side == "tag":
					this_ffr = ffrs[axis]["tag"][True]["HLT_all"]
				elif side == "probe":
					this_ffr = ffrs[axis]["probe"][True]["HLT_all"]
				xbins = this_ffr.xbins()
				for ibin in range(len(xbins)-1):
					xleft = xbins[ibin]
					xright = xbins[ibin+1]
					if axis == "pt":
						bin_str = "{:d}-{:d}".format(int(xleft), int(xright))
					else:
						bin_str = "{:.2f}-{:.2f}".format(xleft, xright)
					ftable.write("\t\t{bin_str} & {Ns} & {Nd} & {Nu} & {Rsu} & {Rsd} & {Rdu} \\\\ \n\t\t\\hline\n".format(
						bin_str=bin_str,
						Ns=format_N(this_ffr.N("Bs")[ibin], this_ffr.dN("Bs")[ibin]),
						Nd=format_N(this_ffr.N("Bd")[ibin], this_ffr.dN("Bd")[ibin]),
						Nu=format_N(this_ffr.N("Bu")[ibin], this_ffr.dN("Bu")[ibin]),
						Rsu=format_R(this_ffr.Rsu()[ibin], this_ffr.dRsu()[ibin]),
						Rsd=format_R(this_ffr.Rsd()[ibin], this_ffr.dRsd()[ibin]),
						Rdu=format_R(this_ffr.Rdu()[ibin], this_ffr.dRdu()[ibin]),
						))
					'''
					ftable.write("\t\t{bin_str} & ${Ns:.1f}\\pm{dNs:.1f}$ & ${Nd:.1f}\\pm{dNd:.1f}$ & ${Nu:.1f}\\pm{dNu:.1f}$ & ${Rsu:.3f}\\pm{dRsu:.3f}$ & ${Rsd:.3f}\\pm{dRsd:.3f}$ & ${Rdu:.3f}\\pm{dRdu:.3f}$ \\\\ \n\t\t\\hline\n".format(
						bin_str=bin_str,
						Ns=this_ffr.N("Bs")[ibin],
						dNs=this_ffr.dN("Bs")[ibin],
						Nd=this_ffr.N("Bd")[ibin],
						dNd=this_ffr.dN("Bd")[ibin],
						Nu=this_ffr.N("Bu")[ibin],
						dNu=this_ffr.dN("Bu")[ibin],
						Rsu=this_ffr.Rsu()[ibin], 
						dRsu=this_ffr.dRsu()[ibin], 
						Rsd=this_ffr.Rsd()[ibin], 
						dRsd=this_ffr.dRsd()[ibin], 
						Rdu=this_ffr.Rdu()[ibin], 
						dRdu=this_ffr.dRdu()[ibin], 
						))
					'''
				ftable.write("\t\t{bin_str} & {Ns} & {Nd} & {Nu} & {Rsu} & {Rsd} & {Rdu} \\\\ \n\t\t\\hline\n".format(
					bin_str="Total",
					Ns=format_N(this_ffr.N_avg("Bs"), this_ffr.dN_avg("Bs")),
					Nd=format_N(this_ffr.N_avg("Bd"), this_ffr.dN_avg("Bd")),
					Nu=format_N(this_ffr.N_avg("Bu"), this_ffr.dN_avg("Bu")),
					Rsu=format_R(this_ffr.Rsu_avg(), this_ffr.dRsu_avg()),
					Rsd=format_R(this_ffr.Rsd_avg(), this_ffr.dRsd_avg()),
					Rdu=format_R(this_ffr.Rdu_avg(), this_ffr.dRdu_avg()),
					))
				'''
				ftable.write("\t\t{bin_str} & ${Ns:.1f}\\pm{dNs:.1f}$ & ${Nd:.1f}\\pm{dNd:.1f}$ & ${Nu:.1f}\\pm{dNu:.1f}$ & ${Rsu:.3f}\\pm{dRsu:.3f}$ & ${Rsd:.3f}\\pm{dRsd:.3f}$ & ${Rdu:.3f}\\pm{dRdu:.3f}$ \\\\ \n\t\t\\hline\n".format(
					bin_str="Avg.",
					Ns=this_ffr.N_avg("Bs"),
					dNs=this_ffr.dN_avg("Bs"),
					Nd=this_ffr.N_avg("Bd"),
					dNd=this_ffr.dN_avg("Bd"),
					Nu=this_ffr.N_avg("Bu"),
					dNu=this_ffr.dN_avg("Bu"),
					Rsu=this_ffr.Rsu_avg(), 
					dRsu=this_ffr.dRsu_avg(), 
					Rsd=this_ffr.Rsd_avg(), 
					dRsd=this_ffr.dRsd_avg(), 
					Rdu=this_ffr.Rdu_avg(), 
					dRdu=this_ffr.dRdu_avg(), 
					))
				'''
				if axis == "pt":
					axis_pretty = "$\\pt$"
				elif axis == "y":
					axis_pretty = "$|y|$"
				ftable.write("""
\t\\end{{tabular}}
\t}}
\t\\caption{{Corrected yields and yield ratios versus {}.}}
\t\\label{{table:ffrs-{}-{}}}
\\end{{table}}
""".format(axis_pretty, side, axis))



	# Make beamer presentation of results
	with open(f"/home/dyu7/BFrag/data/ffrs/Rtable_beamer_{args.selection}.tex", 'w') as ftable:
		for side in sides_to_run:
			for axis in ["pt", "y"]:
				ftable.write("""
\n
\\begin{frame}
\\begin{table}[htbp]
\t\\centering
\t\\resizebox{\\textwidth}{!}{
\t\\begin{tabular}{|l|c|c|c|c|c|c|}
\t\t\\hline
\t\tBin & $N^{\\mathrm{corr}}_{\\Bs}$ & $N^{\\mathrm{corr}}_{\\Bd}$ & $N^{\\mathrm{corr}}_{\\Bu}$ & $\\Rsu$ & $\\Rsd$ & $\\Rdu$ \\\\
\t\t\\hline""")
				if side == "tag":
					this_ffr = ffrs[axis]["tag"][True]["HLT_all"]
				elif side == "probe":
					this_ffr = ffrs[axis]["probe"][True]["HLT_all"]
				xbins = this_ffr.xbins()
				for ibin in range(len(xbins)-1):
					xleft = xbins[ibin]
					xright = xbins[ibin+1]
					if axis == "pt":
						bin_str = "{:d}-{:d}".format(int(xleft), int(xright))
					else:
						bin_str = "{:.2f}-{:.2f}".format(xleft, xright)
					ftable.write("\t\t{bin_str} & {Ns} & {Nd} & {Nu} & {Rsu} & {Rsd} & {Rdu} \\\\ \n\t\t\\hline\n".format(
						bin_str=bin_str,
						Ns=format_N(this_ffr.N("Bs")[ibin], this_ffr.dN("Bs")[ibin]),
						Nd=format_N(this_ffr.N("Bd")[ibin], this_ffr.dN("Bd")[ibin]),
						Nu=format_N(this_ffr.N("Bu")[ibin], this_ffr.dN("Bu")[ibin]),
						Rsu=format_R(this_ffr.Rsu()[ibin], this_ffr.dRsu()[ibin]),
						Rsd=format_R(this_ffr.Rsd()[ibin], this_ffr.dRsd()[ibin]),
						Rdu=format_R(this_ffr.Rdu()[ibin], this_ffr.dRdu()[ibin]),
						))
					'''
					ftable.write("\t\t{bin_str} & ${Ns:.1f}\\pm{dNs:.1f}$ & ${Nd:.1f}\\pm{dNd:.1f}$ & ${Nu:.1f}\\pm{dNu:.1f}$ & ${Rsu:.3f}\\pm{dRsu:.3f}$ & ${Rsd:.3f}\\pm{dRsd:.3f}$ & ${Rdu:.3f}\\pm{dRdu:.3f}$ \\\\ \n\t\t\\hline\n".format(
						bin_str=bin_str,
						Ns=this_ffr.N("Bs")[ibin],
						dNs=this_ffr.dN("Bs")[ibin],
						Nd=this_ffr.N("Bd")[ibin],
						dNd=this_ffr.dN("Bd")[ibin],
						Nu=this_ffr.N("Bu")[ibin],
						dNu=this_ffr.dN("Bu")[ibin],
						Rsu=this_ffr.Rsu()[ibin], 
						dRsu=this_ffr.dRsu()[ibin], 
						Rsd=this_ffr.Rsd()[ibin], 
						dRsd=this_ffr.dRsd()[ibin], 
						Rdu=this_ffr.Rdu()[ibin], 
						dRdu=this_ffr.dRdu()[ibin], 
						))
					'''
				ftable.write("\t\t{bin_str} & {Ns} & {Nd} & {Nu} & {Rsu} & {Rsd} & {Rdu} \\\\ \n\t\t\\hline\n".format(
					bin_str="Total",
					Ns=format_N(this_ffr.N_avg("Bs"), this_ffr.dN_avg("Bs")),
					Nd=format_N(this_ffr.N_avg("Bd"), this_ffr.dN_avg("Bd")),
					Nu=format_N(this_ffr.N_avg("Bu"), this_ffr.dN_avg("Bu")),
					Rsu=format_R(this_ffr.Rsu_avg(), this_ffr.dRsu_avg()),
					Rsd=format_R(this_ffr.Rsd_avg(), this_ffr.dRsd_avg()),
					Rdu=format_R(this_ffr.Rdu_avg(), this_ffr.dRdu_avg()),
					))
				'''
				ftable.write("\t\t{bin_str} & ${Ns:.1f}\\pm{dNs:.1f}$ & ${Nd:.1f}\\pm{dNd:.1f}$ & ${Nu:.1f}\\pm{dNu:.1f}$ & ${Rsu:.3f}\\pm{dRsu:.3f}$ & ${Rsd:.3f}\\pm{dRsd:.3f}$ & ${Rdu:.3f}\\pm{dRdu:.3f}$ \\\\ \n\t\t\\hline\n".format(
					bin_str="Avg.",
					Ns=this_ffr.N_avg("Bs"),
					dNs=this_ffr.dN_avg("Bs"),
					Nd=this_ffr.N_avg("Bd"),
					dNd=this_ffr.dN_avg("Bd"),
					Nu=this_ffr.N_avg("Bu"),
					dNu=this_ffr.dN_avg("Bu"),
					Rsu=this_ffr.Rsu_avg(), 
					dRsu=this_ffr.dRsu_avg(), 
					Rsd=this_ffr.Rsd_avg(), 
					dRsd=this_ffr.dRsd_avg(), 
					Rdu=this_ffr.Rdu_avg(), 
					dRdu=this_ffr.dRdu_avg(), 
					))
				'''
				if axis == "pt":
					axis_pretty = "$\\pt$"
				elif axis == "y":
					axis_pretty = "$|y|$"
				ftable.write("""
\t\\end{{tabular}}
\t}}
\t\\caption{{Corrected yields and yield ratios versus {}.}}
\t\\label{{table:ffrs-{}-{}}}
\\end{{table}}
\\end{{frame}}
""".format(axis_pretty, side, axis))

	pprint(ffrs)