import os
import sys
import pickle
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
from brazil.seaborn_colors import SeabornColors
seaborn_colors = SeabornColors()
palette_dir = os.path.expandvars("/home/dryu/BFrag/boffea/brazil/seaborn_palettes")
seaborn_colors.load_palette("Blues_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Reds_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Oranges_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Greens_d", palette_dir=palette_dir)
seaborn_colors.load_palette("Purples_d", palette_dir=palette_dir)
seaborn_colors.load_palette("RdPu_r", palette_dir=palette_dir)
seaborn_colors.load_palette("hls", palette_dir=palette_dir)
seaborn_colors.load_palette("hls_light", palette_dir=palette_dir)

sys.path.append("./fitting")
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
def style_graph(graph, btype):
	graph.SetMarkerStyle(style[btype]["marker_style"])
	graph.SetMarkerColor(style[btype]["marker_color"])
	graph.SetLineColor(style[btype]["line_color"])
	graph.SetLineWidth(style[btype]["line_width"])

class FFRData:
	def __init__(self, axis="pt", side="probe", trigger_strategy="HLT_all", binned=False):
		self._axis = axis
		self._side = side
		self._binned = binned
		self._trigger_strategy = trigger_strategy

	def load_yields(self):
		self._yields = {}
		self._dyields = {}
		yield_files = {
			"Bs": {
				"binned": "/home/dryu/BFrag/boffea/barista/fitting/Bs/yields_binned.pkl",
				"unbinned": "/home/dryu/BFrag/boffea/barista/fitting/Bs/yields.pkl",
			},
			"Bu": {
				"binned": "/home/dryu/BFrag/boffea/barista/fitting/Bu/yields_binned.pkl",
				"unbinned": "/home/dryu/BFrag/boffea/barista/fitting/Bu/yields.pkl",
			},
			"Bd": {
				"binned": "/home/dryu/BFrag/boffea/barista/fitting/Bd/yields_binned.pkl",
				"unbinned": "/home/dryu/BFrag/boffea/barista/fitting/Bd/yields.pkl",
			},
		}
		yields_tmp = {}
		cuts_tmp = []
		for btype in ["Bu", "Bd", "Bs"]:
			if self._binned:
				yield_file = yield_files[btype]["binned"]
			else:
				yield_file = yield_files[btype]["unbinned"]
			with open(yield_file, "rb") as f:
				yields_tmp[btype] = pickle.load(f)[self._side][self._trigger_strategy]
			cuts_tmp.extend(yields_tmp[btype].keys())
		cuts_tmp = list(set(cuts_tmp))
		if self._axis == "pt":
			cuts_tmp = [x for x in cuts_tmp if "ptbin" in x]
		elif self._axis == "y":
			cuts_tmp = [x for x in cuts_tmp if "ybin" in x]
		cuts_tmp.sort(key=lambda x: cut_xvals[x][0])

		# Make numpy arrays: x bin boundaries, y=yields
		self._xbins = np.array(sorted(list(set([cut_xvals[x][0] for x in cuts_tmp] + [cut_xvals[x][1] for x in cuts_tmp]))))
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
		with open("/home/dryu/BFrag/data/efficiency/efficiency2.pkl", "rb") as f:
			eff_tmp = pickle.load(f) # eff_deff[btype][side][trigger_strategy]
		self._efficiencies = {}
		self._defficiencies = {}

		# For probe side, load total efficiencies (including probefilter; use for yields, but not ratio!)
		if self._side == "probe":
			self._total_efficiencies = {}
			self._dtotal_efficiencies = {}

		for btype in ["Bu", "Bd", "Bs"]:
			self._efficiencies[btype] = np.array([x[0] for x in eff_tmp[self._axis][btype][self._side][self._trigger_strategy]])
			self._defficiencies[btype] = np.array([x[1] for x in eff_tmp[self._axis][btype][self._side][self._trigger_strategy]])
			if self._side == "probe":
				self._total_efficiencies[btype] = np.array([x[0] for x in eff_tmp[self._axis][btype]["probe_total"][self._trigger_strategy]])
				self._dtotal_efficiencies[btype] = np.array([x[1] for x in eff_tmp[self._axis][btype]["probe_total"][self._trigger_strategy]])
		#print("Efficiencies:")
		#pprint(self._efficiencies)

	def finalize(self):
		self._n = {}
		self._dn = {}
		self._total_n = {}
		self._dtotal_n = {}

		for btype in ["Bu", "Bd", "Bs"]:
			self._n[btype] = self._yields[btype] / self._efficiencies[btype]
			self._dn[btype] = self._n[btype] * np.sqrt(
				(self._dyields[btype] / self._yields[btype])**2
				+ (self._defficiencies[btype] / self._efficiencies[btype])**2)

			if self._side == "probe":
				self._total_n[btype] = self._yields[btype] / self._total_efficiencies[btype]
				self._dtotal_n[btype] = self._total_n[btype] * np.sqrt(
					(self._dyields[btype] / self._yields[btype])**2
					+ (self._dtotal_efficiencies[btype] / self._total_efficiencies[btype])**2)
			else:
				self._total_n[btype] = self._n[btype]
				self._dtotal_n[btype] = self._dn[btype]

		BR_phiKK = 0.492
		BR_KstarKpi = 0.665
		BR_BuKJpsi = 1.006e-3
		BR_BdKstarJpsi = 1.27e-3
		BR_BsPhiJpsi = 1.08e-3
		self._Rsu = self._n["Bs"] / self._n["Bu"] / BR_phiKK
		self._dRsu = self._Rsu * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)

		self._Rsd = self._n["Bs"] / self._n["Bd"] / BR_phiKK * BR_KstarKpi
		self._dRsd = self._Rsd * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bd"] / self._n["Bd"])**2)

		self._Rdu = self._n["Bd"] / self._n["Bu"] / BR_KstarKpi
		self._dRdu = self._Rdu * np.sqrt((self._dn["Bd"] / self._n["Bd"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)

		self._fsfu = self._n["Bs"] / self._n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
		self._dfsfu = self._fsfu * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)

		self._fsfd = self._n["Bs"] / self._n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
		self._dfsfd = self._fsfd * np.sqrt((self._dn["Bs"] / self._n["Bs"])**2 + (self._dn["Bd"] / self._n["Bd"])**2)

		self._fdfu = self._n["Bd"] / self._n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
		self._dfdfu = self._fdfu * np.sqrt((self._dn["Bd"] / self._n["Bd"])**2 + (self._dn["Bu"] / self._n["Bu"])**2)

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

	def N(self, btype):
		return self._n[btype]

	def dN(self, btype):
		return self._dn[btype]

	def TotalN(self, btype):
		return self._total_n[btype]

	def dTotalN(self, btype):
		return self._dtotal_n[btype]

class FFRPlot:
	def __init__(self, ffrs, legend_entries, 
		save_tag="test",
		xlabel="",
		xlim=[0., 30.],
		ylim=[0., 0.4]
		):
		print(save_tag)
		self._save_tag = save_tag
		self._ffrs = ffrs
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
			bin_centers = 0.5 * (ffr.xbins()[1:] + ffr.xbins()[:-1])
			bin_widths = 0.5 * (ffr.xbins()[1:] - ffr.xbins()[:-1])
			#print(bin_centers)
			#print(bin_widths)
			#print(ffr.Rsu())
			#print(ffr.dRsu())
			ax_su.errorbar(
				x=bin_centers, 
				y=ffr.Rsu(),
				xerr=bin_widths,
				yerr=ffr.dRsu(), 
				marker=".", 
				markersize=10.,
				color=colorwheel[i],
				label=legend_entries[i],
				ls="none",
				ecolor=colorwheel[i],
				elinewidth=1,
				)
		ax_su.set_xlabel(self._xlabel, fontsize=18)
		ax_su.set_xlim(self._xlim)
		ax_su.set_ylim(self._ylim)
		ax_su.set_ylabel(r"$\mathcal{R}_{su}$", fontsize=18)
		ax_su.xaxis.set_ticks_position("both")
		ax_su.yaxis.set_ticks_position("both")
		ax_su.tick_params(direction="in")
		ax_su.legend()
		fig_su.savefig(f"/home/dryu/BFrag/data/ffrs/Rsu_{save_tag}.png")

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
		ax_sd.set_xlabel(self._xlabel)
		ax_sd.set_xlim(self._xlim)
		ax_sd.set_ylim(self._ylim)
		ax_sd.set_ylabel(r"$\mathcal{R}_{su}$")
		ax_sd.xaxis.set_ticks_position("both")
		ax_sd.yaxis.set_ticks_position("both")
		ax_sd.tick_params(direction="in")
		ax_sd.legend()
		fig_sd.savefig(f"/home/dryu/BFrag/data/ffrs/Rsd_{save_tag}.png")

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
		ax_du.set_xlabel(self._xlabel)
		ax_du.set_xlim(self._xlim)
		ax_du.set_ylim(self._ylim)
		ax_du.set_ylabel(r"$\mathcal{R}_{su}$")
		ax_du.xaxis.set_ticks_position("both")
		ax_du.yaxis.set_ticks_position("both")
		ax_du.tick_params(direction="in")
		ax_du.legend()
		fig_du.savefig(f"/home/dryu/BFrag/data/ffrs/Rdu_{save_tag}.png")


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
		ax_fsfu.set_xlabel(self._xlabel)
		ax_fsfu.set_xlim(self._xlim)
		ax_fsfu.set_ylim(self._ylim)
		ax_fsfu.set_ylabel(r"$f_{s}/f_{u}$")
		ax_fsfu.xaxis.set_ticks_position("both")
		ax_fsfu.yaxis.set_ticks_position("both")
		ax_fsfu.tick_params(direction="in")
		ax_fsfu.legend()
		fig_fsfu.savefig(f"/home/dryu/BFrag/data/ffrs/fsfu_{save_tag}.png")

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
		ax_fsfd.set_xlabel(self._xlabel)
		ax_fsfd.set_xlim(self._xlim)
		ax_fsfd.set_ylim(self._ylim)
		ax_fsfd.set_ylabel(r"f_{s}/f_{d}$")
		ax_fsfd.xaxis.set_ticks_position("both")
		ax_fsfd.yaxis.set_ticks_position("both")
		ax_fsfd.tick_params(direction="in")
		ax_fsfd.legend()
		fig_fsfd.savefig(f"/home/dryu/BFrag/data/ffrs/fsfd_{save_tag}.png")

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
		ax_fdfu.set_xlabel(self._xlabel)
		ax_fdfu.set_xlim(self._xlim)
		ax_fdfu.set_ylim(self._ylim)
		ax_fdfu.set_ylabel(r"f_{s}/f_{d}$")
		ax_fdfu.xaxis.set_ticks_position("both")
		ax_fdfu.yaxis.set_ticks_position("both")
		ax_fdfu.tick_params(direction="in")
		ax_fdfu.legend()
		fig_fdfu.savefig(f"/home/dryu/BFrag/data/ffrs/fdfu_{save_tag}.png")

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
		#print(maxy)
		ax_yield.set_ylim([1.e3, 1.e8])
		ax_yield.set_ylabel(r"Corrected yield / GeV")
		ax_yield.xaxis.set_ticks_position("both")
		ax_yield.yaxis.set_ticks_position("both")
		ax_yield.tick_params(direction="in")
		ax_yield.legend()
		fig_yield.savefig(f"/home/dryu/BFrag/data/yields/corr_yields_{self._save_tag}.png")


if __name__ == "__main__":
	ffrs = {}
	for axis in ["pt", "y"]:
		ffrs[axis] = {}
		for side in ["tag", "probe"]:
			ffrs[axis][side] = {}
			for binned in [True, False]:
				ffrs[axis][side][binned] = {}
				for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"]:
					print(f"Computing FFRs for {axis} {side} binned={binned} {trigger_strategy}")
					ffrs[axis][side][binned][trigger_strategy] = FFRData(axis=axis, side=side, binned=binned, trigger_strategy=trigger_strategy)
					ffrs[axis][side][binned][trigger_strategy].load_yields()
					ffrs[axis][side][binned][trigger_strategy].load_efficiencies()
					ffrs[axis][side][binned][trigger_strategy].finalize()

	# FFR plot comparing tag and probe
	xlims = {"pt": [0, 45], "y": [0, 2.5]}
	xlabels = {"pt": r"$p_{T}$ [GeV]", "y": r"$|y|$"}
	for axis in ["pt"]:#, "y"]:
		for binned in [True, False]:
			if binned:
				binned_str = "binned"
			else:
				binned_str = "unbinned"
			for trigger_strategy in ["HLT_all", "HLT_Mu9", "HLT_Mu7"]:
				print(f"Plotting FFRs for {axis} binned={binned} {trigger_strategy}")
				ffrplot = FFRPlot([ffrs[axis]["tag"][binned][trigger_strategy], ffrs[axis]["probe"][binned][trigger_strategy]], 
							legend_entries=["Tag", "Probe"],
							xlabel=xlabels[axis],
							xlim=xlims[axis],
							ylim=[0., 0.4],
							save_tag=f"{axis}_{binned_str}_{trigger_strategy}"
					)
				ffrplot.corrected_yield_plot()
				for side in ["tag", "probe"]:
					print(f"Relative {side} uncertainties:")
					print("Rsu = ")
					print(ffrs[axis][side][binned][trigger_strategy].Rsu())
					print("dRsu = ")
					print(ffrs[axis][side][binned][trigger_strategy].dRsu())
					print("dRsu / Rsu = ")
					print(ffrs[axis][side][binned][trigger_strategy].dRsu() / ffrs[axis][side][binned][trigger_strategy].Rsu())

	for btype in ["Bu", "Bd", "Bs"]:
		for side in ["tag", "probe"]:
			print(f"Corrected yields {side} {btype} = {ffrs['pt'][side][True]['HLT_all'].N(btype)} +/- {ffrs['pt'][side][True]['HLT_all'].dN(btype)}")