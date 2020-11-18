import os
import sys
from pprint import pprint
import pickle
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
		"marker_style": 21, 
		"marker_color": seaborn_colors.get_root_color("Greens_d", 2),
		"line_color": seaborn_colors.get_root_color("Greens_d", 3),
		"line_width": 1,
	},
	"Bs":{
		"marker_style": 24, 
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

# Load yields from RooFit result
def load_yields(btype, binned=False, correct_eff=False):
	yields_file = f"/home/dryu/BFrag/boffea/barista/fitting/{btype}/yields"
	if binned:
		yields_file += "_binned"
	if correct_eff:
		yields_file += "_correcteff"
	yields_file += ".pkl"
	with open(yields_file, "rb") as f:
		yields = pickle.load(f)
	return yields

def yield_pt_graph(yields, normalize_bin_width=None):
	#print(yields)
	cut_names = [x for x in yields.keys() if "ptbin" in x]
	cut_names.sort(key=lambda x: cut_xvals[x][0])
	tg = ROOT.TGraphErrors(len(cut_names))
	for ipoint, cut_name in enumerate(cut_names):
		this_yield = yields[cut_name][0]
		this_dyield = yields[cut_name][1]
		bin_width = (cut_xvals[cut_name][1] - cut_xvals[cut_name][0])
		if normalize_bin_width:
			normalization_factor = bin_width / normalize_bin_width
		else:
			normalization_factor = 1.0
		tg.SetPoint(ipoint, 0.5*(cut_xvals[cut_name][0] + cut_xvals[cut_name][1]), this_yield / normalization_factor)
		tg.SetPointError(ipoint, 0.5*bin_width, this_dyield / normalization_factor)
	return tg

def yield_pt_plot(yields, save_tag, normalize_bin_width=False, correct_eff=False):
	print("\n*** Making pT plot {} ***".format(save_tag))
	canvas = ROOT.TCanvas(f"c_yield_pt_{save_tag}", f"c_yield_pt_{save_tag}", 800, 600)
	canvas.SetLeftMargin(0.1)
	pprint(yields)
	if "probe" in save_tag:
		legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.88)
	else:
		legend = ROOT.TLegend(0.58, 0.7, 0.88, 0.88)
	legend.SetFillColor(0)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)

	graphs = {}
	for btype, yields_btype in yields.items():
		graphs[btype] = yield_pt_graph(yields_btype, normalize_bin_width=normalize_bin_width)

	graph_max = max([max(x.GetY()) for x in graphs.values()])
	#print(f"graph_max={graph_max}")
	frame = ROOT.TH1D("frame", "frame", 100, 0., 45.)
	frame.GetXaxis().SetTitle("p_{T} [GeV]")
	if normalize_bin_width:
		frame.GetYaxis().SetTitle("Event / {} GeV".format(normalize_bin_width))
	else:
		frame.GetYaxis().SetTitle("Events per bin")
	frame.SetMaximum(graph_max * 1.2)
	frame.SetMinimum(0.)
	frame.Draw("axis")

	for btype in ["Bu", "Bd", "Bs"]:
		style_graph(graphs[btype], btype)
		graphs[btype].Draw("p")
		legend.AddEntry(graphs[btype], legend_entries[btype], "pl")
	legend.Draw()

	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}.png")
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}.pdf")

	# Log y
	canvas.SetLogy()
	frame.SetMinimum(0.5)
	frame.SetMaximum(graph_max * 50.)
	frame.Draw("axis")
	for btype in ["Bu", "Bd", "Bs"]:
		graphs[btype].Draw("p")
	legend.Draw()
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}_logy.png")
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}_logy.pdf")

def yield_y_graph(yields):
	cut_names = [x for x in yields.keys() if "ybin" in x]
	cut_names.sort(key=lambda x: cut_xvals[x][0])
	tg = ROOT.TGraphErrors(len(cut_names))
	print(cut_xvals)
	for ipoint, cut_name in enumerate(cut_names):
		this_yield = yields[cut_name][0]
		this_dyield = yields[cut_name][1]
		bin_width = 0.5*(cut_xvals[cut_name][1] - cut_xvals[cut_name][0])
		tg.SetPoint(ipoint, 0.5*(cut_xvals[cut_name][0] + cut_xvals[cut_name][1]), this_yield)
		tg.SetPointError(ipoint, bin_width, this_dyield)
	return tg

def yield_y_plot(yields, save_tag, correct_eff=False):
	canvas = ROOT.TCanvas(f"c_yield_y_{save_tag}", f"c_yield_y_{save_tag}", 800, 600)
	legend = ROOT.TLegend(0.6, 0.62, 0.88, 0.87)
	legend.SetFillColor(0)
	legend.SetBorderSize(0)

	graphs = {}
	for btype, yields_btype in yields.items():
		graphs[btype] = yield_y_graph(yields_btype)

	graph_max = max([max(x.GetY()) for x in graphs.values()])
	frame = ROOT.TH1D("frame", "frame", 100, 0., 3.0)
	frame.GetXaxis().SetTitle("y")
	frame.GetYaxis().SetTitle("Bin yields")
	frame.SetMaximum(graph_max * 1.2)
	frame.SetMinimum(0.)
	frame.Draw("axis")

	for btype in ["Bu", "Bd", "Bs"]:
		style_graph(graphs[btype], btype)
		graphs[btype].Draw("p")
		legend.AddEntry(graphs[btype], legend_entries[btype], "pl")
	legend.Draw()

	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}.png")
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}.pdf")

	# Log y
	canvas.SetLogy()
	frame.SetMinimum(0.5)
	frame.SetMaximum(graph_max * 10.)
	frame.Draw("axis")
	for btype in ["Bu", "Bd", "Bs"]:
		graphs[btype].Draw("p")
	legend.Draw()
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}_logy.png")
	canvas.SaveAs(f"/home/dryu/BFrag/data/yields/{canvas.GetName()}_logy.pdf")



if __name__ == "__main__":
	for binned in [True, False]:#, False]:
		for correct_eff in [False]: # False
			for side in ["tag", "probe"]:
				for trigger_strategy in ['HLT_all', 'HLT_Mu7', 'HLT_Mu9']:
					yields = {}
					for btype in ["Bu", "Bd", "Bs"]:
						yields[btype] = load_yields(btype, binned)[side][trigger_strategy]
					save_tag = f"{side}_{trigger_strategy}"
					if binned:
						save_tag += "_binned"
					if correct_eff:
						save_tag += "_correcteff"

					if side == "tag":
						normalize_bin_width = 1.0
					else:
						normalize_bin_width = False
					yield_pt_plot(yields, save_tag=save_tag, correct_eff=correct_eff, normalize_bin_width=normalize_bin_width)
					yield_y_plot(yields, save_tag=save_tag, correct_eff=correct_eff)
