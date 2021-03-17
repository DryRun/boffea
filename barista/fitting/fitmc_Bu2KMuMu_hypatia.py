'''
Fit MC mass distributions
'''
from pprint import pprint
import ROOT
from brazil.aguapreta import *
import pickle
ROOT.gROOT.SetBatch(True)

#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
#from ROOT import TripleGaussian
#ROOT.gSystem.Load("include/TripleGaussianPdf2.so")
#from ROOT import TripleGaussianPdf2

figure_dir = "/home/dryu/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW_MC, BD_FIT_WINDOW_MC, BS_FIT_WINDOW_MC, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeHypatia2
rcache = []


def plot_mc(tree, mass_range=BU_FIT_WINDOW_MC, cut="", tag=""):
	h_mc = ROOT.TH1D("h_mc", "h_mc", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_mc", cut)
	c = ROOT.TCanvas("c_mc_{}".format(tag), "c_mc_{}".format(tag), 800, 600)
	h_mc.SetMarkerStyle(20)
	h_mc.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_mc.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/mc/{}.pdf".format(c.GetName()))

import re
re_ptcut = re.compile("\(pt \> (?P<ptmin>\d+\.\d+)\) && \(pt < (?P<ptmax>\d+\.\d+)\)")
def fit_mc(tree, mass_range=BU_FIT_WINDOW_MC, incut="1", cut_name="inclusive", binned=False):
	ws = ROOT.RooWorkspace('ws')

	# Slightly widen pT window, to min of 5 GeV
	if "ptbin" in cut_name:
		re_match = re_ptcut.search(incut)
		ptmin = float(re_match.group("ptmin"))
		ptmax = float(re_match.group("ptmax"))
		if ptmax - ptmin < 5.0:
			extra_pt = 5.0 - (ptmax - ptmin)
			incut = incut.replace("pt > {}".format(ptmin), "pt > {}".format(ptmin - extra_pt / 2.))
			incut = incut.replace("pt < {}".format(ptmax), "pt < {}".format(ptmax + extra_pt / 2.))
	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitMC", "fitMC", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Optional: bin data
	if binned:
		mass.setBins(BU_FIT_NBINS)
		rdatahist = ROOT.RooDataHist("fitMCBinned", "fitMCBinned", ROOT.RooArgSet(mass), rdataset)
		rdata = rdatahist
	else:
		rdata = rdataset

	# Signal: Hypatia function
	signal_hyp = MakeHypatia2(ws, mass_range, rcache=rcache)
	getattr(ws, "import")(signal_hyp, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_hyp, nsignal)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))

	# Tweaks
	if cut_name == "ptbin_8p0_13p0":
		ws.var('hyp_a').setVal(7.275745654055582)
		ws.var('hyp_lambda').setVal(-1.0067107376670137)
		ws.var('hyp_mu').setVal(5.279661966888891)
		ws.var('hyp_n').setVal(8.265412496047785)
		ws.var('hyp_sigma').setVal(0.025528613146354573)
	elif cut_name == "ptbin_15p0_16p0":
		ws.var('hyp_a').setVal(62.110822807767384)
		ws.var('hyp_lambda').setVal(-1.2472323943707782)
		ws.var('hyp_mu').setVal(5.280327479922378)
		ws.var('hyp_n').setVal(7.375310140389502e-07)
		ws.var('hyp_sigma').setVal(0.026784749158424062)
		ws.var('nsignal').setVal(3761.96071872678)
	elif cut_name == "ptbin_18p0_23p0" \
		or cut_name == "ptbin_18p0_20p0" \
		or cut_name == "ptbin_16p0_18p0":
		ws.var('hyp_a').setVal(9.635504664390693)
		ws.var('hyp_lambda').setVal(-1.1403319564083407)
		ws.var('hyp_mu').setVal(5.279257602044328)
		ws.var('hyp_n').setVal(5.606733613136833)
		ws.var('hyp_sigma').setVal(0.024140061131064254)
	elif cut_name == "ybin_1p25_1p5":
		ws.var('hyp_a').setVal(40.92480088013374)
		ws.var('hyp_lambda').setVal(-1.468838761142754)
		ws.var('hyp_mu').setVal(5.279297279329609)
		ws.var('hyp_n').setVal(0.6721768715035447)
		ws.var('hyp_sigma').setVal(0.04327205409255316)

	elif cut_name == "ybin_1p75_2p0":
		ws.var('hyp_a').setVal(3.0569513567559827)
		ws.var('hyp_lambda').setVal(-1.4835065290891034)
		ws.var('hyp_mu').setVal(5.276668272043223)
		ws.var('hyp_n').setVal(1.5467417558961083)
		ws.var('hyp_n').setMin(1.0)
		ws.var('hyp_sigma').setVal(0.058803608523859206)


	# Perform fit
	##nll = model.createNLL(rdata, ROOT.RooFit.NumCPU(8))
	#minimizer = ROOT.RooMinuit(nll)
	#minimizer.migrad()
	#minimizer.minos()
	fit_result = model.fitTo(rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())
	fit_result.Print()

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	if binned:
		getattr(ws, "import")(rdataset)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, tag="", text=None, binned=False):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	#rdata = ws.data("fitMC")
	if binned:
		rdata = ws.data("fitMCBinned")
	else:
		rdata = ws.data("fitMC")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_mcfit_{}".format(tag), "c_mcfit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()
	top.SetLogy()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdata.plotOn(rplot, ROOT.RooFit.Name("data"))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(0.06)
	rplot.GetYaxis().SetTitleSize(0.06)
	rplot.GetYaxis().SetTitleOffset(0.8)
	rplot.Draw()

	l = ROOT.TLegend(0.7, 0.45, 0.88, 0.88)
	l.SetFillColor(0)
	l.SetBorderSize(0)
	l.AddEntry("data", "Data", "lp")
	l.AddEntry("fit", "Total fit", "l")
	l.AddEntry("signal", "B_{u} #rightarrow J/#psi K", "lf")
	l.Draw()
	if text:
		textbox = ROOT.TLatex(0.15, 0.7, text)
		textbox.SetNDC()
		textbox.Draw()

	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.5)
	bottom.SetTopMargin(0.02)
	bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	binning = ROOT.RooBinning(100, BU_FIT_WINDOW_MC[0], BU_FIT_WINDOW_MC[1])
	data_hist = ROOT.RooAbsData.createHistogram(rdata, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -7
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		pull_val = (data_val - fit_val) / max(math.sqrt(fit_val), 1.e-10)
		pull_hist.SetBinContent(xbin, pull_val)
		pull_hist.SetBinError(xbin, 1)
		chi2 += pull_val**2
		ndf += 1
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}} [GeV]")
	pull_hist.GetYaxis().SetTitle("Pull w.r.t. fit [#sigma]")
	pull_hist.SetMarkerStyle(20)
	pull_hist.SetMarkerSize(1)
	pull_hist.GetXaxis().SetTitleSize(0.06)
	pull_hist.GetXaxis().SetLabelSize(0.06)
	pull_hist.GetYaxis().SetTitleSize(0.06)
	pull_hist.GetYaxis().SetLabelSize(0.06)
	pull_hist.GetYaxis().SetTitleOffset(0.6)
	pull_hist.SetMinimum(-5.)
	pull_hist.SetMaximum(5.)
	pull_hist.Draw("p")

	zero = ROOT.TLine(BU_FIT_WINDOW_MC[0], 0., BU_FIT_WINDOW_MC[1], 0.)
	zero.SetLineColor(ROOT.kGray)
	zero.SetLineStyle(3)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	chi2text = ROOT.TLatex(0.15, 0.6, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	chi2text.SetNDC()
	chi2text.Draw()

	canvas.cd()
	canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
	canvas.SaveAs("{}/{}.pdf".format(figure_dir, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)

def extract_yields(ws):
	return ws.var("nsignal").getVal()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	args = parser.parse_args()

	import glob
	mc_files = glob.glob("MCEfficiency_Bu.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = list(set(fit_cuts["tag"] + fit_cuts["probe"]))
	elif args.some:
		cuts = args.some.split(",")
		if not set(cuts).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))

	if args.fits:
		chain = ROOT.TChain("Bcands_recomatch_Bu2KJpsi2KMuMu_probefilter")
		for mc_file in mc_files:
			chain.Add(mc_file)

		for cut_name in cuts:
			print("\n*** Fitting {} ***".format(cut_name))
			cut_str = cut_strings[cut_name]
			#plot_mc(chain, cut=cut_str, tag="Bu_{}".format(cut_name))

			ws, fit_result = fit_mc(chain, incut=cut_str, cut_name=cut_name)
			ws.Print()
			#print("DEBUG : fit result = ")
			fit_result.Print()
			#print("DEBUG : writing to " + "Bu/fitws_hyp_mc_Bu_{}.root".format(cut_name))
			ws_file = ROOT.TFile("Bu/fitws_hyp_mc_Bu_{}.root".format(cut_name), "RECREATE")
			ws.Write()
			fit_result.Write()
			ws_file.Close()

			print("\nDone fitting {}\n".format(cut_name))

	if args.plots:
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bu/fitws_hyp_mc_Bu_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			plot_fit(ws, tag="Bu_hyp_{}".format(cut_name), text=fit_text[cut_name])

	if args.tables:
		print("\n\n*** Printing tables ***\n")
		yields = {}
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bu/fitws_hyp_mc_Bu_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			yields[cut_name] = extract_yields(ws)
		pprint(yields)

		print("\n\n*** Printing fit results ***\n")
		final_params = {}
		final_errors = {}
		parnames = []
		final_params = {}
		final_errors = {}
		for cut_name in cuts:
			final_params[cut_name] = {}
			final_errors[cut_name] = {}
			print("{}".format(cut_name))
			ws_file = ROOT.TFile("Bu/fitws_hyp_mc_Bu_{}.root".format(cut_name), "READ")
			#ws = ws_file.Get("ws")
			#ws.Print()
			fit_result = ws_file.Get("fitresult_model_fitMC")
			if not fit_result:
				print("WARNING : Didn't find fit result for {}".format(cut_name))
			this_final_params = fit_result.floatParsFinal()
			if cut_name == "ptbin_23p0_26p0":
				print("DEBUG : Loading fit parameters for ptbin_23p0_26p0")
				fit_result.Print()
				this_final_params.Print()
			covm = fit_result.covarianceMatrix()
			for i in range(this_final_params.getSize()):
				parname = this_final_params[i].GetName()
				if not parname in parnames:
					parnames.append(parname)
				final_params[cut_name][parname] = this_final_params[i].getVal()
				final_errors[cut_name][parname] = this_final_params[i].getError()
		print("Printing parameters and errors:")
		for cut_name in cuts:
			for parname in final_params[cut_name]:
				print(f"{cut_name} \t {parname} = {final_params[cut_name][parname]} +/- {final_errors[cut_name][parname]}")
		#pprint(final_params)
		#pprint(final_errors)
		if args.all:
			with open("Bu/fitparams_MC_Bu_hypatia.pkl", "wb") as f:
				pickle.dump(final_params, f)
			with open("Bu/fiterrs_MC_Bu_hypatia.pkl", "wb") as f:
				pickle.dump(final_errors, f)
