'''
Fit MC mass distributions
'''
import pickle
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
#from ROOT import TripleGaussian
ROOT.gSystem.Load("include/TripleGaussianPdf.so")
from ROOT import TripleGaussianPdf

figure_dir = "/home/dryu/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS

# Enums for PDFs
from enum import Enum
class BackgroundModel_t(Enum):
	kPolynomial = 1, 
	kChebychev = 2,
	kExp = 3

class SignalModel_t(Enum):
	kGaussian = 1,
	kDoubleGaussian = 2,
	kCB = 3

def pyerfc(x, par):
	erfc_arg = (x - par[0]) / par[1]
	return ROOT.TMath.Erfc(erfc_arg)

def plot_mc(tree, mass_range=BS_FIT_WINDOW, cut="", tag=""):
	h_mc = ROOT.TH1D("h_mc", "h_mc", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_mc", cut)
	c = ROOT.TCanvas("c_mc_{}".format(tag), "c_mc_{}".format(tag), 800, 600)
	h_mc.SetMarkerStyle(20)
	h_mc.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{+}K^{-}} [GeV]")
	h_mc.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/mc/{}.pdf".format(c.GetName()))


def fit_mc(tree, mass_range=BS_FIT_WINDOW, incut="1"):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitMC", "fitMC", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Signal: triple Gaussian
	mean = ws.factory(f"mean[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
	sigma1 = ws.factory("sigma1[0.005, 0.0005, 1.0]")
	sigma2 = ws.factory("sigma2[0.02, 0.0005, 1.0]")
	sigma3 = ws.factory("sigma3[0.05, 0.0005, 1.0]")
	aa = ws.factory(f"aa[0.6, 0.001, 1.0]")
	bb = ws.factory(f"bb[0.6, 0.001, 1.0]")
	#signal_tg = ws.factory(f"GenericPdf::signal_tg('TripleGaussian(mass, mean, sigma1, sigma2, sigma3, aa, bb)', {{mass, mean, sigma1, sigma2, sigma3, aa, bb}})")
	signal_tg = TripleGaussianPdf("signal_tg", "signal_tg", mass, mean, sigma1, sigma2, sigma3, aa, bb)
	getattr(ws, "import")(signal_tg, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_tg, nsignal)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))

	# Perform fit
	##nll = model.createNLL(rdataset, ROOT.RooFit.NumCPU(8))
	#minimizer = ROOT.RooMinuit(nll)
	#minimizer.migrad()
	#minimizer.minos()
	fit_result = model.fitTo(rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, tag="", text=None):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	rdataset = ws.data("fitMC")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_mcfit_{}".format(tag), "c_mcfit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()
	top.SetLogy()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
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
	l.AddEntry("signal", "B_{s}#rightarrowJ/#psi #phi", "lf")
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

	binning = ROOT.RooBinning(100, BS_FIT_WINDOW[0], BS_FIT_WINDOW[1])
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
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
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}K^{#mp}} [GeV]")
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

	zero = ROOT.TLine(BS_FIT_WINDOW[0], 0., BS_FIT_WINDOW[1], 0.)
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
	mc_files = glob.glob("MCEfficiency_Bs.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = list(set(fit_cuts["tag"] + fit_cuts["probe"]))
	elif args.some:
		cuts = args.some.split(",")
		if not set(cuts).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))

	if args.fits:
		chain = ROOT.TChain("Bcands_recomatch_Bs2PhiJpsi2KKMuMu_probefilter")
		#chain.Add("data_Run2018C_part3.root")
		#chain.Add("data_Run2018D_part1.root")
		#chain.Add("data_Run2018D_part2.root")
		for mc_file in mc_files:
			chain.Add(mc_file)

		for cut_name in cuts:
			cut_str = cut_strings[cut_name]
			plot_mc(chain, cut=cut_str, tag="Bs_{}".format(cut_name))

			ws, fit_result = fit_mc(chain, incut=cut_str)
			ws.Print()
			fit_result.Print()
			ws_file = ROOT.TFile("Bs/fitws_mc_Bs_{}.root".format(cut_name), "RECREATE")
			ws.Write()
			fit_result.Write()
			ws_file.Close()

	if args.plots:
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bs/fitws_mc_Bs_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			plot_fit(ws, tag="Bs_{}".format(cut_name), text=fit_text[cut_name])

	if args.tables:
		print("\n\n*** Printing tables ***\n")
		yields = {}
		for cut_name in cuts:
			ws_file = ROOT.TFile("Bs/fitws_mc_Bs_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			yields[cut_name] = extract_yields(ws)
		pprint(yields)

		print("\n\n*** Printing fit results ***\n")
		final_params = {}
		final_errors = {}
		parnames = []
		for cut_name in cuts:
			final_params[cut_name] = {}
			final_errors[cut_name] = {}
			print("{}".format(cut_name))
			ws_file = ROOT.TFile("Bs/fitws_mc_Bs_{}.root".format(cut_name), "READ")
			#ws = ws_file.Get("ws")
			#ws.Print()
			fit_result = ws_file.Get("fitresult_model_fitMC")
			this_final_params = fit_result.floatParsFinal()
			covm = fit_result.covarianceMatrix()
			iaa = -1
			ibb = -1
			for i in range(this_final_params.getSize()):
				parname = this_final_params[i].GetName()
				if not parname in parnames:
					parnames.append(parname)
				final_params[cut_name][parname] = this_final_params[i].getVal()
				final_errors[cut_name][parname] = this_final_params[i].getError()

				# Keep track of aa and bb indices
				if parname == "aa":
					iaa = i
				elif parname == "bb":
					ibb = i
			aa = final_params[cut_name]["aa"]
			daa = final_errors[cut_name]["aa"]
			bb = final_params[cut_name]["bb"]
			dbb = final_errors[cut_name]["bb"]
			c1 = aa * max(aa, bb) / (aa + bb)
			c2 = bb * max(aa, bb) / (aa + bb)
			if aa > bb:
				# c1 = aa**2 / (aa + bb)
				# c2 = aa * bb / (aa + bb)
				dc1_daa = (((aa + bb) * 2 * aa) - (aa**2)) / (aa + bb)**2
				dc1_dbb = -1. * aa**2 / (aa + bb)**2
				dc2_daa = (((aa + bb) * bb) - (aa * bb)) / (aa + bb)**2
				dc2_dbb = (((aa + bb) * aa) - (aa * bb)) / (aa + bb)**2
			else:
				# c1 = aa * bb / (aa + bb)
				# c2 = bb**2 / (aa + bb)
				dc1_daa = (((aa + bb) * bb) - (aa * bb)) / (aa + bb)**2
				dc1_dbb = (((aa + bb) * aa) - (aa * bb)) / (aa + bb)**2
				dc2_daa = -1. * bb**2 / (aa + bb)**2
				dc2_dbb = (((aa + bb) * 2 * bb) - (bb**2)) / (aa + bb)**2
			cov_aa_bb = covm[iaa][ibb]
			dc1 = math.sqrt(max(0., (dc1_daa * daa)**2 + (dc1_dbb * dbb)**2 + 2 * dc1_daa * dc1_dbb * cov_aa_bb))
			dc2 = math.sqrt(max(0., (dc2_daa * daa)**2 + (dc2_dbb * dbb)**2 + 2 * dc2_daa * dc2_dbb * cov_aa_bb))

			final_params[cut_name]["c1"] = c1
			final_params[cut_name]["c2"] = c2
			final_errors[cut_name]["c1"] = dc1
			final_errors[cut_name]["c2"] = dc2

		'''
		for parname in parnames:
			print(parname)
			pargraph = ROOT.TGraphErrors(len(cuts))
			for i, cut_name in enumerate(cuts):
				pargraph.SetPoint(i, i, final_params[side][cut_name][parname])
				pargraph.SetPointError(i, i, final_params[side][cut_name][parname])
			constant = ROOT.TF1("constant", "[0]", -0.5, i+0.5)
			pargraph.Fit(constant, "R0")
			pargraph.Print("all")
		'''
		pprint(final_params)
		pprint(final_errors)
		if args.all:
			with open("Bs/fitparams_MC_Bs.pkl", "wb") as f:
				pickle.dump(final_params, f)
			with open("Bs/fiterrs_MC_Bs.pkl", "wb") as f:
				pickle.dump(final_errors, f)
