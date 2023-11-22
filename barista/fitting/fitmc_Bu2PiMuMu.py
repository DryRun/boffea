'''
Fit MC mass distributions
'''
from pprint import pprint
import ROOT
from brazil.aguapreta import *
import pickle
ROOT.gROOT.SetBatch(True)

figure_dir = "/home/dyu7/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS

MMIN=5.15
MMAX=5.6

def fit_mc(chain, mass_range=[MMIN, MMAX], incut="1"):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"

	# Turn chain into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	#mass = ws.var("mass")
	chain.Print()
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(chain), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Signal: triple Gaussian
	'''
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.034, 0.001, 0.4])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.15, 0.001, 0.4])")
	g3 = ws.factory(f"Gaussian::g3(mass, mean, sigma3[0.25, 0.001, 0.4])")
	intermediate = ws.factory(f"SUM::intermediate(f1[0.95, 0.01, 0.99]*g2, g1)")
	signal_3g = ws.factory(f"SUM::signal_3g(f2[0.95, 0.01, 0.99]*g3, intermediate)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_3g, nsignal)
	'''

	# Crystal ball + gaussian
	bkgd_jpsipi_cb = ws.factory(f"RooCBShape::bkgd_jpsipi_cb(mass, mean_jpsipi[{mass_range[0]}, {mass_range[1]}], sigma_jpsipi_cb[0.1, 0.005, 0.4], alpha_jpsipi_cb[-1., -20, -0.01], n_jpsipi_cb[1, 0.01, 100])");
	bkgd_jpsipi_gauss = ws.factory(f"Gaussian::bkgd_jpsipi_gauss(mass, mean_jpsipi, sigma_jpsipi_gauss[0.15, 0.05, 0.3])")
	c_cb_jpsipi = ws.factory("c_cb_jpsipi[0.9, 0.8, 1.0]")
	bkgd_jpsipi_pdf = ROOT.RooAddPdf("bkgd_jpsipi_pdf", "bkgd_jpsipi_pdf", ROOT.RooArgList(bkgd_jpsipi_cb, bkgd_jpsipi_gauss), ROOT.RooArgList(c_cb_jpsipi))
	getattr(ws, "import")(bkgd_jpsipi_pdf, ROOT.RooFit.RecycleConflictNodes())
	nbkgd_jpsipi = ws.factory(f"nbkgd_jpsipi[{ndata*0.5}, 0.0, {ndata*2.0}]")
	bkgd_jpsipi_model = ROOT.RooExtendPdf("bkgd_jpsipi_model", "bkgd_jpsipi_model", bkgd_jpsipi_pdf, nbkgd_jpsipi)
	getattr(ws, "import")(bkgd_jpsipi_model, ROOT.RooFit.RecycleConflictNodes())	

	#model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(bkgd_jpsipi_model, bkgd_exp_model))

	# Perform fit
	##nll = model.createNLL(rdataset, ROOT.RooFit.NumCPU(8))
	#minimizer = ROOT.RooMinuit(nll)
	#minimizer.migrad()
	#minimizer.minos()
	fit_result = bkgd_jpsipi_model.fitTo(rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(bkgd_jpsipi_model,  ROOT.RooFit.RecycleConflictNodes())
	ws.Print()
	return ws, fit_result

def plot_fit(ws, tag="", text=None):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("bkgd_jpsipi_model")
	rdataset = ws.data("fitData")
	xvar = ws.var("mass")
	print("DEBUG : xvar:")
	xvar.Print()
	binning = ROOT.RooBinning(100, MMIN, MMAX)
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))

	canvas = ROOT.TCanvas("c_mcfit_Bu2PiMuMu_{}".format(tag), "c_mcfit_Bu2PiMuMu_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()
	top.SetLogy()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
	ws.Print()
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
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

	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -4
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		pull_val = (data_val - fit_val) / max(math.sqrt(fit_val), 1.e-10)
		pull_hist.SetBinContent(xbin, pull_val)
		pull_hist.SetBinError(xbin, 1)
		chi2 += pull_val**2
		ndf += 1
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi #pi^{#pm}} [GeV]")
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

	zero = ROOT.TLine(BU_FIT_WINDOW[0], 0., BU_FIT_WINDOW[1], 0.)
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


if __name__ == "__main__":
	cuts = list(set(fit_cuts["tag"] + fit_cuts["probe"]))

	do_fit = True
	if do_fit:
		#tfile = ROOT.TFile("MCEfficiency_Bu2PiJpsi.root", "READ")
		#tree = tfile.Get(f"Bcands_recomatch_Bu2PiJpsi2KMuMu_inclusive")
		chain = ROOT.TChain("Bcands_recomatch_Bu2PiJpsi2KMuMu_inclusive")
		chain.Add("MCEfficiency_Bu2PiJpsi.root")
		for cut_name in cuts:
			cut_str = cut_strings[cut_name]
			ws, fit_result = fit_mc(chain, incut=cut_str)
			ws.Print()
			fit_result.Print()
			ws_file = ROOT.TFile("Bu/fitws_mc_Bu2PiJpsi_{}.root".format(cut_name), "RECREATE")
			ws.Write()
			fit_result.Write()
			ws_file.Close()

	do_plot = True
	if do_plot:
		for cut_name in cuts:
			if not "inclusive" in cut_name: 
				continue
			ws_file = ROOT.TFile("Bu/fitws_mc_Bu2PiJpsi_{}.root".format(cut_name), "READ")
			#ws_file.ls()
			ws = ws_file.Get("ws")
			plot_fit(ws, tag="Bu2PiJpsi_{}".format(cut_name), text=fit_text[cut_name])

