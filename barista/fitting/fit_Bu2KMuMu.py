'''
Fit MC mass distributions
'''
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

ROOT.gROOT.ProcessLine(open('models.cc').read())
from ROOT import MyErfc

figure_dir = "/home/dyu7/BFrag/data/fits/data"

# Fit bins
ptbins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
ybins = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.0, 2.25]

fit_cuts = {}
fit_text = {} # Text for plots
fit_cuts["inclusive"] = "1"
fit_text["inclusive"] = "Inclusive"
for ipt in range(len(ptbins) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins[ipt], ptbins[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins[ipt], ptbins[ipt+1]).replace(".", "p")
	fit_cuts[cut_name] = cut_str
	fit_text[cut_name] = "{}<pT<{}".format(ptbins[ipt], ptbins[ipt+1])

for iy in range(len(ybins) - 1):
	cut_str = "(pt > 5.0) && (pt < 30.0) && ({} < abs(y)) && (abs(y) < {})".format(ybins[iy], ybins[iy+1])
	cut_name = "ybin_{}_{}".format(ybins[iy], ybins[iy+1]).replace(".", "p")
	fit_cuts[cut_name] = cut_str
	fit_text[cut_name] = "{}<|y|<{}".format(ybins[iy], ybins[iy+1])

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

def plot_data(tree, mass_range=[5.05, 5.5], cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dyu7/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=[5.05, 5.5], cut="1"):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{cut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	#tree.Print()
	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")

	#mass = ws.var("mass")
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Signal: double Gaussian
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.034, 0.001, 0.2])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.15, 0.001, 0.2])")
	signal_dg = ws.factory(f"SUM::signal_dg(f1[0.95, 0.01, 0.99]*g1, g2)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_dg, nsignal)

	# Background model: exponential + (Bu > Jpsi pi gaussian) + (partial reco ERF)
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.66, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	#bkgd_JpsiPi = ws.factory(f"Gaussian::bkgd_JpsiPi(mass, 5.37, 0.038)")
	#nJpsiPi = ROOT.RooFormulaVar("nJpsiPi", "nJpsiPi", "0.04 * nsignal", ROOT.RooArgList(nsignal))
	#print(nJpsiPi)
	#print(bkgd_JpsiPi)
	#bkgd_JpsiPi_model = ROOT.RooExtendPdf("bkgd_JpsiPi_model", "bkgd_JpsiPi_model", bkgd_JpsiPi, nJpsiPi)
	bkgd_JpsiPi = ws.factory(f"RooCBShape::CBall(mass, 5.34716e+00, 3.93177e-02, -6.59331e-01, 1.85890e+00)");
	nJpsiPi = ROOT.RooFormulaVar("nJpsiPi", "nJpsiPi", "0.04 * nsignal", ROOT.RooArgList(nsignal))
	bkgd_JpsiPi_model = ROOT.RooExtendPdf("bkgd_JpsiPi_model", "bkgd_JpsiPi_model", bkgd_JpsiPi, nJpsiPi)


	erfc_x0 = ws.factory(f"erfc_x0[5.12, 5.07, 5.2]")
	erfc_width = ws.factory(f"erfc_width[0.005, 0.0001, 0.1]")
	bkgd_erfc = ws.factory(f"GenericPdf::bkgd_erfc('MyErfc(mass, erfc_x0, erfc_width)', {{mass, erfc_x0, erfc_width}})")
	#erfc_arg = ROOT.RooFormulaVar("erfc_arg", "erfc_arg", "(mass - erfc_x) / (erfc_width)", ROOT.RooArgList(mass, erfc_x, erfc_width))
	#erfc_tf1 = ROOT.TF1("erfc", pyerfc, 5.05, 5.2, 2)
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erfc_tf1, erfc_arg)
	#ROOT.gInterpreter.ProcessLine('RooAbsPdf* myerfc = RooFit::bindPdf("erfc_pdf", TMath::Erfc, erfc_arg);')
	#x = ROOT.x
	#erfc_pdf = ROOT.myerfc
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erf, erfc_arg)
	nbkgd_erfc = ws.factory(f"nbkgd_erfc[{ndata*0.1}, 0.0, {ndata*2.0}]")
	bkgd_erfc_model = ROOT.RooExtendPdf("bkgd_erfc_model", "bkgd_erfc_model", bkgd_erfc, nbkgd_erfc)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model, bkgd_JpsiPi_model, bkgd_erfc_model))

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
	getattr(ws, "import")(model)
	return ws, fit_result

def plot_fit(ws, tag="", text=None):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	rdataset = ws.data("fitData")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_datafit_{}".format(tag), "c_datafit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_exp"), ROOT.RooFit.Components("bkgd_exp_model"), ROOT.RooFit.LineColor(ROOT.kRed+1))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_JpsiPi"), ROOT.RooFit.Components("bkgd_JpsiPi_model"), ROOT.RooFit.LineColor(ROOT.kOrange-3))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_erfc"), ROOT.RooFit.Components("bkgd_erfc_model"), ROOT.RooFit.LineColor(ROOT.kMagenta+1))
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
	l.AddEntry("signal", "B^{+}#rightarrowJ/#psi K^{#pm}", "lf")
	l.AddEntry("bkgd_exp", "Comb. bkgd.", "l")
	l.AddEntry("bkgd_JpsiPi", "B^{+}#rightarrowJ/#psi#pi^{#pm}", "l")
	l.AddEntry("bkgd_erfc", "B^{+}#rightarrowJ/#psi+hadrons", "l")
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

	binning = ROOT.RooBinning(100, 5.05, 5.5)
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -11
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		pull_val = (data_val - fit_val) / math.sqrt(fit_val)
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

	zero = ROOT.TLine(5.05, 0., 5.5, 0.)
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
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("data_Run2018*.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = sorted(fit_cuts.keys())

	if args.fits:
		for side in ["tag", "probe"]:
			chain = ROOT.TChain("Bcands_Bu_{}_HLT_Mu9_IP5|HLT_Mu9_IP6".format(side))
			#chain.Add("data_Run2018C_part3.root")
			#chain.Add("data_Run2018D_part1.root")
			#chain.Add("data_Run2018D_part2.root")
			for data_file in data_files:
				chain.Add(data_file)

			for cut_name in cuts:
				cut_str = fit_cuts[cut_name]
				plot_data(chain, cut=cut_str, tag="Bu_{}_{}".format(side, cut_name))

				ws, fit_result = fit_data(chain, cut=cut_str)
				ws.Print()
				fit_result.Print()
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "RECREATE")
				ws.Write()
				fit_result.Write()
				ws_file.Close()

	if args.plots:
		for side in ["tag", "probe"]:
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				plot_fit(ws, tag="Bu_{}_{}".format(side, cut_name), text=fit_text[cut_name])

	if args.tables:
		yields = {}
		for side in ["tag", "probe"]:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
		pprint(yields)