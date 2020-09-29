'''
Fit MC mass distributions
'''
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

ROOT.gROOT.ProcessLine(open('models.cc').read())
from ROOT import MyErfc

figure_dir = "/home/dryu/BFrag/data/fits/mc"

MMIN=5.15
MMAX=5.7



def fit_Bu2PiMuMu(tree, mass_range=[MMIN, MMAX], cut="1"):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{cut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	#mass = ws.var("mass")
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
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

	# Crystal ball
	signal_cb = ws.factory(f"RooCBShape::CBall(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma[0.15, 0.001, 0.4], alpha[-1., -100., 0.], n[1, 0,100000])");
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_cb, nsignal)

	#model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model))

	# Perform fit
	##nll = model.createNLL(rdataset, ROOT.RooFit.NumCPU(8))
	#minimizer = ROOT.RooMinuit(nll)
	#minimizer.migrad()
	#minimizer.minos()
	fit_result = signal_model.fitTo(rdataset, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save())

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdataset)
	getattr(ws, "import")(signal_model)
	return ws, fit_result

def plot_fit(ws, tag=""):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	signal_model = ws.pdf("signal_model")
	rdataset = ws.data("fitData")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_mcfit_Bu2PiMuMu_{}".format(side), "c_mcfit_Bu2PiMuMu_{}".format(side), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"))
	signal_model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(0.06)
	rplot.GetYaxis().SetTitleSize(0.06)
	rplot.GetYaxis().SetTitleOffset(0.8)
	rplot.Draw()

	l = ROOT.TLegend(0.15, 0.45, 0.35, 0.7)
	l.SetFillColor(0)
	l.SetBorderSize(0)
	l.AddEntry("data", "Data", "lp")
	l.AddEntry("fit", "Total fit", "l")
	l.Draw()

	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.5)
	bottom.SetTopMargin(0.02)
	bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	binning = ROOT.RooBinning(100, MMIN, MMAX)
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = signal_model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
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

	zero = ROOT.TLine(MMIN, 0., MMAX, 0.)
	zero.SetLineColor(ROOT.kGray)
	zero.SetLineStyle(3)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	chi2text = ROOT.TLatex(0.15, 0.8, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	chi2text.SetNDC()
	chi2text.Draw()

	canvas.cd()
	canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
	canvas.SaveAs("{}/{}.pdf".format(figure_dir, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)


if __name__ == "__main__":
	import glob
	mc_files = glob.glob("")

	do_fit = True
	if do_fit:
		tfile = ROOT.TFile("mc_Bu2PiJPsi2PiMuMu.root", "READ")
		for side in ["tag", "probe"]:
			tree = tfile.Get(f"Bcands_{side}match")
			ws, fit_result = fit_Bu2PiMuMu(tree)
			ws.Print()
			fit_result.Print()
			ws_file = ROOT.TFile("fitws_mc_Bu2PiJPsi2PiMuMu_{}.root".format(side), "RECREATE")
			ws.Write()
			fit_result.Write()
			ws_file.Close()

	for side in ["tag", "probe"]:
		ws_file = ROOT.TFile("fitws_mc_Bu2PiJPsi2PiMuMu_{}.root".format(side), "READ")
		ws_file.ls()
		ws = ws_file.Get("ws")
		plot_fit(ws, tag="_fit_mc_Bu2PiJPsi2PiMuMu_{}".format(side))