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

figure_dir = "/home/dyu7/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW_MC, BD_FIT_WINDOW_MC, BS_FIT_WINDOW_MC, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeSymHypatia, MakeJohnson, MakeDoubleGaussian, MakeTripleGaussian, \
	get_Bcands_name_mc, get_MC_fit_params, get_MC_fit_errs
rcache = []

import re
re_ptcut = re.compile("\(pt \> (?P<ptmin>\d+\.\d+)\) && \(pt < (?P<ptmax>\d+\.\d+)\)")
def fit_mc(tree, mass_range=BU_FIT_WINDOW_MC, incut="1", cut_name="inclusive", binned=False, side="", fitfunc=""):
	ws = ROOT.RooWorkspace('ws')

	# Slightly widen pT window, to min of 5 GeV
	if "ptbin" in cut_name:
		re_match = re_ptcut.search(incut)
		ptmin = float(re_match.group("ptmin"))
		ptmax = float(re_match.group("ptmax"))
		if ptmax - ptmin < 7.5:
			extra_pt = 7.5 - (ptmax - ptmin)
			incut = incut.replace("pt > {}".format(ptmin), "pt > {}".format(ptmin - extra_pt / 2.))
			incut = incut.replace("pt < {}".format(ptmax), "pt < {}".format(ptmax + extra_pt / 2.))
	if cut_name == "ybin_1p5_1p75" or cut_name == "ybin_1p75_2p0":
		mass_range = [5.15, 5.4]
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

	# Signal
	if fitfunc == "hypatia":
		signal_pdf = MakeSymHypatia(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

		# Tweaks
		if cut_name == "ptbin_8p0_13p0" and "tag" in side:
			ws.var('hyp_a').setVal(3.386815155802858)
			ws.var('hyp_lambda').setVal(-1.2691101086204366)
			ws.var('hyp_mu').setVal(5.2802650937726)
			ws.var('hyp_n').setVal(1.1901921443686263)
			ws.var('hyp_sigma').setVal(0.02837524828019758)
		elif cut_name == "ptbin_12p0_13p0":
			ws.var('hyp_a').setVal(3.386815155802858)
			ws.var('hyp_lambda').setVal(-1.2691101086204366)
			ws.var('hyp_mu').setVal(5.2802650937726)
			ws.var('hyp_n').setVal(1.1901921443686263)
			ws.var('hyp_sigma').setVal(0.02837524828019758)
		elif cut_name == "ptbin_13p0_14p0" and "tag" in side:
			ws.var('hyp_a').setVal(3.7307352899265624)
			ws.var('hyp_lambda').setVal(-1.5435436168481083)
			ws.var('hyp_mu').setVal(5.279799841986943)
			ws.var('hyp_n').setVal(0.5000000655629718)
			ws.var('hyp_sigma').setVal(0.02708680964920084)
		elif cut_name == "ptbin_14p0_15p0":
			ws.var('hyp_a').setVal(3.5673340893868404)
			ws.var('hyp_lambda').setVal(-1.2573374473317394)
			ws.var('hyp_mu').setVal(5.280025979857378)
			ws.var('hyp_n').setVal(1.1193611497398772)
			ws.var('hyp_sigma').setVal(0.026958320399402953)
		elif cut_name == "ptbin_16p0_18p0" and ("tag" in side or "reco" in side):
			ws.var('hyp_a').setVal(3.5673340893868404)
			ws.var('hyp_lambda').setVal(-1.2573374473317394)
			ws.var('hyp_mu').setVal(5.280025979857378)
			ws.var('hyp_n').setVal(1.1193611497398772)
			ws.var('hyp_sigma').setVal(0.026958320399402953)
		elif cut_name == "ptbin_18p0_23p0" and "tag" in side:
			ws.var('hyp_a').setVal(3.9963e+00)
			ws.var('hyp_lambda').setVal(-1.3095e+00)
			ws.var('hyp_mu').setVal(5.2793e+00)
			ws.var('hyp_n').setVal(7.5060e-01)
			ws.var('hyp_sigma').setVal(2.7072e-02)
		elif cut_name == "ptbin_28p0_33p0" and "tag" in side:
			ws.var('hyp_a').setVal(3.9963e+00)
			ws.var('hyp_lambda').setVal(-1.3095e+00)
			ws.var('hyp_mu').setVal(5.2793e+00)
			ws.var('hyp_n').setVal(7.5060e-01)
			ws.var('hyp_sigma').setVal(2.7072e-02)
		elif cut_name == "ptbin_29p0_34p0":
			ws.var('hyp_a').setVal(3.9963e+00)
			ws.var('hyp_lambda').setVal(-1.3095e+00)
			ws.var('hyp_mu').setVal(5.2793e+00)
			ws.var('hyp_n').setVal(7.5060e-01)
			ws.var('hyp_sigma').setVal(2.7072e-02)

		elif cut_name == "ybin_0p5_0p75" and "tag" in side:
			ws.var('hyp_a').setVal(3.56325251653428)
			ws.var('hyp_lambda').setVal(-1.5860776603929185)
			ws.var('hyp_mu').setVal(5.279740259834927)
			ws.var('hyp_n').setVal(0.5000020086296986)
			ws.var('hyp_sigma').setVal(0.03665061887058412)
		elif cut_name == "ybin_0p0_0p25" and "probe" in side:
			ws.var('hyp_a').setVal(3.9963e+00)
			ws.var('hyp_lambda').setVal(-1.3095e+00)
			ws.var('hyp_mu').setVal(5.2793e+00)
			ws.var('hyp_n').setVal(7.5060e-01)
			ws.var('hyp_sigma').setVal(2.7072e-02)
		elif cut_name == "ybin_0p75_1p0":
			ws.var('hyp_a').setVal(3.56325251653428)
			ws.var('hyp_lambda').setVal(-1.5860776603929185)
			ws.var('hyp_mu').setVal(5.279740259834927)
			ws.var('hyp_n').setVal(0.5000020086296986)
			ws.var('hyp_sigma').setVal(0.03665061887058412)
		elif cut_name == "ybin_1p0_1p25" and "reco" in side:
			ws.var('hyp_a').setMin(2.0)
			ws.var('hyp_a').setVal(3.650591372779841)
			ws.var('hyp_lambda').setVal(-2.053138446173822)
			ws.var('hyp_mu').setVal(5.277492617451097)
			ws.var('hyp_n').setVal(2)
			ws.var('hyp_sigma').setVal(0.050863683221835154)
		elif cut_name == "ybin_1p25_1p5" and "reco" in side:
			ws.var('hyp_a').setMin(2.0)
			ws.var('hyp_a').setVal(3.650591372779841)
			ws.var('hyp_lambda').setVal(-2.053138446173822)
			ws.var('hyp_mu').setVal(5.277492617451097)
			ws.var('hyp_n').setVal(2)
			ws.var('hyp_sigma').setVal(0.050863683221835154)
		elif cut_name == "ybin_1p5_1p75":
			ws.var('hyp_a').setMin(2.0)
			ws.var('hyp_a').setVal(3.650591372779841)
			ws.var('hyp_lambda').setVal(-2.053138446173822)
			ws.var('hyp_mu').setVal(5.277492617451097)
			ws.var('hyp_n').setVal(2)
			ws.var('hyp_sigma').setVal(0.050863683221835154)
		elif cut_name == "ybin_1p5_1p75":
			ws.var('hyp_a').setMin(2.0)
			ws.var('hyp_a').setVal(2.5)
			ws.var('hyp_lambda').setVal(-2.053138446173822)
			ws.var('hyp_mu').setVal(5.277492617451097)
			ws.var('hyp_n').setVal(3.0)
			ws.var('hyp_sigma').setVal(0.050863683221835154)
		elif cut_name == "ybin_1p75_2p0":
			ws.var('hyp_sigma').setVal(0.06)
		elif cut_name == "inclusive": 
			ws.var('hyp_a').setVal(3.3789932616572416)
			ws.var('hyp_lambda').setVal(-1.3043934756507838)
			ws.var('hyp_mu').setVal(5.279555940275684)
			ws.var('hyp_n').setVal(1.6223656570181437)
			ws.var('hyp_sigma').setVal(0.02760893395253671)

	elif fitfunc == "johnson":
		signal_pdf = MakeJohnson(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

	elif fitfunc == "2gauss":
		signal_pdf = MakeDoubleGaussian(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

	elif fitfunc == "3gauss":
		signal_pdf = MakeTripleGaussian(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
		if cut_name == "ybin_1p25_1p5" and "tag" in side:
			ws.var('tg_aa').setVal(0.5917308572208606)
			ws.var('tg_bb').setVal(0.5914699171629932)
			ws.var('tg_mean').setVal(5.280916518182134)
			ws.var('tg_sigma1').setVal(0.023238858150439003)
			ws.var('tg_sigma2').setVal(0.023419539842810762)
			ws.var('tg_sigma3').setVal(0.050496542583524914)

	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)
	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model))


	'''
	if cut_name == "ptbin_8p0_13p0":
		ws.var('hyp_a').setVal(7.275745654055582)
		#ws.var('hyp_a2').setVal(7.275745654055582)		
		ws.var('hyp_lambda').setVal(-1.0067107376670137)
		ws.var('hyp_mu').setVal(5.279661966888891)
		ws.var('hyp_n').setVal(8.265412496047785)
		#ws.var('hyp_n2').setVal(8.265412496047785)		
		ws.var('hyp_sigma').setVal(0.025528613146354573)
	elif cut_name == "ptbin_11p0_12p0" or cut_name == "ptbin_10p0_11p0" or cut_name == "ptbin_12p0_13p0" or cut_name == "inclusive":
		ws.var('hyp_a').setVal(6.641113571220526)
		#ws.var('hyp_a2').setVal(6.641113571220526)		
		ws.var('hyp_lambda').setVal(-1.1427511878656773)
		ws.var('hyp_mu').setVal(5.280033853167267)
		ws.var('hyp_n').setVal(0.500380069381496)
		#ws.var('hyp_n2').setVal(0.500380069381496)		
		ws.var('hyp_sigma').setVal(0.027263835445174116)
	elif cut_name == "ptbin_13p0_14p0": 
		ws.var('hyp_a').setVal(4.098838030245997)
		#ws.var('hyp_a2').setVal(84.02652476227416)
		ws.var('hyp_lambda').setVal(-1.2600957756675726)
		ws.var('hyp_mu').setVal(5.2801108553608795)
		ws.var('hyp_n').setVal(0.500018322140428)
		#ws.var('hyp_n2').setVal(0.735157026018763)
		ws.var('hyp_sigma').setVal(0.026695544175453105)
		ws.var('nsignal').setVal(9097.021260376974)
	elif cut_name == "ptbin_15p0_16p0" or cut_name == "ptbin_13p0_18p0":
		ws.var('hyp_a').setVal(5)
		#ws.var('hyp_a2').setVal(5)		
		ws.var('hyp_lambda').setVal(-1.2472323943707782)
		ws.var('hyp_mu').setVal(5.280327479922378)
		ws.var('hyp_n').setVal(7.375310140389502e-07)
		#ws.var('hyp_n2').setVal(7.375310140389502e-07)		
		ws.var('hyp_sigma').setVal(0.026784749158424062)
		ws.var('nsignal').setVal(3761.96071872678)
	elif cut_name == "ptbin_18p0_23p0" \
		or cut_name == "ptbin_18p0_20p0" \
		or cut_name == "ptbin_16p0_18p0":
		ws.var('hyp_a').setVal(5)
		#ws.var('hyp_a2').setVal(5)		
		ws.var('hyp_lambda').setVal(-1.1403319564083407)
		ws.var('hyp_mu').setVal(5.279257602044328)
		ws.var('hyp_n').setVal(5.606733613136833)
		#ws.var('hyp_n2').setVal(5.606733613136833)		
		ws.var('hyp_sigma').setVal(0.024140061131064254)
	elif cut_name == "ptbin_23p0_28p0" or cut_name == "ptbin_26p0_29p0":
		ws.var('hyp_a').setVal(5)
		#ws.var('hyp_a2').setVal(5)		
		ws.var('hyp_lambda').setVal(-1.1403319564083407)
		ws.var('hyp_mu').setVal(5.279257602044328)
		ws.var('hyp_n').setVal(5.606733613136833)
		#ws.var('hyp_n2').setVal(5.606733613136833)		
		ws.var('hyp_sigma').setVal(0.024140061131064254)
	elif cut_name == "ptbin_29p0_34p0" or cut_name == "ptbin_34p0_45p0":
		ws.var('hyp_a').setVal(1.5000038914473408)
		#ws.var('hyp_a2').setVal(1.5000038914473408)		
		ws.var('hyp_lambda').setVal(-1.5069293094758294)
		ws.var('hyp_mu').setVal(5.279348036994916)
		ws.var('hyp_n').setVal(3.3435488374681617)
		#ws.var('hyp_n2').setVal(3.3435488374681617)		
		ws.var('hyp_sigma').setVal(0.02934758343510314)

	elif cut_name == "ybin_0p0_0p25":
		ws.var("hyp_a").setVal(1.5000189727317448)
		ws.var("hyp_a").setMin(1.3)
		ws.var("hyp_lambda").setVal(-2.701193266084048)
		ws.var("hyp_mu").setVal(5.279697621492577)
		ws.var("hyp_n").setVal(1.7375285014517434)
		ws.var("hyp_sigma").setVal(0.02800542858363888)
		ws.var("nsignal").setVal(4874)
	elif cut_name == "ybin_0p25_0p5":
		ws.var('hyp_a').setVal(1.8562932288678495)
		#ws.var('hyp_a2').setVal(1.8562932288678495)		
		ws.var('hyp_lambda').setVal(-2.148967160060706)
		ws.var('hyp_mu').setVal(5.279306716727721)
		ws.var('hyp_n').setVal(1.77367683473856)
		#ws.var('hyp_n2').setVal(1.77367683473856)		
		ws.var('hyp_sigma').setVal(0.031176524193378317)
		ws.var('nsignal').setVal(4713.016784980348)
	elif cut_name == "ybin_0p75_1p0":
		ws.var('hyp_a').setVal(22.279724510193837)
		#ws.var('hyp_a2').setVal(10.946535752800338)
		ws.var('hyp_lambda').setVal(-1.625582749494109)
		ws.var('hyp_mu').setVal(5.279301558989023)
		ws.var('hyp_n').setVal(2.790682028946305)
		#ws.var('hyp_n2').setVal(2.6397502195924134)
		ws.var('hyp_sigma').setVal(0.026193245865654385)

	elif cut_name == "ybin_1p25_1p5":
		ws.var('hyp_a').setVal(5)
		#ws.var('hyp_a2').setVal(5)		
		ws.var('hyp_lambda').setVal(-1.468838761142754)
		ws.var('hyp_mu').setVal(5.279297279329609)
		ws.var('hyp_n').setVal(0.6721768715035447)
		#ws.var('hyp_n2').setVal(0.6721768715035447)		
		ws.var('hyp_sigma').setVal(0.04327205409255316)
	elif cut_name == "ybin_1p75_2p0":
		ws.var('hyp_a').setVal(3.0569513567559827)
		#ws.var('hyp_a2').setVal(3.0569513567559827)		
		ws.var('hyp_lambda').setVal(-1.4835065290891034)
		ws.var('hyp_mu').setVal(5.276668272043223)
		ws.var('hyp_n').setVal(1.5467417558961083)
		#ws.var('hyp_n2').setVal(1.5467417558961083)		
		ws.var('hyp_n').setMin(1.0)
		#ws.var('hyp_n2').setMin(1.0)
		ws.var('hyp_sigma').setVal(0.058803608523859206)
	'''

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

def plot_fit(ws, fit_result, tag="", text=None, binned=False):
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

	#binning = ROOT.RooBinning(100, xvar.getMin(), xvar.getMax()) # BU_FIT_WINDOW_MC[0], BU_FIT_WINDOW_MC[1]
	binning = xvar.getBinning()
	data_hist = ROOT.RooAbsData.createHistogram(rdata, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.
	ndf = -5
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

	top.SetLogy(False)
	canvas.SaveAs("{}/{}_lineary.png".format(figure_dir, canvas.GetName()))
	canvas.SaveAs("{}/{}_lineary.pdf".format(figure_dir, canvas.GetName()))

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
	parser.add_argument("--selection", type=str, default="nominal", help="Selection (nominal, HiTrkPt)")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="johnson, hypatia")
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

	if args.selection == "nominal":
		sides = ["tagmatch", "probematch", "recomatch"]
	elif args.selection == "HiTrkPt":
		sides = ["tagHiTrkPtmatch", "probeHiTrkPtmatch"]
	elif args.selection in ["HiMuonPt", "MediumMuonPt", "MediumMuonID"]:
		sides = [f"tag{args.selection}match"]
	#elif "MuonPt" in args.selection:
	#	sides = [f"tag{args.selection}match", f"probe{args.selection}match"]
	#elif args.selection == "MediumMuon":
#		sides = [f"tag{args.selection}match", f"probe{args.selection}match"]
	else:
		raise ValueError("Can't handle selection {}".format(args.selection))

	if args.fits:
		for side in sides:
			chain = ROOT.TChain(get_Bcands_name_mc(btype="Bu", side=side, trigger="HLT_Mu7_IP4", selection=args.selection))
			for mc_file in mc_files:
				chain.Add(mc_file)

			for cut_name in cuts:
				print("\n*** Fitting {} ***".format(cut_name))
				cut_str = cut_strings[cut_name]
				#plot_mc(chain, cut=cut_str, tag="Bu_{}".format(cut_name))

				ws, fit_result = fit_mc(chain, incut=cut_str, cut_name=cut_name, side=side, fitfunc=args.fitfunc)
				ws.Print()
				#print("DEBUG : fit result = ")
				fit_result.Print()
				#print("DEBUG : writing to " + "Bu/fitws_mc_Bu_{}.root".format(cut_name))
				ws_file = ROOT.TFile(f"Bu/fitws_mc_Bu_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "RECREATE")
				ws.Write()
				fit_result.Write()
				ws_file.Close()

				print("\nDone fitting {}\n".format(cut_name))

				# Clear cache
				del ws
				rcache = []

	if args.plots:
		for side in sides:
			for cut_name in cuts:
				ws_file = ROOT.TFile(f"Bu/fitws_mc_Bu_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
				#ws_file.ls()
				ws = ws_file.Get("ws")
				fit_result = ws_file.Get("fitresult_model_fitMC")
				plot_fit(ws, fit_result, tag=f"Bu_{side}_{cut_name}_{args.selection}_{args.fitfunc}", text=fit_text[cut_name])

	if args.tables:
		print("\n\n*** Printing tables ***\n")
		yields = {}
		for side in sides:
			for cut_name in cuts:
				ws_file = ROOT.TFile(f"Bu/fitws_mc_Bu_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
				#ws_file.ls()
				ws = ws_file.Get("ws")
				yields[cut_name] = extract_yields(ws)
			pprint(yields)

		print("\n\n*** Printing fit results ***\n")
		final_params = {}
		final_errors = {}
		parnames = []
		for side in sides:
			final_params[side] = {}
			final_errors[side] = {}
			for cut_name in cuts:
				final_params[side][cut_name] = {}
				final_errors[side][cut_name] = {}
				print("{}".format(cut_name))
				ws_file = ROOT.TFile(f"Bu/fitws_mc_Bu_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
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
					final_params[side][cut_name][parname] = this_final_params[i].getVal()
					final_errors[side][cut_name][parname] = this_final_params[i].getError()

		print("Printing parameters and errors:")
		bad_fits = []
		for side in sides:
			for cut_name in sorted(cuts):
				for parname in final_params[side][cut_name]:
					print(f"{side} / {cut_name} / {parname} = {final_params[side][cut_name][parname]} +/- {final_errors[side][cut_name][parname]}")
					if final_errors[side][cut_name][parname] < 1.e-6 and parname != "hyp_mu":
						bad_fits.append((side, cut_name, parname))

		if len(bad_fits) >= 1:
			print("WARNING : Bad fits were found:")
			for bad_fit in bad_fits:
				print(f"{bad_fit[0]} / {bad_fit[1]} / {bad_fit[2]} = {final_params[bad_fit[0]][bad_fit[1]][bad_fit[2]]} +/- {final_errors[bad_fit[0]][bad_fit[1]][bad_fit[2]]}")

		#pprint(final_params)
		#pprint(final_errors)
		if args.all:
			#with open(f"Bu/fitparams_MC_Bu_{args.selection}_{args.fitfunc}.pkl", "wb") as f:
			with open(get_MC_fit_params("Bu", selection=args.selection, fitfunc=args.fitfunc, frozen=False), "wb") as f:
				pickle.dump(final_params, f)
			#with open(f"Bu/fiterrs_MC_Bu_{args.selection}_{args.fitfunc}.pkl", "wb") as f:
			with open(get_MC_fit_errs("Bu", selection=args.selection, fitfunc=args.fitfunc, frozen=False), "wb") as f:
				pickle.dump(final_errors, f)
