'''
Fit MC mass distributions
'''
import pickle
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

#ROOT.gSystem.Load("libRooFit")
#ROOT.gSystem.Load("libRooFitCore")
#ROOT.gSystem.Load("libRooFitMore")
#ROOT.gSystem.Load("libHistFactory")
#ROOT.gSystem.Load("libRooStats")
#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
#from ROOT import TripleGaussian
#ROOT.gSystem.Load("include/TripleGaussianPdf.so")
#from ROOT import TripleGaussianPdf
#ROOT.gROOT.ProcessLine(".L include/TripleGaussianPdf.cc+");

figure_dir = "/home/dyu7/BFrag/data/fits/mc"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeSymHypatia, MakeJohnson, MakeDoubleGaussian, MakeTripleGaussian, \
	get_Bcands_name_mc, get_MC_fit_params, get_MC_fit_errs

rcache = [] # Prevent RooFit objects from disappearing

def pyerfc(x, par):
	erfc_arg = (x - par[0]) / par[1]
	return ROOT.TMath.Erfc(erfc_arg)

import re
re_ptcut = re.compile("\(pt \> (?P<ptmin>\d+\.\d+)\) && \(pt < (?P<ptmax>\d+\.\d+)\)")

def fit_mc(tree, mass_range=BS_FIT_WINDOW, incut="1", cut_name="", side="", fitfunc=""):
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

	# Signal
	if fitfunc == "hypatia":
		signal_pdf = MakeSymHypatia(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

		# Tweaks
		if cut_name == "ptbin_14p0_15p0":
			ws.var('hyp_a').setVal(1.2133e+00)
			ws.var('hyp_lambda').setVal(-9.7927e-01)
			ws.var('hyp_mu').setVal(5.3675e+00)
			ws.var('hyp_n').setVal(9.1553e+00)
			ws.var('hyp_sigma').setVal(1.3194e-02)
		elif cut_name == "ptbin_20p0_23p0" or cut_name == "ptbin_18p0_20p0" or cut_name == "ptbin_16p0_18p0":
			ws.var('hyp_a').setVal(1.3343536631565844)
			ws.var('hyp_lambda').setVal(-1.0188861578823865)
			ws.var('hyp_mu').setVal(5.367383710855175)
			ws.var('hyp_n').setVal(6.920729092472266)
			ws.var('hyp_sigma').setVal(0.01451002855831304)
		elif cut_name == "ptbin_23p0_26p0" or cut_name == "ptbin_23p0_28p0":
			# Initialize to ptbin_20p0_23p0, which looks better
			ws.var('hyp_a').setVal(1.188809962550541)
			ws.var('hyp_lambda').setVal(-1.1197579691299318)
			ws.var('hyp_mu').setVal(5.3674362914924325)
			ws.var('hyp_n').setVal(7.293739479229372)
			ws.var('hyp_sigma').setVal(0.015496519054279859)
		elif cut_name == "ptbin_29p0_34p0" or cut_name == "ptbin_34p0_45p0":
			ws.var('hyp_a').setVal(1.2536175536272454)
			ws.var('hyp_lambda').setVal(-0.852146074615403)
			ws.var('hyp_mu').setVal(5.3670243896118475)
			ws.var('hyp_n').setVal(7.520065932798015)
			ws.var('hyp_sigma').setVal(0.013348095459792361)
		elif cut_name == "ybin_0p0_0p25" or cut_name == "ybin_0p25_0p5":
			ws.var('hyp_a').setVal(10)
			ws.var('hyp_lambda').setVal(-2.3077468252815088)
			ws.var('hyp_mu').setVal(5.367377390841137)
			ws.var('hyp_n').setVal(5.343132835540865)
			ws.var('hyp_sigma').setVal(0.018490119893094497)
		elif cut_name == "ybin_1p0_1p25":
			ws.var('hyp_a').setVal(10)
			ws.var('hyp_lambda').setVal(-1.906980676697593)
			ws.var('hyp_mu').setVal(5.367222659859475)
			ws.var('hyp_n').setVal(18.140293764367716)
			ws.var('hyp_sigma').setVal(0.03174428507256944)

	elif fitfunc == "johnson":
		signal_pdf = MakeJohnson(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

	elif fitfunc == "2gauss":
		signal_pdf = MakeDoubleGaussian(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

	elif fitfunc == "3gauss":
		signal_pdf = MakeTripleGaussian(ws, mass_range, rcache=rcache)
		getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())

	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)
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

def plot_fit(ws, fit_result, tag="", text=None):
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

	#binning = ROOT.RooBinning(100, BS_FIT_WINDOW[0], BS_FIT_WINDOW[1])
	binning = xvar.getBinning()
	data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
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
	parser.add_argument("--selection", type=str, default="nominal", help="Selection (nominal, HiTrkPt)")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="johnson, hypatia")
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

	if args.selection == "nominal":
		sides = ["tagmatch", "probematch", "recomatch"]
	elif args.selection == "HiTrkPt":
		sides = ["tagHiTrkPtmatch", "probeHiTrkPtmatch"]
	elif args.selection in ["HiMuonPt", "MediumMuonPt", "MediumMuonID"]:
		sides = [f"tag{args.selection}match"]
	#elif "MuonPt" in args.selection:
	#	sides = [f"tag{args.selection}match", f"probe{args.selection}match"]
	#elif args.selection == "MediumMuon":
	#	sides = [f"tag{args.selection}match", f"probe{args.selection}match"]
	else:
		raise ValueError("Can't handle selection {}".format(args.selection))

	if args.fits:
		for side in sides:
			chain = ROOT.TChain(get_Bcands_name_mc(btype="Bs", side=side, trigger="HLT_Mu7_IP4", selection=args.selection))
			#chain.Add("data_Run2018C_part3.root")
			#chain.Add("data_Run2018D_part1.root")
			#chain.Add("data_Run2018D_part2.root")
			for mc_file in mc_files:
				chain.Add(mc_file)

			for cut_name in cuts:
				cut_str = cut_strings[cut_name]
				#plot_mc(chain, cut=cut_str, tag="Bs_{}".format(cut_name))

				ws, fit_result = fit_mc(chain, incut=cut_str, cut_name=cut_name, side=side, fitfunc=args.fitfunc)
				ws.Print()
				fit_result.Print()
				ws_file = ROOT.TFile(f"Bs/fitws_mc_Bs_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "RECREATE")
				ws.Write()
				fit_result.Write()
				ws_file.Close()

				# Clear cache
				del ws
				rcache = []

	if args.plots:
		for side in sides:
			for cut_name in cuts:
				ws_file = ROOT.TFile(f"Bs/fitws_mc_Bs_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
				#ws_file.ls()
				ws = ws_file.Get("ws")
				fit_result = ws_file.Get("fitresult_model_fitMC")
				plot_fit(ws, fit_result, tag=f"Bs_{side}_{cut_name}_{args.selection}_{args.fitfunc}", text=fit_text[cut_name])

	if args.tables:
		print("\n\n*** Printing tables ***\n")
		yields = {}
		for side in sides:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile(f"Bs/fitws_mc_Bs_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
				#ws_file.ls()
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
			pprint(yields)

		print("\n\n*** Printing fit results ***\n")
		final_params = {}
		final_errors = {}
		parnames = []
		bad_fits = []		
		for side in sides:
			final_params[side] = {}
			final_errors[side] = {}
			for cut_name in sorted(cuts):
				final_params[side][cut_name] = {}
				final_errors[side][cut_name] = {}
				print("{}".format(cut_name))
				ws_file = ROOT.TFile(f"Bs/fitws_mc_Bs_{side}_{cut_name}_{args.selection}_{args.fitfunc}.root", "READ")
				#ws = ws_file.Get("ws")
				#ws.Print()
				fit_result = ws_file.Get("fitresult_model_fitMC")
				this_final_params = fit_result.floatParsFinal()
				covm = fit_result.covarianceMatrix()
				for i in range(this_final_params.getSize()):
					parname = this_final_params[i].GetName()
					if not parname in parnames:
						parnames.append(parname)
					final_params[side][cut_name][parname] = this_final_params[i].getVal()
					final_errors[side][cut_name][parname] = this_final_params[i].getError()

					if "lambda" in parname:
						this_lambda = final_params[side][cut_name][parname]
					elif "delta" in parname:
						this_delta = final_params[side][cut_name][parname]
					elif "gamma" in parname:
						this_gamma = final_params[side][cut_name][parname]
					elif "mu" in parname:
						this_mu = final_params[side][cut_name][parname]

				if args.fitfunc == "johnson":
					this_mean = this_mu - this_lambda * math.exp(this_delta**-2 / 2) * math.sinh(this_gamma / this_delta)
					this_variance = math.sqrt(this_lambda**2 / 2 * (math.exp(this_delta**-2)-1) * (math.exp(this_delta**-2) * math.cosh(2*this_gamma/this_delta) + 1))
					this_xhwhm = this_mu + this_lambda * math.sinh((math.sqrt(2 * math.log(2)) - this_gamma) / this_delta)
					this_hwhm = this_xhwhm - this_mean
					#pprint(final_params[side][cut_name])
					print(f"\tmean = {this_mean}")
					print(f"\tvariance = {this_variance}")
					print(f"\tHWHM = {this_hwhm}")

				for parname in sorted(final_params[side][cut_name].keys()):
					print(f"\t{parname} = {final_params[side][cut_name][parname]} +/- {final_errors[side][cut_name][parname]}")
					if final_errors[side][cut_name][parname] < 1.e-6 and not "mu" in parname:
						bad_fits.append((side, cut_name, parname))

		if len(bad_fits) >= 1:
			print("WARNING : Bad fits were found:")
			for bad_fit in bad_fits:
				print(f"{bad_fit[0]} / {bad_fit[1]} / {bad_fit[2]} = {final_params[bad_fit[0]][bad_fit[1]][bad_fit[2]]} +/- {final_errors[bad_fit[0]][bad_fit[1]][bad_fit[2]]}")


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
		#pprint(final_params)
		#pprint(final_errors)
		if args.all:
			#with open(f"Bs/fitparams_MC_Bs_{args.selection}_{args.fitfunc}.pkl", "wb") as f:
			with open(get_MC_fit_params("Bs", selection=args.selection, fitfunc=args.fitfunc, frozen=False), "wb") as f:
				pickle.dump(final_params, f)
			#with open(f"Bs/fiterrs_MC_Bs_{args.selection}_{args.fitfunc}.pkl", "wb") as f:
			with open(get_MC_fit_errs("Bs", selection=args.selection, fitfunc=args.fitfunc, frozen=False), "wb") as f:
				pickle.dump(final_errors, f)
