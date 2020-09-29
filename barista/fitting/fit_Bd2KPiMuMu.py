'''
Fit MC mass distributions
'''
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

# Fit bins
ptbins = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
ybins = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.0, 2.25]

fit_cuts = {}
fit_cuts["inclusive"] = "1"
for ipt in range(len(ptbins) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins[ipt], ptbins[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins[ipt], ptbins[ipt+1]).replace(".", "p")
	fit_cuts[cut_name] = cut_str

for iy in range(len(ybins) - 1):
	cut_str = "(pt > 5.0) && (pt < 30.0) && ({} < abs(y)) && (abs(y) < {})".format(ybins[iy], ybins[iy+1])
	cut_name = "ybin_{}_{}".format(ybins[iy], ybins[iy+1]).replace(".", "p")
	fit_cuts[cut_name] = cut_str

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

def plot_data(tree, mass_range=[5.05, 5.5], cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data{}".format(tag), "c_data{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/\\Psi K^{\\pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=[5.05, 5.5], cut=""):
	ws = ROOT.RooWorkspace('ws')

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Signal: double Gaussian
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.1, 0.01, 1.0])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.5, 0.01, 1.0])")
	signal_dg = ws.factory(f"SUM::signal_dg(f1[0.9, 0., 1.]*g1, g2)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_dg, nsignal)

	# Background model: exponential
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-1.0, -100., -0.01])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	bkgd_JpsiPi = ws.factory(f"Gaussian::bkgd_JpsiPi(mass, 5.37, 0.02)")
	nJpsiPi = ROOT.RooFormulaVar("nJpsiPi", "nJpsiPi", "0.04 * nsignal", ROOT.RooArgList(nsignal))
	print(nJpsiPi)
	print(bkgd_JpsiPi)
	bkgd_JpsiPi_model = ROOT.RooExtendPdf("bkgd_JpsiPi_model", "bkgd_JpsiPi_model", bkgd_JpsiPi, nJpsiPi)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model, bkgd_JpsiPi_model))

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

def extract_yields(ws):
	return ws.var("nsignal").getVal()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--fit", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("data_Run2018*.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = sorted(fit_cuts.keys())

	if args.fit:
		for side in ["tag", "probe"]:
			chain = ROOT.TChain("Bcands_Bd_{}_HLT_Mu9_IP5|HLT_Mu9_IP6".format(side))
			#chain.Add("data_Run2018C_part3.root")
			#chain.Add("data_Run2018D_part1.root")
			#chain.Add("data_Run2018D_part2.root")
			for data_file in data_files:
				chain.Add(data_file)

			for cut_name in cuts:
				cut_str = fit_cuts[cut_name]
				plot_data(chain, cut=cut_str, tag="{}_{}".format(side, cut_name))

				ws, fit_result = fit_data(chain, cut=cut_str)
				ws.Print()
				fit_result.Print()
				ws_file = ROOT.TFile("fitws_data_Bd_{}_{}.root".format(side, cut_name), "RECREATE")
				ws.Write()
				fit_result.Write()
				ws_file.Close()

	if args.plots:
		for side in ["tag", "probe"]:
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bd_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				plot_fit(ws, tag="{}_{}".format(side, cut_name))

	if args.tables:
		yields = {}
		for side in ["tag", "probe"]:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bd_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
		pprint(yields)