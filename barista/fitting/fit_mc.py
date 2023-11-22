'''
Fit MC mass distributions
'''
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)

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


def fit_dg(tree, mass_range=[4.8, 5.85], cut=""):
	ws = ROOT.RooWorkspace('ws')

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	#mass = ws.var("mass")
	rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))

	# Construct model: double Gaussian
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.1, 0.01, 1.0])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.5, 0.01, 1.0])")
	g3 = ws.factory(f"Gaussian::g3(mass, mean, sigma3[0.8, 0.01, 1.0])")
	intsum = ws.factory(f"SUM::intsum(f2[0.5, 0., 1.]*g2, g1)")
	model = ws.factory(f"SUM::model(f3[0.5, 0., 1.]*g3, intsum)")
	#model = ws.factory("SUM::model(alpha[0.5, 0., 1.]*g1, g2)")

	# Perform fit
	nll = model.createNLL(rdataset, ROOT.RooFit.NumCPU(8))
	minimizer = ROOT.RooMinuit(nll)
	minimizer.migrad()

	# Generate return info
	fit_result = minimizer.save()
	rplot = mass.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot)
	model.plotOn(rplot)
	return ws, fit_result, rplot


if __name__ == "__main__":
	figure_dir = "/home/dyu7/BFrag/data/fits/mc"
	long_names = {
		"Bs": "Bs2PhiJpsi2KKMuMu",
		"Bu": "Bu2KJpsi2KMuMu",
		"Bd": "Bd2KstarJpsi2KPiMuMu"
	}
	for side in ["tag", "probe"]:
		f_Bs = ROOT.TFile(f"MCEfficiency_Bs.root", "READ")
		if side == "tag":
			sample = "inclusive"
		else:
			sample = "probefilter"
		tree = f_Bs.Get("Bcands_{}match_{}_{}".format(side, long_names["Bs"], sample))
		ws, fit_result, rplot = fit_dg(tree, mass_range=[BS_MASS*0.9, BS_MASS*1.1])
		ws.Print()
		fit_result.Print()
		canvas = ROOT.TCanvas("c_mcfit_Bs_{}".format(side), "c_mcfit_Bs_{}".format(side), 800, 600)
		rplot.Draw()
		canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
		f_Bs.Close()

		f_Bu = ROOT.TFile(f"MCEfficiency_Bu.root", "READ")
		if side == "tag":
			sample = "inclusive"
		else:
			sample = "probefilter"
		tree = f_Bu.Get("Bcands_{}match_{}_{}".format(side, long_names["Bu"], sample))
		ws, fit_result, rplot = fit_dg(tree, mass_range=[BU_MASS*0.9, BU_MASS*1.1])
		ws.Print()
		fit_result.Print()
		canvas = ROOT.TCanvas("c_mcfit_Bu_{}".format(side), "c_mcfit_Bu_{}".format(side), 800, 600)
		rplot.Draw()
		canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
		f_Bu.Close()

		f_Bd = ROOT.TFile(f"MCEfficiency_Bd.root", "READ")
		if side == "tag":
			sample = "inclusive"
		else:
			sample = "probefilter"
		tree = f_Bd.Get("Bcands_{}match_{}_{}".format(side, long_names["Bd"], sample))
		ws, fit_result, rplot = fit_dg(tree, mass_range=[BD_MASS*0.9, BD_MASS*1.1])
		ws.Print()
		fit_result.Print()
		canvas = ROOT.TCanvas("c_mcfit_Bd_{}".format(side), "c_mcfit_Bd_{}".format(side), 800, 600)
		rplot.Draw()
		canvas.SaveAs("{}/{}.png".format(figure_dir, canvas.GetName()))
		f_Bd.Close()

