'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
import pickle
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("include/TripleGaussianPdf_cc.so")
from ROOT import TripleGaussianPdf

from brazil.tdrstyle import SetTDRStyle
tdr_style = SetTDRStyle()

from brazil.cmslabel import CMSLabel, LuminosityLabel

import sys
sys.path.append(".")
from fitmc_Bd2KPiMuMu import make_signal_pdf_main, make_signal_pdf_swap
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	get_Bcands_name_data, get_MC_fit_params, get_MC_fit_errs, \
	MakeHypatia, MakeJohnson, MakeDoubleGaussian, MakeTripleGaussian, MakeTripleGaussianConstrained

# hack
#BD_FIT_WINDOW = [5.05, 5.4]
figure_dir = "/home/dyu7/BFrag/data/fits/data"

use_mc_constraints = True

rcache = [] # Prevent RooFit objects from disappearing

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

def plot_data(tree, mass_range=BD_FIT_WINDOW, cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dyu7/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BD_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False, side=None, selection="nominal", fitfunc=""):
	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

	if fitfunc == "poly":
		fitfunc = "johnson"
		bkgdfunc = "poly"
	else:
		bkgdfunc = "exp"

	# Turn tree into RooDataSet
	mass = ws.factory(f"mass[{(mass_range[0]+mass_range[1])/2}, {mass_range[0]}, {mass_range[1]}]")
	pt = ws.factory(f"pt[10.0, 0.0, 200.0]")
	y = ws.factory("y[0.0, -5.0, 5.0]")

	if correct_eff:
		w_eff = ws.factory("w_eff[0.0, 1.e5]")
		rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y, w_eff), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut), ROOT.RooFit.WeightVar("w_eff"))
	else:
		rdataset = ROOT.RooDataSet("fitData", "fitData", ROOT.RooArgSet(mass, pt, y), ROOT.RooFit.Import(tree), ROOT.RooFit.Cut(cut))
	ndata = rdataset.sumEntries()

	# Optional: bin data
	if binned:
		mass.setBins(BS_FIT_NBINS)
		rdatahist = ROOT.RooDataHist("fitDataBinned", "fitDataBinned", ROOT.RooArgSet(mass), rdataset)
		rdata = rdatahist
	else:
		rdata = rdataset

	# Signal: double Gaussian
	'''
	g1 = ws.factory(f"Gaussian::g1(mass, mean[{mass_range[0]}, {mass_range[1]}], sigma1[0.034, 0.001, 0.2])")
	g2 = ws.factory(f"Gaussian::g2(mass, mean, sigma2[0.15, 0.001, 0.2])")
	signal_dg = ws.factory(f"SUM::signal_dg(f1[0.95, 0.01, 0.99]*g1, g2)")
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_dg, nsignal)
	'''
	# Signal
	if fitfunc == "hypatia":
		signal_pdf_main = MakeHypatia(ws, mass_range, tag="_main", rcache=rcache) # make_signal_pdf_main(ws, mass_range)
		signal_pdf_swap = MakeHypatia(ws, mass_range, tag="_swap", rcache=rcache) # make_signal_pdf_swap(ws, mass_range)
	elif fitfunc == "johnson":
		signal_pdf_main = MakeJohnson(ws, mass_range, tag="_main", rcache=rcache) # make_signal_pdf_main(ws, mass_range)
		signal_pdf_swap = MakeJohnson(ws, mass_range, tag="_swap", rcache=rcache) # make_signal_pdf_swap(ws, mass_range)
	elif fitfunc == "2gauss":
		signal_pdf_main = MakeDoubleGaussian(ws, mass_range, tag="_main", rcache=rcache) # make_signal_pdf_main(ws, mass_range)
		signal_pdf_swap = MakeDoubleGaussian(ws, mass_range, tag="_swap", rcache=rcache) # make_signal_pdf_swap(ws, mass_range)
	elif fitfunc == "3gauss":
		signal_pdf_main = MakeTripleGaussianConstrained(ws, mass_range, tag="_main", rcache=rcache) # make_signal_pdf_main(ws, mass_range)
		#signal_pdf_swap = MakeDoubleGaussian(ws, mass_range, tag="_swap", rcache=rcache) # make_signal_pdf_swap(ws, mass_range)
		signal_pdf_swap = MakeJohnson(ws, mass_range, tag="_swap", rcache=rcache) # make_signal_pdf_swap(ws, mass_range)
	csig_main = ws.factory(f"csig_main[0.9, 0.7, 1.0]")
	signal_pdf = ROOT.RooAddPdf("signal_pdf", "signal_pdf", 
								ROOT.RooArgList(signal_pdf_main, signal_pdf_swap), 
								ROOT.RooArgList(csig_main))
	#getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	# Background model: exponential
	if bkgdfunc == "exp":
		bkgd_comb = ws.factory(f"Exponential::bkgd_comb(mass, alpha_bkgd[-3.66, -100., -0.01])")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)
	elif bkgdfunc == "poly":
		bkgd_comb = ws.factory(f"Chebychev::bkgd_comb(mass, {{p1[0., -10., 10.]}})")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)

	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_comb_model))

	# Load prefit results and add constraints
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	if use_mc_constraints:
		print("jkl;")
		constraints = {}
		with open(get_MC_fit_params("Bd", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			prefit_params = pickle.load(f)
		with open(get_MC_fit_errs("Bd", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			prefit_errs = pickle.load(f)

		# Remap some cuts due to low stats/bad MC fits
		cut_name_remapped = cut_name
		cut_name_remapped_swap = cut_name
		if cut_name in ["ybin_1p5_1p75", "ybin_1p75_2p0", "ybin_2p0_2p25"]:
			cut_name_remapped_swap = "ybin_1p25_1p5"

		# Set constraints on parameters
		# Which side to use?
		if selection == "nominal":
			mcside_main = f"{side}match"
			mcside_swap = f"{side}matchswap"
		elif selection == "HiTrkPt":
			mcside_main = f"{side}HiTrkPtmatch"
			mcside_swap = f"{side}HiTrkPtmatchswap"
		elif "MuonPt" in selection:
			mcside_main = f"{side}{selection}match"
			mcside_swap = f"{side}{selection}matchswap"
		elif selection == "MediumMuon":
			mcside_main = f"{side}{selection}match"
			mcside_swap = f"{side}{selection}matchswap"
		else:
			raise ValueError("asdfg Don't know what to do with selection {}".format(selection))

		# For tagx, just use the tag shapes
		if "tagx" in mcside_main:
			mcside_main = mcside_main.replace("tagx", "tag")
		if "tagx" in mcside_swap:
			mcside_swap = mcside_swap.replace("tagx", "tag")

		# For triple Gaussian, use Johnson for swap
		if fitfunc == "3gauss":
			with open(get_MC_fit_params("Bd", selection=selection, fitfunc="johnson", frozen=True), "rb") as f:
				prefit_params2 = pickle.load(f)
			with open(get_MC_fit_errs("Bd", selection=selection, fitfunc="johnson", frozen=True), "rb") as f:
				prefit_errs2 = pickle.load(f)
			prefit_params[mcside_swap] = prefit_params2[mcside_swap]
			prefit_errs[mcside_swap] = prefit_errs2[mcside_swap]

		# Fix all swap parameters
		for param_name in prefit_params[mcside_swap][cut_name].keys():
			if param_name[:2] != "j_":
				continue
			param_val = prefit_params[mcside_swap][cut_name_remapped_swap][param_name]
			#param_err = prefit_errs[side][cut_name_remapped][param_name] * err_multiplier
			ws.var(param_name).setVal(param_val)
			ws.var(param_name).setConstant(True)

		# Main parameters: complicated constraints...
		if fitfunc == "hypatia":
			for param_name in prefit_params[mcside_main][cut_name].keys():
				if "nsignal" in param_name or "zeta" in param_name or "beta" in param_name:
					continue

				err_multiplier = 1.0
				param_val = prefit_params[mcside_main][cut_name_remapped][param_name]
				param_err = prefit_errs[mcside_main][cut_name_remapped][param_name] * err_multiplier
				if param_err < 1.e-6 and not "hyp_mu" in param_name:
					print("WARNING : Param {} has small error {} ; cut_name = {}".format(param_name, param_err, cut_name))
					raise ValueError("Quitting")
					sys.exit(1)

				# Loose rectangular constraint on mean (via variable range)
				if "hyp_mu" in param_name:
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue

				if "hyp_lambda" in param_name or "hyp_sigma" in param_name:
					# For core width parameters, set very loose constraint
					param_err = max(abs(param_val / 2.), param_err * 10.)
				'''
				elif "hyp_n" in param_name:
					# Tail exponent n: restrict to max of n/10
					param_err = min(param_err, abs(param_val / 10.))
				elif "hyp_a" in param_name:
					# Tail distance from core: restrict to max of 0.5
					param_err = min(param_err, 0.5)
				'''

				# Fix tails
				if "hyp_a" in param_name or "hyp_n" in param_name:
					ws.var(param_name).setVal(param_val)
					ws.var(param_name).setConstant(True)
					continue

				# Adjust variable value and range to match constraints
				ws.var(param_name).setVal(param_val)
				param_min = max(ws.var(param_name).getMin(), param_val - 10. * param_err)
				param_max = min(ws.var(param_name).getMax(), param_val + 10. * param_err)
				if "hyp_lambda" in param_name:
					param_max = min(0., param_max)
				elif "hyp_a" in param_name or "hyp_n" in param_name or "hyp_sigma" in param_name:
					param_min = max(0., param_min)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				constraints[param_name] = ROOT.RooGaussian(
					f"constr_{param_name}", f"constr_{param_name}",
					ws.var(param_name),
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err),
				)
				print("Added constraint for {} (range [{}, {}])".format(param_name, param_min, param_max))
				print(constraints[param_name])
			# End loop over parameters

		elif fitfunc == "johnson":
			for param_name in prefit_params[mcside_main][cut_name].keys():
				if "nsignal" in param_name:
					continue

				err_multiplier = 1.0
				param_val = prefit_params[mcside_main][cut_name_remapped][param_name]
				param_err = prefit_errs[mcside_main][cut_name_remapped][param_name] * err_multiplier
				if param_err < 1.e-6 and not "mu" in param_name:
					print("WARNING : Param {} has small error {} ; cut_name = {}".format(param_name, param_err, cut_name))
					raise ValueError("Quitting")
					sys.exit(1)

				# Loose rectangular range for mean, no constraint
				if "j_mu" in param_name:
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue

				elif "j_lambda" in param_name:
					# For core width parameters, set very loose constraint
					param_err = max(abs(param_val / 2.), param_err * 10.)

				else:
					# For gamma and delta, constrain to MC error
					pass

				# Adjust variable value and range to match constraints
				ws.var(param_name).setVal(param_val)
				param_min = max(ws.var(param_name).getMin(), param_val - 10. * param_err)
				param_max = min(ws.var(param_name).getMax(), param_val + 10. * param_err)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				# Add constraint
				constraints[param_name] = ROOT.RooGaussian(
					f"constr_{param_name}", f"constr_{param_name}",
					ws.var(param_name),
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err),
				)
				print("Added constraint for {} (range [{}, {}])".format(param_name, param_min, param_max))
				print(constraints[param_name])
			# End loop over parameters
		elif fitfunc == "2gauss":
			raise NotImplementedError("2gauss not implemented!")
		elif fitfunc == "3gauss":
			for param_name in prefit_params[mcside_main][cut_name].keys():
				if "nsignal" in param_name:
					continue


				err_multiplier = 1.0
				param_val = prefit_params[mcside_main][cut_name_remapped][param_name]
				param_err = prefit_errs[mcside_main][cut_name_remapped][param_name] * err_multiplier
				if param_err < 1.e-6 and not "mean" in param_name:
					print("WARNING : Param {} has small error {} ; cut_name = {}".format(param_name, param_err, cut_name))
					raise ValueError("Quitting")
					sys.exit(1)

				# Fix sigma_3, aa, bb
				constant_params = ["tg_aa", "tg_bb", "tg_sigma1", "tg_sigma2", "tg_sigma3"]
				for name2 in constant_params:
					if name2 in param_name:
						print(param_name)
						ws.Print()
						ws.var(param_name).setVal(param_val)
						ws.var(param_name).setConstant(True)

				# Loose rectangular range for mean, no constraint
				if "tg_mean" in param_name:
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue

				elif "tg_cs" in param_name:
					# For core width parameters, set very loose constraint
					param_err = 1.0

				else:
					# For gamma and delta, constrain to MC error
					pass

				# Adjust variable value and range to match constraints
				ws.var(param_name).setVal(param_val)
				param_min = max(ws.var(param_name).getMin(), param_val - 10. * param_err)
				param_max = min(ws.var(param_name).getMax(), param_val + 10. * param_err)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				# Add constraint
				constraints[param_name] = ROOT.RooGaussian(
					f"constr_{param_name}", f"constr_{param_name}",
					ws.var(param_name),
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err),
				)
				print("Added constraint for {} (range [{}, {}])".format(param_name, param_min, param_max))
				print(constraints[param_name])
			# End loop over parameters
		# End if hypatia or johnson

		# Fix swap fraction
		prefit_nmain = prefit_params[mcside_main][cut_name]["nsignal"]
		prefit_nswap = prefit_params[mcside_swap][cut_name]["nsignal"]
		if prefit_nmain == 0:
			print(f"Skipping this point due to prefit_nmain == 0\n {incut}_{cut_name}_{side}_{selection}_{fitfunc}")
			return None, None
		print("Constraint on main fraction = {}".format(prefit_nmain / max(prefit_nmain + prefit_nswap, 1.e-20)))
		csig_main.setVal(prefit_nmain / (prefit_nmain + prefit_nswap))
		csig_main.setConstant(True)
		#constraints["csig_main"] = ROOT.RooGaussian(
		#		f"constr_csig_main", f"constr_csig_main",
		#		csig_main,
		#		ROOT.RooFit.RooConst(prefit_nmain / (prefit_nmain + prefit_nswap)),
		#		ROOT.RooFit.RooConst(0.1),
		#)
		print("Constraints: ")
		print(constraints)
		constraints_set = ROOT.RooArgSet()
		for constraint in constraints.values():
			constraints_set.add(constraint)
		fit_args.append(ROOT.RooFit.ExternalConstraints(constraints_set))

	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	# Tweaks
	if fitfunc == "hypatia":
		if cut_name == "ptbin_12p0_13p0":
			ws.var('alpha_bkgd').setVal(-2.2882e+00)
			ws.var('hyp_a2_main').setVal(2.0093e+00)
			ws.var('hyp_a2_swap').setVal(3.0862e+00)
			ws.var('hyp_a_main').setVal(6.2810e+00)
			ws.var('hyp_a_swap').setVal(2.0000e+00)
			ws.var('hyp_lambda_main').setVal(-1.3063e+00)
			ws.var('hyp_lambda_swap').setVal(-9.7521e-01)
			ws.var('hyp_mu_main').setVal(5.2797e+00)
			ws.var('hyp_mu_swap').setVal(5.2723e+00)
			ws.var('hyp_n2_main').setVal(8.0398e+00)
			ws.var('hyp_n2_swap').setVal(5.0000e-01)
			ws.var('hyp_n_main').setVal(5.0113e-01)
			ws.var('hyp_n_swap').setVal(7.3646e-01)
			ws.var('hyp_sigma_main').setVal(2.7956e-02)
			ws.var('hyp_sigma_swap').setVal(4.9041e-02)

		if cut_name == "ptbin_13p0_14p0":
			ws.var('alpha_bkgd').setVal(-2.6480e+00)
			ws.var('hyp_a2_main').setVal(2.0000e+00)
			ws.var('hyp_a2_swap').setVal(3.0846e+00)
			ws.var('hyp_a_main').setVal(6.1186e+01)
			ws.var('hyp_a_swap').setVal(2.0000e+00)
			ws.var('hyp_lambda_main').setVal(-1.1357e+00)
			ws.var('hyp_lambda_swap').setVal(-1.0681e+00)
			ws.var('hyp_mu_main').setVal(5.2798e+00)
			ws.var('hyp_sigma_swap').setVal(5.0494e-02)

		elif cut_name == "ptbin_14p0_15p0":
			ws.var('alpha_bkgd').setVal(-2.8828e+00)
			ws.var('hyp_a2_main').setVal(2.0000e+00)
			ws.var('hyp_a2_swap').setVal(2.3633e+00)
			ws.var('hyp_a_main').setVal(2.2866e+01)
			ws.var('hyp_a_swap').setVal(2.0000e+00)
			ws.var('hyp_lambda_main').setVal(-1.1457e+00)
			ws.var('hyp_lambda_swap').setVal(-1.0425e+00)
			ws.var('hyp_mu_main').setVal(5.2796e+00)
			ws.var('hyp_mu_swap').setVal(5.2686e+00)
			ws.var('hyp_n2_main').setVal(2.0000e+01)
			ws.var('hyp_n2_swap').setVal(7.8752e+00)
			ws.var('hyp_n_main').setVal(1.5143e+00)
			ws.var('hyp_n_swap').setVal(7.2163e-01)
			ws.var('hyp_sigma_main').setVal(2.1643e-02)
			ws.var('hyp_sigma_swap').setVal(5.3602e-02)
		elif cut_name == "ptbin_23p0_26p0":
			ws.var('alpha_bkgd').setVal(-3.1535e+00)
			ws.var('hyp_a2_main').setVal(2.0000e+00)
			ws.var('hyp_a2_swap').setVal(2.0000e+00)
			ws.var('hyp_a_main').setVal(7.5160e+01)
			ws.var('hyp_a_swap').setVal(2.0018e+00)
			ws.var('hyp_lambda_main').setVal(-1.3445e+00)
			ws.var('hyp_lambda_swap').setVal(-2.9883e+00)
			ws.var('hyp_mu_main').setVal(5.2783e+00)
			ws.var('hyp_sigma_swap').setVal(2.1476e-02)
			ws.var('nbkgd').setVal(1.0919e+04)
			ws.var('nsignal').setVal(2.3436e+04)

			ws.var('alpha_bkgd').setMin(-4.0)
		elif cut_name == "ptbin_26p0_29p0":
			ws.var('alpha_bkgd').setVal(-3.2752e+00)
			ws.var('hyp_a2_main').setVal(2.0234e+00)
			ws.var('hyp_a2_swap').setVal(9.6296e+01)
			ws.var('hyp_a_main').setVal(1.4653e+01)
			ws.var('hyp_a_swap').setVal(2.0126e+00)
			ws.var('hyp_lambda_main').setVal(-1.5556e+00)
			ws.var('hyp_lambda_swap').setVal(-1.3890e+00)
			ws.var('hyp_mu_main').setVal(5.2791e+00)
			ws.var('hyp_mu_swap').setVal(5.2717e+00)
			ws.var('hyp_n2_main').setVal(7.0542e+00)
			ws.var('hyp_n2_swap').setVal(5.6387e-01)
			ws.var('hyp_n_main').setVal(4.9378e+00)
			ws.var('hyp_n_swap').setVal(5.0079e-01)
			ws.var('hyp_sigma_main').setVal(2.4151e-02)
			ws.var('hyp_sigma_swap').setVal(4.7164e-02)

		elif cut_name == "ybin_0p75_1p0":
			ws.var('alpha_bkgd').setVal(-3.4694e+00)
			ws.var('hyp_a2_main').setVal(2.8277e+01)
			ws.var('hyp_a2_swap').setVal(2.0000e+00)
			ws.var('hyp_a_main').setVal(2.0000e+00)
			ws.var('hyp_a_swap').setVal(2.0559e+00)
			ws.var('hyp_lambda_main').setVal(-1.9068e+00)
			ws.var('hyp_lambda_swap').setVal(-8.6448e-01)
			ws.var('hyp_mu_main').setVal(5.2801e+00)
			ws.var('hyp_mu_swap').setVal(5.2641e+00)
			ws.var('hyp_n2_main').setVal(1.6352e+01)
			ws.var('hyp_n2_swap').setVal(5.0000e-01)
			ws.var('hyp_n_main').setVal(1.7772e+01)
			ws.var('hyp_n_swap').setVal(2.0008e+00)
			ws.var('hyp_sigma_main').setVal(2.9054e-02)
			ws.var('hyp_sigma_swap').setVal(2.2502e-02)
		elif cut_name == "ybin_1p0_1p25":
			ws.var('alpha_bkgd').setVal(-4.5156e+00)
			ws.var('hyp_a2_main').setVal(2.0000e+00)
			ws.var('hyp_a2_swap').setVal(2.0000e+00)
			ws.var('hyp_a_main').setVal(1.0000e+02)
			ws.var('hyp_a_swap').setVal(2.0004e+00)
			ws.var('hyp_lambda_main').setVal(-1.5513e+00)
			ws.var('hyp_lambda_swap').setVal(-9.6528e-01)
			ws.var('hyp_mu_main').setVal(5.2793e+00)
			ws.var('hyp_mu_swap').setVal(5.2671e+00)
			ws.var('hyp_n2_main').setVal(5.0001e-01)
			ws.var('hyp_n2_swap').setVal(5.0008e-01)
			ws.var('hyp_n_main').setVal(1.9832e+01)
			ws.var('hyp_n_swap').setVal(1.8075e+00)
			ws.var('hyp_sigma_main').setVal(3.2129e-02)
			ws.var('hyp_sigma_swap').setVal(3.5593e-02)

	# Perform fit
	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	return ws, fit_result

def plot_fit(ws, fit_result, tag="", subfolder="", text=None, binned=False, correct_eff=False):
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetOptTitle(0)

	model = ws.pdf("model")
	if binned:
		rdataset = ws.data("fitDataBinned")
	else:
		rdataset = ws.data("fitData")
	xvar = ws.var("mass")

	canvas = ROOT.TCanvas("c_datafit_{}".format(tag), "c_datafit_{}".format(tag), 1400, 1400)

	top = ROOT.TPad("top", "top", 0., 0.3, 1., 1.)
	top.SetBottomMargin(0.025)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data2"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), 
					ROOT.RooFit.MarkerColor(ROOT.kBlack), 
					ROOT.RooFit.MarkerSize(0.0), 
					ROOT.RooFit.MarkerStyle(20))
	model.plotOn(rplot, ROOT.RooFit.Name("signal_total"), ROOT.RooFit.Components("signal_model"), 
					ROOT.RooFit.LineColor(ROOT.kRed), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.FillColor(ROOT.kRed), 
					ROOT.RooFit.FillStyle(3007), 
					ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal_main"), ROOT.RooFit.Components("signal_main"), 
					ROOT.RooFit.LineColor(ROOT.kOrange+7), 
					ROOT.RooFit.LineWidth(1))
	model.plotOn(rplot, ROOT.RooFit.Name("signal_swap"), ROOT.RooFit.Components("signal_swap"), 
					ROOT.RooFit.LineColor(ROOT.kGreen+2), 
					ROOT.RooFit.LineWidth(1))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_comb"), ROOT.RooFit.Components("bkgd_comb_model"), 
					ROOT.RooFit.LineColor(ROOT.kBlue+2), 
					ROOT.RooFit.LineWidth(1))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"), 
					ROOT.RooFit.LineColor(ROOT.kBlue), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.LineStyle(1))		
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), 
					ROOT.RooFit.MarkerColor(ROOT.kBlack), 
					ROOT.RooFit.MarkerSize(1.0), 
					ROOT.RooFit.MarkerStyle(20), 
					ROOT.RooFit.DrawOption("p e same"))
	rplot.GetXaxis().SetTitleSize(0)
	rplot.GetXaxis().SetLabelSize(0)
	rplot.GetYaxis().SetLabelSize(40)
	rplot.GetYaxis().SetTitleSize(40)
	rplot.GetYaxis().SetTitleOffset(2.25)
	rplot.SetMaximum(1.3 * rplot.GetMaximum())
	rplot.Draw()

	legend = ROOT.TLegend(0.61, 0.49, 0.95, 0.9)
	legend.SetFillColor(0)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextFont(43)
	legend.SetTextSize(40)
	legend.AddEntry("data", "Data", "lp")
	legend.AddEntry("fit", "Signal + bkgd. fit", "l")
	legend.AddEntry("signal_total", "B_{d}#rightarrowJ/#psi K^{*} (total)", "lf")
	legend.AddEntry("signal_main", "B_{d}#rightarrowJ/#psi K^{*} (main)", "lf")
	legend.AddEntry("signal_swap", "B_{d}#rightarrowJ/#psi K^{*} (swap)", "lf")
	legend.AddEntry("bkgd_comb", "Comb. bkgd.", "l")
	legend.Draw()

	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.3)
	bottom.SetTopMargin(0.03)
	#bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	#binning = ROOT.RooBinning(BD_FIT_NBINS, BD_FIT_WINDOW[0], BD_FIT_WINDOW[1])
	binning = xvar.getBinning()
	if binned:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar)
	else:
		data_hist = ROOT.RooAbsData.createHistogram(rdataset, "data_hist", xvar, ROOT.RooFit.Binning(binning))
	data_hist.Sumw2()
	fit_binned = model.generateBinned(ROOT.RooArgSet(xvar), 0, True)
	fit_hist = fit_binned.createHistogram("model_hist", xvar, ROOT.RooFit.Binning(binning))
	pull_hist = data_hist.Clone()
	pull_hist.Reset()
	chi2 = 0.

	# Count fit parameters (free only)
	nparams = 0
	for param in fit_result.floatParsFinal(): 
		if  param.isConstant():
			continue
		if param.GetName() in ["j_delta_main", "j_gamma_main"]:
			print(f"Skipping {param.GetName()} because it is constrained to MC")
			continue
		print(f"DEBUGG : Counting param {param.GetName()} as -1df")
		nparams += 1

	ndf = -1 * nparams
	for xbin in range(1, pull_hist.GetNbinsX()+1):
		data_val = data_hist.GetBinContent(xbin)
		data_unc = data_hist.GetBinError(xbin)
		fit_val = fit_hist.GetBinContent(xbin)
		if correct_eff:
			fit_unc = fit_hist.GetBinError(xbin) * math.sqrt(data_unc**2 / max(data_val, 1.e-10))
		else:
			fit_unc = math.sqrt(fit_val)
		pull_val = (data_val - fit_val) / max(fit_unc, 1.e-10)
		#print(f"Pull xbin {xbin} = ({data_val} - {fit_val}) / ({fit_unc}) = {pull_val}")
		pull_hist.SetBinContent(xbin, pull_val)
		pull_hist.SetBinError(xbin, 0)
		chi2 += pull_val**2
		ndf += 1
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}#pi^{#mp}} [GeV]")
	pull_hist.GetYaxis().SetTitle("Pull w.r.t. fit [#sigma]")
	pull_hist.GetXaxis().SetLabelSize(40)
	pull_hist.GetXaxis().SetTitleSize(40)
	pull_hist.GetYaxis().SetLabelSize(40)
	pull_hist.GetYaxis().SetTitleSize(40)
	pull_hist.SetMarkerStyle(20)
	pull_hist.SetMarkerSize(1.0)	
	#pull_hist.SetMarkerStyle(20)
	#pull_hist.SetMarkerSize(1)
	#pull_hist.GetXaxis().SetTitleSize(0.06)
	#pull_hist.GetXaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleSize(0.06)
	#pull_hist.GetYaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleOffset(0.6)
	pull_hist.SetMinimum(-4.)
	pull_hist.SetMaximum(4.)
	pull_hist.Draw("p")

	zero = ROOT.TLine(BD_FIT_WINDOW[0], 0., BD_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kBlack)
	zero.SetLineStyle(1)
	zero.SetLineWidth(2)
	zero.SetLineWidth(2)
	zero.Draw()

	canvas.cd()
	top.cd()
	if text:
		chi2text = f"#chi^{{2}}/ndof={round(chi2/ndf, 3):.3f}"
		totaltext = f"#splitline{{{text}}}{{{chi2text}}}"
		#print(totaltext)
		textbox = ROOT.TLatex(0.22, 0.55, totaltext)
		textbox.SetNDC()
		textbox.SetTextSize(0.047)
		textbox.SetTextFont(42)
		textbox.Draw()

	cmslabel = CMSLabel()
	cmslabel.sublabel.text = "Preliminary"
	cmslabel.scale = 0.9
	cmslabel.draw()

	lumilabel = LuminosityLabel("34.7 fb^{-1} (13 TeV)")	
	lumilabel.scale = 0.75
	lumilabel.draw()

	os.system(f"mkdir -pv {figure_dir}/{subfolder}")
	canvas.SaveAs("{}/{}/{}.png".format(figure_dir, subfolder, canvas.GetName()))
	canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))
	#canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

	top.SetLogy()
	rplot.SetMinimum(max(data_hist.GetMinimum() / 10., 0.1))
	rplot.SetMaximum(data_hist.GetMaximum() * 30.)
	canvas.SaveAs("{}/{}/{}_logy.png".format(figure_dir, subfolder, canvas.GetName()))
	#canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

	ROOT.SetOwnership(canvas, False)
	ROOT.SetOwnership(top, False)
	ROOT.SetOwnership(bottom, False)

def extract_yields(ws):
	return (ws.var("nsignal").getVal(), ws.var("nsignal").getError())


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bd fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	parser.add_argument("--binned", action="store_true", help="Do binned fit")
	parser.add_argument("--correct_eff", action="store_true", help="Apply efficiency correction before fitting")
	parser.add_argument("--fitparams", action="store_true", help="Print fit parameters")
	parser.add_argument("--selection", type=str, default="nominal", help="Selection name (nominal, HiTrkPt, ...)")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="Fit function name (hypatia, johnson)")
	parser.add_argument("--trigger_strategies", type=str, default="HLT_all,HLT_Mu7,HLT_Mu9,HLT_Mu9_IP5,HLT_Mu9_IP6", help="Trigger strategies to run")
	parser.add_argument("--sides", type=str, default="tag,probe,tagx", help="Sides to run")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dyu7/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = {"tag": ["inclusive"], "probe": ["inclusive"]}
	elif args.all:
		cuts = fit_cuts
	elif args.some:
		cuts_list = args.some.split(",")
		if not set(cuts_list).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))
		cuts = {"tag": cuts_list, "probe": cuts_list}

	cuts["tagx"] = cuts["tag"]

	trigger_strategies_to_run = args.trigger_strategies.split(",") #["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"
	sides_to_run = args.sides.split(",")
	if args.fits:
		for side in sides_to_run:
		#for side in ["probeMaxPt", "tagMaxPt"]:
			trigger_strategies = {
				"HLT_all": ["HLT_Mu7_IP4", "HLT_Mu9_IP5_only", "HLT_Mu9_IP6_only", "HLT_Mu12_IP6_only"],
				"HLT_Mu9": ["HLT_Mu9_IP5", "HLT_Mu9_IP6_only"],
				"HLT_Mu7": ["HLT_Mu7_IP4"],
				"HLT_Mu9_IP5": ["HLT_Mu9_IP5"],
				"HLT_Mu9_IP6": ["HLT_Mu9_IP6"],
				
			}
			for trigger_strategy in trigger_strategies_to_run:
				chain = ROOT.TChain()
				for trigger in trigger_strategies[trigger_strategy]:
					tree_name = get_Bcands_name_data(btype="Bd", trigger=trigger, side=side, selection=args.selection)
					for data_file in data_files:
						chain.Add(f"{data_file}/{tree_name}")

				print("Total entries = {}".format(chain.GetEntries()))
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"					
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bd_{}".format(save_tag))

					try:
						ws, fit_result = fit_data(chain,
												incut=cut_str,
												cut_name=cut_name,
												binned=args.binned,
												correct_eff=args.correct_eff,
												side=side,
												fitfunc=args.fitfunc, 
												selection=args.selection)
						if not ws:
							continue
					except ValueError as err:
						print(err)
						sys.exit(1)
					ws.Print()
					print("asdf Fit results for {}_{}_{}".format(side, cut_name, trigger_strategy))
					fit_result.Print()
					ws_file = ROOT.TFile(f"Bd/fitws_data_Bd_{save_tag}.root", "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

					# Clear cache
					del ws
					rcache = []

	if args.plots:
		for side in sides_to_run:
		#for side in ["probeMaxPt", "tagMaxPt"]:
			for trigger_strategy in trigger_strategies_to_run:
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"					
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bd/fitws_data_Bd_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					fit_result_name = "fitresult_model_fitData"
					if args.binned:
						fit_result_name += "Binned"
					fit_result = ws_file.Get(fit_result_name)

					if args.binned:
						subfolder = "binned"
					else:
						subfolder = "unbinned"
					if args.correct_eff:
						subfolder += "_correcteff"

					if "pt" in cut_name:
						pt_line = fit_text[cut_name]
						if "probe" in side:
							y_line = r"|y|#in[0, 2.25)"
						else:
							y_line = r"|y|#in[0, 1.5)"
						plot_text = f"#splitline{pt_line}{y_line}"
					else:
						pt_line = "p_{T}#in(13, 50)"
						y_line = fit_text[cut_name]
						plot_text = f"#splitline{{{pt_line}}}{{{y_line}}}"

					plot_fit(ws, fit_result, tag="Bd_{}".format(save_tag), subfolder=subfolder, text=plot_text, binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in sides_to_run:
		#for side in ["probeMaxPt", "tagMaxPt"]:
			yields[side] = {}
			for trigger_strategy in trigger_strategies_to_run:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"					
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bd/fitws_data_Bd_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		yields_file = f"Bd/yields_{args.fitfunc}_{args.selection}"
		#yields_file = "Bd/yields_maxPt_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		print("Saving yields to {}".format(yields_file))
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)
		pprint(yields)

	if args.fitparams:
		for side in ["probe", "tag", "tagx"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bd/fitws_data_Bd_{}.root".format(save_tag)
					if not os.path.isfile(fitresult_file):
						print("No fit result for {}, skipping".format(save_tag))
						continue
					ws_file = ROOT.TFile(fitresult_file, "READ")
					ws_file.ls()

					#ws_file.ls()
					fit_result = ws_file.Get(f"fitresult_model_fitData{'Binned' if args.binned else ''}")
					print("\n*** Printing fit results for {} ***".format(save_tag))
					if fit_result:
						fit_result.Print()
					else:
						print("Didn't find fit result object {} in file {}".format("fitresult_model_fitData", fitresult_file))
						ws_file.Print()
						continue
