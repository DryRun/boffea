'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf2.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf2.cc+")
#ROOT.gSystem.Load("include/TripleGaussianPdf2.so")

#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
ROOT.gROOT.ProcessLine(open('include/MyErfc.cc').read())
#from ROOT import TripleGaussian
#from ROOT import TripleGaussianPdf2
from ROOT import MyErfc

use_mc_constraints = True

rcache = [] # Prevent RooFit objects from disappearing

figure_dir = "/home/dryu/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	MakeHypatia

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

def plot_data(tree, mass_range=BU_FIT_WINDOW, cut="", tag=""):
	h_data = ROOT.TH1D("h_data", "h_data", 100, mass_range[0], mass_range[1])
	tree.Draw("mass >> h_data", cut)
	c = ROOT.TCanvas("c_data_{}".format(tag), "c_data_{}".format(tag), 800, 600)
	h_data.SetMarkerStyle(20)
	h_data.GetXaxis().SetTitle("Fitted M_{J/#Psi K^{#pm}} [GeV]")
	h_data.Draw()
	c.SaveAs("/home/dryu/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BU_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False, save_tag=None, trigger_strategy=None):

	ws = ROOT.RooWorkspace('ws')

	cut = f"{incut} && (mass > {mass_range[0]}) && (mass < {mass_range[1]})"
	if correct_eff:
		# Ignore very large weights, which are probably from 0 efficiency bins
		cut += " && (w_eff < 1.e6)"

	#tree.Print()
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

	# Signal: hypatia
	signal_hyp = MakeHypatia(ws, mass_range, rcache=rcache)
	getattr(ws, "import")(signal_hyp, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_hyp, nsignal)

	# Background model: exponential + (Bu > Jpsi pi gaussian) + (partial reco ERF)
	bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.75, -20., -0.1])")
	nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
	bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)

	#bkgd_JpsiPi = ws.factory(f"Gaussian::bkgd_JpsiPi(mass, 5.37, 0.038)")
	#nJpsiPi = ROOT.RooFormulaVar("nJpsiPi", "nJpsiPi", "0.04 * nsignal", ROOT.RooArgList(nsignal))
	#print(nJpsiPi)
	#print(bkgd_JpsiPi)
	#bkgd_JpsiPi_model = ROOT.RooExtendPdf("bkgd_JpsiPi_model", "bkgd_JpsiPi_model", bkgd_JpsiPi, nJpsiPi)
	bkgd_jpsipi_model = make_JpsiPi(ws, cut_name)

	erfc_x0 = ws.factory(f"erfc_x0[5.12, 5.07, 5.2]")
	erfc_width = ws.factory(f"erfc_width[0.015, 0.012, 0.04]")
	bkgd_erfc = ws.factory(f"GenericPdf::bkgd_erfc('MyErfc(mass, erfc_x0, erfc_width)', {{mass, erfc_x0, erfc_width}})")
	#erfc_arg = ROOT.RooFormulaVar("erfc_arg", "erfc_arg", "(mass - erfc_x) / (erfc_width)", ROOT.RooArgList(mass, erfc_x, erfc_width))
	#erfc_tf1 = ROOT.TF1("erfc", pyerfc, BU_FIT_WINDOW[0], 5.2, 2)
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erfc_tf1, erfc_arg)
	#ROOT.gInterpreter.ProcessLine('RooAbsPdf* myerfc = RooFit::bindPdf("erfc_pdf", TMath::Erfc, erfc_arg);')
	#x = ROOT.x
	#erfc_pdf = ROOT.myerfc
	#erfc_pdf = ROOT.RooFit.bindPdf("Erfc", erf, erfc_arg)
	nbkgd_erfc = ws.factory(f"nbkgd_erfc[{ndata*0.03}, 0.0, {ndata*2.0}]")
	bkgd_erfc_model = ROOT.RooExtendPdf("bkgd_erfc_model", "bkgd_erfc_model", bkgd_erfc, nbkgd_erfc)

	# For ptbin 5-10, it looks like there's no partial background? Not resolvable, anyways.
	'''
	if cut_name == "ptbin_5p0_10p0":
		nbkgd_erfc.setVal(0)
		nbkgd_erfc.setConstant(True)
		erfc_x0.setConstant(True)
		erfc_width.setConstant(True)

		# Also impose a minimum on the background, otherwise the signal can consume all the background
		nbkgd_exp.setMin(ndata * 0.05)

	elif cut_name == "ptbin_13p0_14p0":
		# Problem with huge tail. Chop it off.
		nbkgd_exp.setMin(5.9e+03)

	elif cut_name == "ptbin_34p0_45p0" and binned:
		sigma3.setMax(0.09)
		nbkgd_exp.setMin(830)
	'''
	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model, bkgd_jpsipi_model, bkgd_erfc_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	
	constraints = {}
	if use_mc_constraints:
		# Load constraints from MC fit
		with open("Bu/fitparams_MC_Bu_hypatia_frozen.pkl", 'rb') as f:
			mc_fit_params = pickle.load(f)
		with open("Bu/fiterrs_MC_Bu_hypatia_frozen.pkl", 'rb') as f:
			mc_fit_errors = pickle.load(f)

		print(f"{cut_name} : adding constraints")
		print("Constraint central values:")
		pprint(mc_fit_params[cut_name])
		print("Constraint widths:")
		pprint(mc_fit_errors[cut_name])
		for param_name in ["hyp_lambda", "hyp_sigma", "hyp_mu", "hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
			var = ws.var(param_name)
			print("Adding constraint for {}".format(param_name))
			var.setVal(mc_fit_params[cut_name][param_name.replace("2", "")])

			err_multiplier = 1.0
			param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
			param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")] * err_multiplier
			if param_err < 1.e-6:
				print("WARNING : Param {} has small error {}".format(param_name, param_err))
				raise ValueError("Quitting")
				sys.exit(1)

			# Loose rectangular constraint on mean (via variable range)
			if "hyp_mu" in param_name:
				ws.var(param_name).setMin(mc_fit_params[cut_name][param_name] - 0.1)
				ws.var(param_name).setMax(mc_fit_params[cut_name][param_name] + 0.1)
				continue

			# Fix tails
			if "hyp_a" in param_name or "hyp_n" in param_name:
				ws.var(param_name).setVal(param_val)
				ws.var(param_name).setConstant(True)
				continue

			if "hyp_lambda" in param_name or "hyp_sigma" in param_name:
				# For core width parameters, set very loose constraint
				param_err = max(abs(param_val / 2.), param_err * 10.)
			#if "hyp_sigma" in param_name:
			#	param_val = param_val * 1.2
			#elif "hyp_n" in param_name:
			#	# Tail exponent n: restrict to max of n/10
			#	param_err = min(param_err, abs(param_val / 20.))
			#elif "hyp_a" in param_name:
			#	# Tail distance from core: restrict to max of 0.5
			#	param_err = min(param_err, 0.25)

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
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				var, 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])

	# Probe: use tag fits to constrain erfc parameters
	if "probe" in save_tag and not "probebins" in save_tag:
		save_tag_tag = save_tag.replace("probeMaxPt", "probe").replace("probe", "tag_probebins")
		tag_file = ROOT.TFile("Bu/fitws_hyp_data_Bu_{}.root".format(save_tag_tag), "READ")
		tag_file.ls()
		frname = "fitresult_model_fitData"
		if binned:
			frname += "Binned"
		tag_fitresult = tag_file.Get(frname)
		if not tag_fitresult:
			print("ERROR : Couldn't find fit result in file {}!".format(tag_file.GetPath()))
			raise ValueError("Quitting")

		tag_parameters = {}
		for var in [erfc_width, erfc_x0]:
			varname = var.GetName()
			param = tag_fitresult.floatParsFinal().find(varname)
			param_val = param.getVal()
			param_err = param.getError()
			var.setVal(param_val)
			var.setMin(max(param_val - 50.*param_err, var.getMin()))
			var.setMax(min(param_val + 50.*param_err, var.getMax()))
			constraints[varname] = ROOT.RooGaussian(
				"constr_{}".format(varname), 
				"constr_{}".format(varname), 
				var, 
				ROOT.RooFit.RooConst(param_val), 
				ROOT.RooFit.RooConst(10 * param_err)
				)

	# Tweaks
	if cut_name == "ptbin_8p0_13p0":
		ws.var('alpha').setVal(-3.1524e-01)
		ws.var('erfc_width').setVal(3.5406e-02)
		ws.var('erfc_x0').setVal(5.1271e+00)
		ws.var('hyp_lambda').setVal(-3.4036e-01)
		ws.var('hyp_mu').setVal(5.2782e+00)
		ws.var('hyp_sigma').setVal(1.8653e-02)

	if cut_name == "ptbin_13p0_18p0" and (trigger_strategy == "HLT_Mu9" or trigger_strategy == "HLT_all"):
		ws.var('alpha').setVal(-4)
		ws.var('erfc_width').setVal(0.02)
		ws.var('erfc_x0').setVal(5.1387e+00)
		#ws.var('hyp_a').setVal(3.0510e+00)
		#ws.var('hyp_a2').setVal(3.05)
		ws.var('hyp_lambda').setVal(-1.3698e+00)
		ws.var('hyp_mu').setVal(5.2790e+00)
		#ws.var('hyp_n').setVal(1.1105e+00)
		#ws.var('hyp_n2').setVal(1.1902e+00)
		ws.var('hyp_sigma').setVal(3.3608e-02)
		ws.var('nbkgd').setVal(2.2665e+03)
		ws.var('nbkgd_erfc').setVal(4e+02)
		ws.var('nsignal').setVal(2.5e+03)
	elif cut_name == "ptbin_11p0_12p0" and binned == False:
		if trigger_strategy == "HLT_Mu7":
			ws.var('erfc_x0').setVal(5.1421e+00)
			ws.var('hyp_a').setVal(1.0001e+00)
			ws.var('hyp_a2').setVal(1.0036e+00)
			ws.var('hyp_lambda').setVal(-1.0237e+00)
			ws.var('hyp_mu').setVal(5.2783e+00)
			ws.var('hyp_n').setVal(2.1542e+00)
			ws.var('hyp_n2').setVal(4.1932e+00)
			ws.var('hyp_sigma').setVal(2.5424e-02)
			ws.var('nbkgd').setVal(4.0560e+03)
			ws.var('nbkgd_erfc').setVal(6.9992e+02)
			ws.var('nsignal').setVal(4.7809e+03)
		elif trigger_strategy == "HLT_all":
			ws.var('erfc_width').setVal(0.02)
			ws.var('erfc_x0').setVal(5.1421e+00)
			ws.var('hyp_a').setVal(1.0001e+00)
			ws.var('hyp_a2').setVal(1.0036e+00)
			ws.var('hyp_lambda').setVal(-1.0237e+00)
			ws.var('hyp_mu').setVal(5.2783e+00)
			ws.var('hyp_n').setVal(2.1542e+00)
			ws.var('hyp_n2').setVal(4.1932e+00)
			ws.var('hyp_sigma').setVal(2.5424e-02)
			ws.var('nbkgd').setVal(4.1040e+03 * 1.3)
			ws.var('nbkgd_erfc').setVal(7.6900e+02)
			ws.var('nsignal').setVal(5.0345e+03 * 0.7)

		#ws.var('alpha').setVal(-1.0)
		#ws.var('erfc_width').setVal(0.03)
		#ws.var('erfc_x0').setVal(5.1501e+00)
		#ws.var('hyp_lambda').setVal(-1.0571e+00)
		#ws.var('hyp_mu').setVal(5.2782e+00)
		#ws.var('hyp_sigma').setVal(3.0e-02)
		#ws.var("hyp_a").setConstant(False)
		#ws.var("hyp_n").setConstant(False)
		#ws.var("hyp_a2").setConstant(False)
		#ws.var("hyp_n2").setConstant(False)
		#for param_name in ["hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
		#	param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
		#	param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")]
		#	constraints[param_name] = ROOT.RooGaussian(
		#		"constr_{}".format(param_name), 
		#		"constr_{}".format(param_name), 
		#		ws.var(param_name), 
		#		ROOT.RooFit.RooConst(param_val),
		#		ROOT.RooFit.RooConst(param_err))
		#	print(constraints[param_name])
	elif cut_name == "ptbin_11p0_12p0" and trigger_strategy == "HLT_Mu7":
		ws.var('alpha').setVal(-1.0)
		ws.var('erfc_width').setVal(0.03)
		ws.var('erfc_x0').setVal(5.1501e+00)
		ws.var('hyp_lambda').setVal(-1.0571e+00)
		ws.var('hyp_mu').setVal(5.2782e+00)
		ws.var('hyp_sigma').setVal(3.0e-02)
	elif cut_name == "ptbin_11p0_12p0" and trigger_strategy == "HLT_all":
		ws.var('alpha').setVal(-1.0)
		ws.var('erfc_width').setVal(0.03)
		ws.var('erfc_x0').setVal(5.1501e+00)
		ws.var('hyp_lambda').setVal(-1.0571e+00)
		ws.var('hyp_mu').setVal(5.2782e+00)
		ws.var('hyp_sigma').setVal(3.0e-02)
		ws.var("hyp_a").setConstant(False)
		ws.var("hyp_n").setConstant(False)
		ws.var("hyp_a2").setConstant(False)
		ws.var("hyp_n2").setConstant(False)
		for param_name in ["hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
			param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
			param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")]
			constraints[param_name] = ROOT.RooGaussian(
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				ws.var(param_name), 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])

	elif cut_name == "ptbin_12p0_13p0" and trigger_strategy == "HLT_all":
		ws.var('alpha').setVal(-2.5045e-01)
		ws.var('erfc_width').setVal(4.0000e-02)
		ws.var('erfc_x0').setVal(5.1448e+00)
		ws.var('hyp_lambda').setVal(-9.9087e-01)
		ws.var('hyp_mu').setVal(5.2786e+00)
		ws.var('hyp_sigma').setVal(2.4339e-02)
	elif cut_name == "ptbin_14p0_15p0" and trigger_strategy == "HLT_Mu9":
		ws.var('alpha').setVal(-1.0482e+00)
		ws.var('erfc_width').setVal(2.1694e-02)
		ws.var('erfc_x0').setVal(5.1384e+00)
		ws.var('hyp_lambda').setVal(-1.3219e+00)
		ws.var('hyp_mu').setVal(5.2786e+00)
		ws.var('hyp_sigma').setVal(2.6376e-02)
		ws.var('nbkgd').setVal(11e+03)
		ws.var('nbkgd_erfc').setVal(2.0624e+03)
		ws.var('nsignal').setVal(1.2e+04)
		ws.var("hyp_a").setConstant(False)
		ws.var("hyp_n").setConstant(False)
		ws.var("hyp_a2").setConstant(False)
		ws.var("hyp_n2").setConstant(False)
		for param_name in ["hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
			param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
			param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")]
			constraints[param_name] = ROOT.RooGaussian(
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				ws.var(param_name), 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])

	elif cut_name == "ptbin_18p0_23p0" and trigger_strategy == "HLT_Mu9":
		ws.var('alpha').setVal(-2.7233e+00)
		ws.var('erfc_width').setVal(1.3189e-02)
		ws.var('erfc_x0').setVal(5.1374e+00)
		ws.var('hyp_lambda').setVal(-1.2549e+00)
		ws.var('hyp_mu').setVal(5.2789e+00)
		ws.var('hyp_sigma').setVal(2.8928e-02)
		ws.var('nbkgd').setVal(2.0167e+03)
		ws.var('nbkgd_erfc').setVal(2.2775e+02)
		ws.var('nsignal').setVal(3.3314e+03)
	elif cut_name == "ptbin_23p0_26p0" and trigger_strategy == "HLT_Mu7":
		ws.var('alpha').setVal(-3.1736e+00)
		ws.var('erfc_width').setVal(2.1022e-02)
		ws.var('erfc_x0').setVal(5.1374e+00)
		ws.var('hyp_lambda').setVal(-1.4819e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.7475e-02)
	elif cut_name == "ptbin_26p0_29p0" and trigger_strategy == "HLT_Mu7":
		ws.var('alpha').setVal(-3.6885e+00)
		ws.var('erfc_width').setVal(2.1711e-02)
		ws.var('erfc_x0').setVal(5.1377e+00)
		ws.var('hyp_lambda').setVal(-1.4327e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.7220e-02)
	elif cut_name == "ptbin_26p0_29p0" and trigger_strategy == "HLT_Mu9":
		ws.var('alpha').setVal(-3.5400e+00)
		ws.var('erfc_width').setVal(2.2533e-02)
		ws.var('erfc_x0').setVal(5.1380e+00)
		ws.var('hyp_lambda').setVal(-1.4295e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.7239e-02)
	elif binned == False and cut_name == "ptbin_29p0_34p0":
		ws.var('alpha').setVal(-4.1479e+00)
		ws.var('erfc_width').setVal(1.9055e-02)
		ws.var('erfc_x0').setVal(5.1386e+00)
		ws.var('hyp_a').setVal(2.2064e+00)
		ws.var('hyp_a2').setVal(2.5859e+00)
		ws.var('hyp_lambda').setVal(-1.5183e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_n').setVal(2.3674e+00)
		ws.var('hyp_n2').setVal(3.1220e+00)
		ws.var('hyp_sigma').setVal(2.9200e-02 * 0.6)
	elif cut_name == "ptbin_29p0_34p0" and trigger_strategy == "HLT_Mu9":
		ws.var('alpha').setVal(-1)
		ws.var('erfc_width').setVal(1.9959e-02 * 0.8)
		ws.var('erfc_x0').setVal(5.1388e+00)
		ws.var('hyp_lambda').setVal(-1.4988e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.8921e-02 * 1.1)
		ws.var('nbkgd').setVal(4.2433e+04 * 1.1)
		ws.var('nbkgd_erfc').setVal(1.3199e+04)
		ws.var('nsignal').setVal(1.4591e+05 * 0.9)
		ws.var("hyp_a").setConstant(False)
		ws.var("hyp_n").setConstant(False)
		ws.var("hyp_a2").setConstant(False)
		ws.var("hyp_n2").setConstant(False)
		for param_name in ["hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
			param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
			param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")]
			constraints[param_name] = ROOT.RooGaussian(
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				ws.var(param_name), 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])
	elif cut_name == "ptbin_29p0_34p0" and trigger_strategy == "HLT_all":
		ws.var('alpha').setVal(-4.3807e+00)
		ws.var('erfc_width').setVal(1.9508e-02)
		ws.var('erfc_x0').setVal(5.1390e+00)
		ws.var('hyp_lambda').setVal(-1.5076e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.9081e-02)
	elif cut_name == "ptbin_34p0_45p0" and trigger_strategy == "HLT_Mu9":
		ws.var('alpha').setVal(-4.2233e+00)
		ws.var('erfc_width').setVal(2.3392e-02)
		ws.var('erfc_x0').setVal(5.1384e+00)
		ws.var('hyp_lambda').setVal(-1.4072e+00)
		ws.var('hyp_mu').setVal(5.2783e+00)
		ws.var('hyp_sigma').setVal(2.8652e-02)

	elif cut_name == "ybin_1p5_1p75" and trigger_strategy == "HLT_Mu9":
		ws.var("hyp_a").setConstant(False)
		ws.var("hyp_n").setConstant(False)
		ws.var("hyp_a2").setConstant(False)
		ws.var("hyp_n2").setConstant(False)
		for param_name in ["hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
			param_val = mc_fit_params[cut_name][param_name.replace("2", "")]
			param_err   = mc_fit_errors[cut_name][param_name.replace("2", "")]
			constraints[param_name] = ROOT.RooGaussian(
				"constr_{}".format(param_name), 
				"constr_{}".format(param_name), 
				ws.var(param_name), 
				ROOT.RooFit.RooConst(param_val),
				ROOT.RooFit.RooConst(param_err))
			print(constraints[param_name])
			ws.var(param_name).setVal(param_val)
			ws.var(param_name).setMin(max(0.01, param_val - 50.*param_err))
			ws.var(param_name).setMax(param_val + 50.*param_err)

	#if cut_name == "ptbin_29p0_34p0":
	#	ws.var('alpha').setVal(-3.6526e+00)
	#	ws.var('erfc_width').setVal(1.8052e-02)
	#	ws.var('erfc_x0').setVal(5.1364e+00)
	#	ws.var('frac_jpsipi').setVal(4.0000e-02)
	#	ws.var('hyp_a').setVal(2.0008e+00)
	#	ws.var('hyp_a2').setVal(3.1134e+00)
	#	ws.var('hyp_lambda').setVal(-1.4896e+00)
	#	ws.var('hyp_mu').setVal(5.2783e+00)
	#	ws.var('hyp_n').setVal(1.3298e+00)
	#	ws.var('hyp_n2').setVal(2.3424e+00)
	#	ws.var('hyp_sigma').setVal(2.8878e-02)
	#	ws.var('nbkgd').setVal(4.1284e+04)
	#	ws.var('nbkgd_erfc').setVal(1.8924e+04)
	#	ws.var('nsignal').setVal(1.8592e+05)


	if len(constraints):
		fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*(constraints.values()))))

	if correct_eff:
		if not binned:
			fit_args.append(ROOT.RooFit.SumW2Error(True)) # Unbinned + weighted needs special uncertainty treatment

	#ws.var("frac_jpsipi").setConstant(False)

	fit_result = model.fitTo(*fit_args)

	# Generate return info
	#fit_result = minimizer.save()

	# Add everything to the workspace
	getattr(ws, "import")(rdata)
	getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())
	getattr(ws, "import")(bkgd_jpsipi_model, ROOT.RooFit.RecycleConflictNodes())
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

	canvas = ROOT.TCanvas("c_datafit_{}".format(tag), "c_datafit_{}".format(tag), 800, 800)

	top = ROOT.TPad("top", "top", 0., 0.5, 1., 1.)
	top.SetBottomMargin(0.02)
	top.Draw()
	top.cd()

	rplot = xvar.frame(ROOT.RooFit.Bins(100))
	rdataset.plotOn(rplot, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"))
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_exp"), ROOT.RooFit.Components("bkgd_exp_model"), ROOT.RooFit.LineColor(ROOT.kRed+1))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_JpsiPi"), ROOT.RooFit.Components("bkgd_jpsipi_model"), ROOT.RooFit.LineColor(ROOT.kOrange-3), ROOT.RooFit.LineWidth(2), ROOT.RooFit.DrawOption("L"))
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
	l.AddEntry("signal", f"B^{{+}}#rightarrowJ/#psi K^{{#pm}} {ws.var('nsignal').getVal():.2f}", "lf")
	l.AddEntry("bkgd_exp", f"Comb. bkgd. {ws.var('nbkgd').getVal():.2f}", "l")
	l.AddEntry("bkgd_JpsiPi", f"B^{{+}}#rightarrowJ/#psi#pi^{{#pm}} {ws.function('n_jpsipi').getVal():.2f}", "l")
	l.AddEntry("bkgd_erfc", f"B^{{+}}#rightarrowJ/#psi+hadrons {ws.var('nbkgd_erfc').getVal():.2f}", "l")
	l.Draw()

	if text:
		textbox = ROOT.TLatex(0.15, 0.7, text)
		textbox.SetNDC()
		textbox.Draw()

	# Draw lines at tail starts
	xtail_left  = ws.var("hyp_mu").getVal() + ws.var("hyp_a").getVal() * ws.var("hyp_sigma").getVal()
	xtail_right = ws.var("hyp_mu").getVal() - ws.var("hyp_a2").getVal() * ws.var("hyp_sigma").getVal()
	line_left   = ROOT.TLine(xtail_left, rplot.GetMinimum(), xtail_left, rplot.GetMaximum())
	line_left.SetLineStyle(3)
	line_left.SetLineColor(ROOT.kGray)
	line_right  = ROOT.TLine(xtail_right, rplot.GetMinimum(), xtail_right, rplot.GetMaximum())
	line_right.SetLineStyle(3)
	line_right.SetLineColor(ROOT.kGray)
	line_left.Draw()
	line_right.Draw()


	canvas.cd()
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.5)
	bottom.SetTopMargin(0.02)
	bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	binning = ROOT.RooBinning(BU_FIT_NBINS, BU_FIT_WINDOW[0], BU_FIT_WINDOW[1])
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
	ndf = -9
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

	os.system(f"mkdir -pv {figure_dir}/{subfolder}")
	canvas.SaveAs("{}/{}/{}.png".format(figure_dir, subfolder, canvas.GetName()))
	canvas.SaveAs("{}/{}/{}.pdf".format(figure_dir, subfolder, canvas.GetName()))

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

def make_JpsiPi(ws, cut_name):
	# Make PDF
	bkgd_jpsipi_cb = ws.factory(f"RooCBShape::bkgd_jpsipi_cb(mass, mean_jpsipi[5.15, 5.7], sigma_jpsipi_cb[0.15, 0.001, 0.4], alpha_jpsipi_cb[-1., -100., 0.], n_jpsipi_cb[1, 0,100000])");
	bkgd_jpsipi_gauss = ws.factory(f"Gaussian::bkgd_jpsipi_gauss(mass, mean_jpsipi, sigma_jpsipi_gauss[0.1, 0.01, 0.25])")
	c_cb_jpsipi = ws.factory("c_cb_jpsipi[0.9, 0.75, 1.0]")
	bkgd_jpsipi_pdf = ROOT.RooAddPdf("bkgd_jpsipi_pdf", "bkgd_jpsipi_pdf", ROOT.RooArgList(bkgd_jpsipi_cb, bkgd_jpsipi_gauss), ROOT.RooArgList(c_cb_jpsipi))
	getattr(ws, "import")(bkgd_jpsipi_pdf, ROOT.RooFit.RecycleConflictNodes())
	#frac_jpsipi = ws.factory("frac_jpsipi[0.03846918489, 0.03846918489*0.9, 0.03846918489*1.1]")
	#getattr(ws, "import")(frac_jpsipi)
	#n_jpsipi = ROOT.RooFormulaVar("n_jpsipi", "n_jpsipi", "frac_jpsipi * nsignal", ROOT.RooArgList(frac_jpsipi, ws.var("nsignal")))
	n_jpsipi = ROOT.RooFormulaVar("n_jpsipi", "n_jpsipi", "0.03846918489 * nsignal", ROOT.RooArgList(ws.var("nsignal")))
	bkgd_jpsipi_model = ROOT.RooExtendPdf("bkgd_jpsipi_model", "bkgd_jpsipi_model", bkgd_jpsipi_pdf, n_jpsipi)
	getattr(ws, "import")(bkgd_jpsipi_model, ROOT.RooFit.RecycleConflictNodes())

	# Remap cut names for bad fits
	cut_name_remapped = cut_name
	if cut_name in ["ptbin_5p0_10p0", "ptbin_10p0_11p0", "ptbin_11p0_12p0", "ptbin_12p0_13p0", "ptbin_13p0_14p0", "ptbin_14p0_15p0"]:
		cut_name_remapped = "ptbin_10p0_15p0"
	elif cut_name in ["ybin_1p5_1p75", "ybin_1p75_2p0", "ybin_2p0_2p25"]:
		cut_name_remapped = "ybin_1p25_1p5"
	f_jpsipi = ROOT.TFile(f"Bu/fitws_mc_Bu2PiJpsi_{cut_name_remapped}.root")
	fit_result = f_jpsipi.Get("fitresult_bkgd_jpsipi_model_fitData")

	# Fix parameters
	params = fit_result.floatParsFinal()
	for i in range(params.getSize()):
		parname = params[i].GetName()
		if parname == "nbkgd_jpsipi":
			continue
		print(parname)
		val = params[i].getVal()
		error = params[i].getError()
		ws.var(parname).setVal(val)
		ws.var(parname).setConstant(True)

	rcache.extend([bkgd_jpsipi_model, bkgd_jpsipi_cb, bkgd_jpsipi_gauss, c_cb_jpsipi, bkgd_jpsipi_pdf, n_jpsipi])

	return bkgd_jpsipi_model



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Do Bu fits on data")
	parser.add_argument("--test", action="store_true", help="Do single test case (inclusive)")
	parser.add_argument("--all", action="store_true", help="Do all fit pT and y bins")
	parser.add_argument("--some", type=str, help="Run select cut strings")
	parser.add_argument("--fits", action="store_true", help="Do fits")
	parser.add_argument("--plots", action="store_true", help="Plot fits")
	parser.add_argument("--tables", action="store_true", help="Make yield tables")
	parser.add_argument("--binned", action="store_true", help="Do binned fit")
	parser.add_argument("--correct_eff", action="store_true", help="Apply efficiency correction before fitting")
	parser.add_argument("--fitparams", action="store_true", help="Print fit parameters")
	args = parser.parse_args()

	import glob
	data_files = glob.glob("/home/dryu/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = {"tag": ["inclusive"], "probe": ["inclusive"]}
	elif args.all:
		cuts = fit_cuts
	elif args.some:
		cuts_list = args.some.split(",")
		if not set(cuts_list).issubset(set(fit_cuts["tag"] + fit_cuts["probe"])):
			raise ValueError("Unrecognized cuts: {}".format(args.some))
		cuts = {"tag": [x for x in cuts_list if x in fit_cuts["tag"]], "probe": [x for x in cuts_list if x in fit_cuts["probe"]]}

	trigger_strategies_to_run = ["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"
	if args.fits:
		for side in ["tag", "tag_probebins", "probe"]: #, "tag_probebins", "probe"
		#for side in ["tagMaxPt", "tagMaxPt_probebins", "probeMaxPt"]: #, "tag_probebins", "probe"
			trigger_strategies = {
				"HLT_all": ["HLT_Mu7_IP4", "HLT_Mu9_IP5_only", "HLT_Mu9_IP6_only", "HLT_Mu12_IP6_only"],
				"HLT_Mu9": ["HLT_Mu9_IP5", "HLT_Mu9_IP6_only"],
				"HLT_Mu7": ["HLT_Mu7_IP4"],
				"HLT_Mu9_IP5": ["HLT_Mu9_IP5"],
				"HLT_Mu9_IP6": ["HLT_Mu9_IP6"],				
				#"HLT_Mu9_IP5&6": ["HLT_Mu9_IP6"],	
			}
			for trigger_strategy in trigger_strategies_to_run:
				print("\n*** Fitting {} / {} ***".format(side, trigger_strategy))
				chain = ROOT.TChain()
				for trigger in trigger_strategies[trigger_strategy]:
					if side == "tag_probebins" or side:
						tree_name = "Bcands_Bu_{}_{}".format("tag", trigger)
					elif side == "tagMaxPt_probebins":
						tree_name = "Bcands_Bu_{}_{}".format("tagMaxPt", trigger)
					else:
						tree_name = "Bcands_Bu_{}_{}".format(side, trigger)
					for data_file in data_files:
						chain.Add(f"{data_file}/{tree_name}")

				print("Total entries = {}".format(chain.GetEntries()))
				if side == "tag_probebins":
					cuts_thisside = cuts["probe"]
				elif side == "tagMaxPt":
					cuts_thisside = cuts["tag"]
				elif side == "probeMaxPt":
					cuts_thisside = cuts["probe"]
				elif side == "tagMaxPt_probebins":
					cuts_thisside = cuts["probe"]
				else:
					cuts_thisside = cuts[side]
				for cut_name in cuts_thisside:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					print("asdf {}".format(save_tag))
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bu_{}".format(save_tag))

					ws, fit_result = fit_data(chain, incut=cut_str, cut_name=cut_name, binned=args.binned, correct_eff=args.correct_eff, save_tag=save_tag, trigger_strategy=trigger_strategy)
					ws.Print()
					fit_result.Print()
					print("DEBUG : Saving to Bu/fitws_hyp_data_Bu_{}.root".format(save_tag))
					ws_file = ROOT.TFile("Bu/fitws_hyp_data_Bu_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

					# Clear cache
					del ws
					rcache = []

	if args.plots:
		for side in ["probe", "tag"]:
		#for side in ["probeMaxPt", "tagMaxPt"]:
			for trigger_strategy in trigger_strategies_to_run:
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bu/fitws_hyp_data_Bu_{}.root".format(save_tag), "READ")
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
					plot_fit(ws, fit_result, tag="Bu_hyp_{}".format(save_tag), subfolder=subfolder, text=fit_text[cut_name], binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in ["probe", "tag"]:
		#for side in ["probeMaxPt", "tagMaxPt"]:
			yields[side] = {}
			for trigger_strategy in trigger_strategies_to_run:
				yields[side][trigger_strategy] = {}
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bu/fitws_hyp_data_Bu_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		pprint(yields)
		#yields_file = "Bu/yields_hyp"
		yields_file = "Bu/yields_maxPt_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)

	if args.fitparams:
		for side in ["probe", "tag"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = "{}_{}_{}".format(side, cut_name, trigger_strategy)
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bu/fitws_hyp_data_Bu_{}.root".format(save_tag)
					if not os.path.isfile(fitresult_file):
						print("No fit result for {}, skipping".format(save_tag))
						continue
					ws_file = ROOT.TFile(fitresult_file, "READ")
					#ws_file.ls()
					fit_result = ws_file.Get(f"fitresult_model_fitData{'Binned' if args.binned else ''}")
					print("\n*** Printing fit results for {} ***".format(save_tag))
					fit_result.Print()

"""
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
	data_files = glob.glob("/home/dryu/BFrag/data/histograms/Run2018*.root")

	if args.test: 
		cuts = ["inclusive"]
	elif args.all:
		cuts = sorted(fit_cuts.keys())

	if args.fits:
		for side in ["probe", "tag"]:
			for trigger in ["HLT_Mu7", "HLT_Mu9_only", "HLT_Mu12_only"]:
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
					ws_file = ROOT.TFile("fitws_hyp_data_Bu_{}_{}_{}.root".format(side, cut_name), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["probe", "tag"]:
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_hyp_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				plot_fit(ws, tag="Bu_{}_{}".format(side, cut_name), text=fit_text[cut_name])

	if args.tables:
		yields = {}
		for side in ["probe", "tag"]:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_hyp_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
		pprint(yields)
"""