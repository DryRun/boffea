'''
Fit MC mass distributions
'''
import os
from pprint import pprint
import ROOT
from brazil.aguapreta import *
ROOT.gROOT.SetBatch(True)
import pickle

from brazil.tdrstyle import SetTDRStyle
tdr_style = SetTDRStyle()

from brazil.cmslabel import CMSLabel, LuminosityLabel

#ROOT.gROOT.ProcessLine(open('include/TripleGaussianPdf2.cc').read())
#ROOT.gROOT.ProcessLine(".x include/TripleGaussianPdf2.cc+")
#ROOT.gSystem.Load("include/TripleGaussianPdf2.so")

#ROOT.gROOT.ProcessLine(open('include/TripleGaussian.cc').read())
ROOT.gSystem.Load("include/TripleGaussianPdf_cc.so")
from ROOT import TripleGaussianPdf
ROOT.gROOT.ProcessLine(open('include/MyErfc.cc').read())
#from ROOT import TripleGaussian
#from ROOT import TripleGaussianPdf2
from ROOT import MyErfc

use_mc_constraints = True

rcache = [] # Prevent RooFit objects from disappearing

figure_dir = "/home/dyu7/BFrag/data/fits/data"

import sys
sys.path.append(".")
from fit_settings import fit_cuts, cut_strings, fit_text, \
	BU_FIT_WINDOW, BD_FIT_WINDOW, BS_FIT_WINDOW, \
	BU_FIT_NBINS, BD_FIT_NBINS, BS_FIT_NBINS, \
	get_Bcands_name_data, get_MC_fit_params, get_MC_fit_errs, \
	MakeHypatia, MakeJohnson, MakeDoubleGaussian, MakeTripleGaussian, MakeTripleGaussianConstrained

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
	c.SaveAs("/home/dyu7/BFrag/data/fits/data/{}.pdf".format(c.GetName()))


def fit_data(tree, mass_range=BU_FIT_WINDOW, incut="1", cut_name="inclusive", binned=False, correct_eff=False, save_tag=None, trigger_strategy=None, side=None, selection="nominal", fitfunc=""):
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
	print("ndata={}".format(ndata))

	# Optional: bin data
	if binned:
		mass.setBins(BS_FIT_NBINS)
		rdatahist = ROOT.RooDataHist("fitDataBinned", "fitDataBinned", ROOT.RooArgSet(mass), rdataset)
		rdata = rdatahist
	else:
		rdata = rdataset

	# Signal: hypatia
	if fitfunc == "hypatia":
		signal_pdf = MakeHypatia(ws, mass_range, rcache=rcache)
	elif fitfunc == "johnson":
		signal_pdf = MakeJohnson(ws, mass_range, rcache=rcache)
	elif fitfunc == "2gauss":
		signal_pdf = MakeDoubleGaussian(ws, mass_range, rcache=rcache)
	elif fitfunc == "3gauss":
		signal_pdf = MakeTripleGaussianConstrained(ws, mass_range, rcache=rcache)
	getattr(ws, "import")(signal_pdf, ROOT.RooFit.RecycleConflictNodes())
	nsignal = ws.factory(f"nsignal[{ndata*0.5}, 0.0, {ndata*2.0}]")
	signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_pdf, nsignal)

	# Background model: exponential + (Bu > Jpsi pi gaussian) + (partial reco ERF)
	if bkgdfunc == "exp":
		bkgd_comb = ws.factory(f"Exponential::bkgd_comb(mass, alpha[-3.75, -20., -0.1])")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)
	elif bkgdfunc == "poly":
		bkgd_comb = ws.factory(f"Chebychev::bkgd_comb(mass, {{p1[0., -10., 10.]}})")
		nbkgd_comb = ws.factory(f"nbkgd[{ndata*0.1}, 0.0, {ndata*2.0}]")
		bkgd_comb_model = ROOT.RooExtendPdf("bkgd_comb_model", "bkgd_comb_model", bkgd_comb, nbkgd_comb)
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
	model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_comb_model, bkgd_jpsipi_model, bkgd_erfc_model))

	# Perform fit
	fit_args = [rdata, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
	
	constraints = {}
	if selection == "nominal":
		mcside = f"{side}match"
	elif selection == "HiTrkPt":
		mcside = f"{side}HiTrkPtmatch"
	elif "MuonPt" in selection:
		mcside = f"{side}{selection}match"
	elif selection == "MediumMuon":
		mcside = f"{side}{selection}match"
	else:
		raise ValueError("asdfg Don't know what to do with selection {}".format(selection))

	# For tagx, just use tag shapes
	if "tagx" in mcside:
		mcside = mcside.replace("tagx", "tag")

	# Same for tag_probebins
	if "tag_probebins" in side:
		mcside = mcside.replace("_probebins", "")

	if use_mc_constraints:
		# Load constraints from MC fit
		with open(get_MC_fit_params("Bu", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			mc_fit_params = pickle.load(f)
		with open(get_MC_fit_errs("Bu", selection=selection, fitfunc=fitfunc, frozen=True), "rb") as f:
			mc_fit_errors = pickle.load(f)

		#print(f"{cut_name} : adding constraints")
		#print("Constraint central values:")
		#pprint(mc_fit_params[cut_name])
		#print("Constraint widths:")
		#pprint(mc_fit_errors[cut_name])
		
		if fitfunc == "hypatia":
			pprint(mc_fit_params)
			pprint(mc_fit_params.keys())
			for param_name in ["hyp_lambda", "hyp_sigma", "hyp_mu", "hyp_a", "hyp_n", "hyp_a2", "hyp_n2"]:
				var = ws.var(param_name)
				print("Adding constraint for {}".format(param_name))
				var.setVal(mc_fit_params[mcside][cut_name][param_name.replace("2", "")])

				err_multiplier = 1.0
				param_val = mc_fit_params[mcside][cut_name][param_name.replace("2", "")]
				param_err   = mc_fit_errors[mcside][cut_name][param_name.replace("2", "")] * err_multiplier
				if param_err < 1.e-6:
					print(cut_name)
					print("Binned={}".format(binned))
					print("mcside={}".format(mcside))
					print(save_tag)
					print(trigger_strategy)
					print("WARNING : Param {} has small error {} +/- {}".format(param_name, param_val, param_err))
					raise ValueError("Quitting")
					sys.exit(1)

				# Loose rectangular constraint on mean (via variable range)
				if "hyp_mu" in param_name:
					ws.var(param_name).setMin(mc_fit_params[mcside][cut_name][param_name] - 0.1)
					ws.var(param_name).setMax(mc_fit_params[mcside][cut_name][param_name] + 0.1)
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
			# End loop over parameters

		elif fitfunc == "johnson":
			for param_name in ["j_mu", "j_lambda", "j_delta", "j_gamma"]:
				err_multiplier = 1.0
				param_val = mc_fit_params[mcside][cut_name][param_name]
				param_err   = mc_fit_errors[mcside][cut_name][param_name] * err_multiplier

				if param_err < 1.e-5 and param_name != "j_mu":
					print("WARNING : Param {} has small error {}".format(param_name, param_err))
					raise ValueError("Quitting")
					sys.exit(1)

				var = ws.var(param_name)
				var.setVal(param_val)

				if param_name == "j_mu":
					# Loose rectangular constraint on mean (via variable range)
					ws.var(param_name).setMin(param_val - 0.1)
					ws.var(param_name).setMax(param_val + 0.1)
					continue
				elif param_name == "j_lambda":
					# For core width parameters, set very loose constraint
					param_err = max(abs(param_val / 2.), param_err * 10.)
				elif param_name == "j_delta" or param_name == "j_gamma":
					# For gamma and delta, constrain to MC error
					pass

				# Adjust variable value and range to match constraints
				var.setVal(param_val)
				param_min = max(var.getMin(), param_val - 10. * param_err)
				param_max = min(var.getMax(), param_val + 10. * param_err)
				ws.var(param_name).setMin(param_min)
				ws.var(param_name).setMax(param_max)

				constraints[param_name] = ROOT.RooGaussian(
					"constr_{}".format(param_name), 
					"constr_{}".format(param_name), 
					var, 
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err))
				print(constraints[param_name])
			# End loop over parameter names
		elif fitfunc == "2gauss":
			raise NotImplementedError("2gauss not implemented!")
		elif fitfunc == "3gauss":
			for param_name in mc_fit_params[mcside][cut_name].keys():
				if "nsignal" in param_name:
					continue

				err_multiplier = 1.0
				param_val = mc_fit_params[mcside][cut_name][param_name]
				param_err = mc_fit_errors[mcside][cut_name][param_name] * err_multiplier
				if param_err < 1.e-6 and not "mean" in param_name:
					print("WARNING : Param {} has small error {} ; cut_name = {}".format(param_name, param_err, cut_name))
					raise ValueError("Quitting")
					sys.exit(1)

				# Fix sigma_3, aa, bb
				constant_params = ["tg_aa", "tg_bb", "tg_sigma1", "tg_sigma2", "tg_sigma3"]
				isConstant = False
				for name2 in constant_params:
					if name2 in param_name:
						print(param_name)
						ws.Print()
						ws.var(param_name).setVal(param_val)
						ws.var(param_name).setConstant(True)
						isConstant = True
				if isConstant:
					continue

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

		# End if hypatia or johnson
	# End if use MC constraints

	# Probe: use tag fits to constrain erfc parameters
	if "probe" in save_tag and not "probebins" in save_tag:
		save_tag_tag = save_tag.replace("probeMaxPt", "probe").replace("probe", "tag_probebins")
		tag_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag_tag), "READ")
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
	if fitfunc == "hypatia":
		if cut_name == "ptbin_8p0_13p0":
			ws.var('alpha').setVal(-3.1524e-01)
			ws.var('erfc_width').setVal(3.5406e-02)
			ws.var('erfc_x0').setVal(5.1271e+00)
			ws.var('hyp_lambda').setVal(-3.4036e-01)
			ws.var('hyp_mu').setVal(5.2782e+00)
			ws.var('hyp_sigma').setVal(1.8653e-02)
			ws.var('hyp_a').setConstant(False)
			ws.var('hyp_a').setVal(1.0)
			ws.var('hyp_a').setConstant(True)
			ws.var('hyp_a2').setConstant(False)
			ws.var('hyp_a2').setVal(1.0)
			ws.var('hyp_a2').setConstant(True)

		elif cut_name == "ptbin_13p0_18p0" and (trigger_strategy == "HLT_Mu9" or trigger_strategy == "HLT_all"):
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
				param_val = mc_fit_params[mcside][cut_name][param_name.replace("2", "")]
				param_err   = mc_fit_errors[mcside][cut_name][param_name.replace("2", "")]
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
		elif cut_name == "ptbin_14p0_15p0":
			ws.var('alpha').setVal(-1.4922e+00)
			ws.var('erfc_width').setVal(4.0000e-02)
			ws.var('erfc_x0').setVal(5.1355e+00)
			ws.var('hyp_lambda').setVal(-1.3564e+00)
			ws.var('hyp_mu').setVal(5.2785e+00)
			ws.var('hyp_sigma').setVal(2.6603e-02)
			ws.var('nbkgd').setVal(1.1127e+04)
			ws.var('nbkgd_erfc').setVal(3.0940e+03)
			'''
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
			'''
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
				param_val = mc_fit_params[mcside][cut_name][param_name.replace("2", "")]
				param_err   = mc_fit_errors[mcside][cut_name][param_name.replace("2", "")]
				constraints[param_name] = ROOT.RooGaussian(
					"constr_{}".format(param_name), 
					"constr_{}".format(param_name), 
					ws.var(param_name), 
					ROOT.RooFit.RooConst(param_val),
					ROOT.RooFit.RooConst(param_err))
				print(constraints[param_name])
		#elif cut_name == "ptbin_29p0_34p0" and trigger_strategy == "HLT_all":
		#	ws.var('alpha').setVal(-4.3807e+00)
		#	ws.var('erfc_width').setVal(1.9508e-02)
		#	ws.var('erfc_x0').setVal(5.1390e+00)
		#	ws.var('hyp_lambda').setVal(-1.5076e+00)
		#	ws.var('hyp_mu').setVal(5.2783e+00)
		#	ws.var('hyp_sigma').setVal(2.9081e-02)
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
				param_val = mc_fit_params[mcside][cut_name][param_name.replace("2", "")]
				param_err   = mc_fit_errors[mcside][cut_name][param_name.replace("2", "")]
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
		elif cut_name == "ybin_0p0_0p25":
			ws.var('alpha').setVal(-4.3807e+00)
			ws.var('erfc_width').setVal(1.9508e-02)
			ws.var('erfc_x0').setVal(5.1390e+00)
			ws.var('hyp_lambda').setVal(-1.5076e+00)
			ws.var('hyp_mu').setVal(5.2783e+00)
			ws.var('hyp_sigma').setVal(2.9081e-02)

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

def plot_fit(ws, fit_result, tag="", subfolder="", text=None, binned=False, correct_eff=False, fitfunc=""):
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
	model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), 
					ROOT.RooFit.LineColor(ROOT.kRed), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.FillColor(ROOT.kRed), 
					ROOT.RooFit.FillStyle(3007), 
					ROOT.RooFit.DrawOption("LF"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_comb"), ROOT.RooFit.Components("bkgd_comb_model"), 
					ROOT.RooFit.LineColor(ROOT.kBlue+2),
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.FillStyle(0))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_JpsiPi"), ROOT.RooFit.Components("bkgd_jpsipi_model"), 
					ROOT.RooFit.LineColor(ROOT.kGreen+2), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.FillStyle(0),
					ROOT.RooFit.DrawOption("L"))
	model.plotOn(rplot, ROOT.RooFit.Name("bkgd_erfc"), ROOT.RooFit.Components("bkgd_erfc_model"), 
					ROOT.RooFit.LineColor(ROOT.kMagenta+1), 
					ROOT.RooFit.FillStyle(0),
					ROOT.RooFit.LineWidth(1))
	model.plotOn(rplot, ROOT.RooFit.Name("fit"), 
					ROOT.RooFit.LineColor(ROOT.kBlue), 
					ROOT.RooFit.LineWidth(1), 
					ROOT.RooFit.LineStyle(1), 
					ROOT.RooFit.FillStyle(0))
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

	#l = ROOT.TLegend(0.7, 0.45, 0.88, 0.88)
	legend = ROOT.TLegend(0.61, 0.49, 0.95, 0.9)
	legend.SetFillColor(0)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextFont(43)
	legend.SetTextSize(40)
	legend.AddEntry("data", "Data", "lp")
	legend.AddEntry("fit", "Signal + bkgd. fit", "l")
	#legend.AddEntry("signal", f"B^{{+}}#rightarrowJ/#psi K^{{#pm}} {ws.var('nsignal').getVal():.2f}", "lf")
	#legend.AddEntry("bkgd_comb", f"Comb. bkgd. {ws.var('nbkgd').getVal():.2f}", "l")
	#legend.AddEntry("bkgd_JpsiPi", f"B^{{+}}#rightarrowJ/#psi#pi^{{#pm}} {ws.function('n_jpsipi').getVal():.2f}", "l")
	#legend.AddEntry("bkgd_erfc", f"B^{{+}}#rightarrowJ/#psi+hadrons {ws.var('nbkgd_erfc').getVal():.2f}", "l")
	legend.AddEntry("signal", f"B^{{+}}#rightarrowJ/#psi K^{{#pm}}", "lf")
	legend.AddEntry("bkgd_comb", f"Comb. bkgd.", "l")
	legend.AddEntry("bkgd_JpsiPi", f"B^{{+}}#rightarrowJ/#psi#pi^{{#pm}}", "l")
	legend.AddEntry("bkgd_erfc", f"B^{{+}}#rightarrowJ/#psi+hadrons", "l")
	legend.Draw()

	# Draw lines at tail starts
	if fitfunc == "hypatia":
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
	bottom = ROOT.TPad("bottom", "bottom", 0., 0., 1., 0.3)
	bottom.SetTopMargin(0.03)
	#bottom.SetBottomMargin(0.25)
	bottom.Draw()
	bottom.cd()

	#binning = ROOT.RooBinning(BU_FIT_NBINS, BU_FIT_WINDOW[0], BU_FIT_WINDOW[1])
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
		if param.GetName() in ["j_delta", "j_gamma"]:
			print(f"Skipping {param.GetName()} because it is constrained to MC")
			continue
		print(f"DEBUGG : Counting param {param.GetName()} as -1df")
		nparams += 1

	print(f"DEBUGG : nparams = {nparams}")
	#ndf = -9
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
	pull_hist.GetXaxis().SetTitle("M_{J/#Psi K^{#pm}} [GeV]")
	pull_hist.GetYaxis().SetTitle("Pull w.r.t. fit [#sigma]")
	pull_hist.GetXaxis().SetLabelSize(40)
	pull_hist.GetXaxis().SetTitleSize(40)
	pull_hist.GetYaxis().SetLabelSize(40)
	pull_hist.GetYaxis().SetTitleSize(40)
	pull_hist.SetMarkerStyle(20)
	pull_hist.SetMarkerSize(1.0)
	#pull_hist.GetXaxis().SetTitleSize(0.06)
	#pull_hist.GetXaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleSize(0.06)
	#pull_hist.GetYaxis().SetLabelSize(0.06)
	#pull_hist.GetYaxis().SetTitleOffset(0.6)
	pull_hist.SetMinimum(-4.)
	pull_hist.SetMaximum(4.)
	pull_hist.Draw("p")

	zero = ROOT.TLine(BU_FIT_WINDOW[0], 0., BU_FIT_WINDOW[1], 0.)
	zero.SetLineColor(ROOT.kBlack)
	zero.SetLineStyle(1)
	zero.SetLineWidth(2)
	zero.Draw()

	# Labels and text
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


	#chi2text = ROOT.TLatex(0.15, 0.6, f"#chi^{{2}}/NDF={round(chi2/ndf, 2)}")
	#chi2text.SetNDC()
	#chi2text = ROOT.TPaveText(0.18, 0.55, 0.35, 0.65, "RNDC")
	#chi2text.AddText(f"#chi^{{2}}/ndof={round(chi2/ndf, 3):.3f}")
	#chi2text.SetFillStyle(0)
	#chi2text.SetLineWidth(0)
	#chi2text.SetTextSize(0.04)
	#chi2text.SetTextFont(42)
	#chi2text.SetTextAlign(10)
	#chi2text.Draw()
	
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
	parser.add_argument("--selection", type=str, default="nominal", help="Selection name (nominal, HiTrkPt, ...)")
	parser.add_argument("--fitfunc", type=str, default="johnson", help="Fit function name (hypatia, johnson, poly)")
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
		cuts = {"tag": [x for x in cuts_list if x in fit_cuts["tag"]], "probe": [x for x in cuts_list if x in fit_cuts["probe"]]}
	cuts["tagx"] = cuts["tag"]

	trigger_strategies_to_run = args.trigger_strategies.split(",") #["HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"] # "HLT_all", "HLT_Mu7", "HLT_Mu9", "HLT_Mu9_IP5", "HLT_Mu9_IP6"
	sides_to_run = args.sides.split(",")
	for i in range(len(sides_to_run)):
		if sides_to_run[i] == "probe" and not "tag_probebins" in sides_to_run:
			sides_to_run.insert(i, "tag_probebins")
			break 

	if args.fits:
		for side in sides_to_run: # ["tag", "tag_probebins", "probe"]: #, "tag_probebins", "probe"
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
					tree_name = get_Bcands_name_data(btype="Bu", trigger=trigger, side=side, selection=args.selection)
					
					for data_file in data_files:
						print(f"Adding {data_file}/{tree_name}")
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
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					print("asdf {}".format(save_tag))
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					cut_str = cut_strings[cut_name]
					plot_data(chain, cut=cut_str, tag="Bu_{}".format(save_tag))

					ws, fit_result = fit_data(chain, 
										incut=cut_str, 
										cut_name=cut_name, 
										binned=args.binned, 
										correct_eff=args.correct_eff, 
										save_tag=save_tag, 
										trigger_strategy=trigger_strategy, 
										side=side,
										selection=args.selection, 
										fitfunc=args.fitfunc)
					ws.Print()
					fit_result.Print()
					print("DEBUG : Saving to Bu/fitws_data_Bu_{}.root".format(save_tag))
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

					# Clear cache
					del ws
					rcache = []

	if args.plots:
		for side in sides_to_run:
			if side == "tag_probebins":
				continue
		#for side in ["probeMaxPt", "tagMaxPt"]:
			for trigger_strategy in trigger_strategies_to_run:
				for cut_name in cuts[side.replace("MaxPt", "")]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"				
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "READ")
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
					plot_fit(ws, fit_result, tag="Bu_{}".format(save_tag), subfolder=subfolder, text=plot_text, binned=args.binned, correct_eff=args.correct_eff)

	if args.tables and args.all:
		yields = {}
		for side in ["probe", "tag", "tagx"]:
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
					ws_file = ROOT.TFile("Bu/fitws_data_Bu_{}.root".format(save_tag), "READ")
					#ws_file.ls()
					ws = ws_file.Get("ws")
					yields[side][trigger_strategy][cut_name] = extract_yields(ws)
		pprint(yields)
		yields_file = f"Bu/yields_{args.fitfunc}_{args.selection}"
		#yields_file = "Bu/yields_maxPt_hyp"
		if args.binned:
			yields_file += "_binned"
		if args.correct_eff:
			yields_file += "_correcteff"
		yields_file += ".pkl"
		print("Saving yields to {}".format(yields_file))
		with open(yields_file, "wb") as f_yields:
			pickle.dump(yields, f_yields)

	if args.fitparams:
		for side in ["probe", "tag", "tagx"]:
			for trigger_strategy in ["HLT_all", "HLT_Mu7", "HLT_Mu9"]:
				for cut_name in cuts[side]:
					save_tag = f"{side}_{cut_name}_{trigger_strategy}_{args.fitfunc}_{args.selection}"
					if args.binned:
						save_tag += "_binned"
					if args.correct_eff:
						save_tag += "_correcteff"
					fitresult_file = "Bu/fitws_data_Bu_{}.root".format(save_tag)
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
	data_files = glob.glob("/home/dyu7/BFrag/data/histograms/Run2018*.root")

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
					ws_file = ROOT.TFile("fitws_data_Bu_{}_{}_{}.root".format(side, cut_name), "RECREATE")
					ws.Write()
					fit_result.Write()
					ws_file.Close()

	if args.plots:
		for side in ["probe", "tag"]:
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws_file.ls()
				ws = ws_file.Get("ws")
				plot_fit(ws, tag="Bu_{}_{}".format(side, cut_name), text=fit_text[cut_name])

	if args.tables:
		yields = {}
		for side in ["probe", "tag"]:
			yields[side] = {}
			for cut_name in cuts:
				ws_file = ROOT.TFile("fitws_data_Bu_{}_{}.root".format(side, cut_name), "READ")
				ws = ws_file.Get("ws")
				yields[side][cut_name] = extract_yields(ws)
		pprint(yields)
"""