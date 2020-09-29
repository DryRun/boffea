import ROOT
import os
import sys
import glob
import math
from brazil.aguapreta import *

chains = {}
chains["data"] = ROOT.TChain("Bcands_Bs_opt")
for f in glob.glob("/home/dryu/BFrag/data/histograms/Run2018*root"):
    chains["data"].Add(f)
chains["mc"] = ROOT.TChain("Bcands_Bs_opt_Bs2PhiJpsi2KKMuMu_probefilter")
chains["mc"].Add("/home/dryu/BFrag/boffea/barista/fitting/optimization_Bs_mc.root")

chains["data"].Print()
chains["mc"].Print()

cuts = {
	"preselection": [
		"l1_pt > 1.0",
		"l2_pt > 1.0",
		"pt < 12.0",
		"dm_phi < 0.03"
	],
	"optimized": [
		"sv_prob > 0.003", 
		"cos2D > 0.0988", 
		"l_xy_sig > 2.0", 
		"dm_phi < 0.016", 
		"k1_pt > 0.5", 
		"k2_pt > 0.5"
	],
	"lep1p0": [
		"l1_pt > 1.0", 
		"l2_pt > 1.0"
	],
	"lep1p5": [
		"l1_pt > 1.5", 
		"l2_pt > 1.5"
	],
	"lepvary": [
		"l1_pt > 1.5", 
		"(l2_pt > 1.5) || (l2_pt > 1.0 && TMath::Abs(eta) > 1.2)"
	]
}
cutstrings = {}
cutstrings["preselection"] = " && ".join(["({})".format(x) for x in cuts["preselection"]])
cutstrings["optimized"] = " && ".join(["({})".format(x) for x in cuts["preselection"] + cuts["optimized"]])
cutstrings["lep1p0"] = " && ".join(["({})".format(x) for x in cuts["preselection"] + cuts["optimized"] + cuts["lep1p0"]])
cutstrings["lep1p5"] = " && ".join(["({})".format(x) for x in cuts["preselection"] + cuts["optimized"] + cuts["lep1p5"]])
cutstrings["lepvary"] = " && ".join(["({})".format(x) for x in cuts["preselection"] + cuts["optimized"] + cuts["lepvary"]])

n = {}

for selname in ["preselection", "optimized", "lep1p0", "lep1p5", "lepvary"]:
	n[selname] = {}
	for what in ["data", "mc"]:
		extra_cuts = ""
		if what == "data":
			extra_cuts = f" && (TMath::Abs(mass - {BS_MASS}) > 0.1)"
		n[selname][what] = chains[what].GetEntries(cutstrings[selname] + extra_cuts)
		print(f"{selname} / {what} = {n[selname][what]}")
	eff_s = n[selname]['mc'] / max(n['preselection']['mc'], 1.e-20)
	eff_b = n[selname]['data'] / max(n['preselection']['data'], 1.e-20)
	print(f"{selname} eff_s = {n[selname]['mc']} / {n['preselection']['mc']} = {eff_s}")
	print(f"{selname} eff_b = {n[selname]['data']} / {n['preselection']['data']} = {eff_b}")
	print(f"{selname} eff_s / sqrt(eff_b) = {eff_s / math.sqrt(eff_b)}")
	print(f"{selname} N_s / sqrt(N_s+N_b) = {10.*eff_s / math.sqrt(10.*eff_s + 50.*eff_b)}")

