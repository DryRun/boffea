#! /usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import os
import sys
import math
import concurrent.futures
import gzip
import pickle
import json
import time
import numexpr
import array

import uproot
import numpy as np
from coffea import hist, lookup_tools
import awkward
import copy

from constants import *

np.set_printoptions(threshold=np.inf)

def where(predicate, iftrue, iffalse):
    predicate = predicate.astype(np.bool)   # just to make sure they're 0/1
    return predicate*iftrue + (1 - predicate)*iffalse

trigger = "HLT_Mu7_IP4"

# Histograms
dataset = hist.Cat("dataset", "Primary dataset")
histograms = {}
histograms["nMuon"]          = hist.Hist("Events", dataset, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
histograms["nMuon_isTrig"]   = hist.Hist("Events", dataset, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
histograms["Muon_pt"]        = hist.Hist("Events", dataset, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
histograms["Muon_pt_isTrig"] = hist.Hist("Events", dataset, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

histograms["BuToKMuMu_fit_pt"]    = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
histograms["BuToKMuMu_fit_eta"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
histograms["BuToKMuMu_fit_phi"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
histograms["BuToKMuMu_fit_mass"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
histograms["BToKMuMu_chi2"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
histograms["BuToKMuMu_fit_cos2D"] = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
histograms["BuToKMuMu_l_xy"]      = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
histograms["BuToKMuMu_l_xy_sig"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

histograms["BuToKMuMu_tag_fit_pt"]    = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
histograms["BuToKMuMu_tag_fit_eta"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
histograms["BuToKMuMu_tag_fit_phi"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
histograms["BuToKMuMu_tag_fit_mass"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
histograms["BuToKMuMu_tag_fit_chi2"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
histograms["BuToKMuMu_tag_fit_cos2D"] = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
histograms["BuToKMuMu_tag_l_xy"]      = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
histograms["BuToKMuMu_tag_l_xy_sig"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

histograms["BuToKMuMu_probe_fit_pt"]    = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
histograms["BuToKMuMu_probe_fit_eta"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
histograms["BuToKMuMu_probe_fit_phi"]   = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
histograms["BuToKMuMu_probe_fit_mass"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
histograms["BuToKMuMu_probe_fit_chi2"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
histograms["BuToKMuMu_probe_fit_cos2D"] = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
histograms["BuToKMuMu_probe_l_xy"]      = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
histograms["BuToKMuMu_probe_l_xy_sig"]  = hist.Hist("Events", dataset, hist.Bin("BToKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

histograms["TruthBuToKMuMu_pt"]    = hist.Hist("Events", dataset, hist.Bin("TruthBToKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
histograms["TruthBuToKMuMu_eta"]   = hist.Hist("Events", dataset, hist.Bin("TruthBToKMuMu_eta", r"$\eta$", 50, -5.0, 5.0))
histograms["TruthBuToKMuMu_phi"]   = hist.Hist("Events", dataset, hist.Bin("TruthBToKMuMu_phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
histograms["TruthBuToKMuMu_mass"]  = hist.Hist("Events", dataset, hist.Bin("TruthBToKMuMu_mass", r"$m$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))

# One entry per truth B
# - If truth B is not matched to reco, or if reco fails selection, fill (-1, truthpt)
histograms["BuToKMuMu_truthpt_recopt_tag"] = hist.Hist("Events", dataset, 
                                          hist.Bin("TruthBToKMuMu_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 500, 0., 100.),
                                          hist.Bin("TruthBToKMuMu_truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 500, 0., 100.))
histograms["BuToKMuMu_truthpt_recopt_probe"] = hist.Hist("Events", dataset, 
                                          hist.Bin("TruthBToKMuMu_reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 500, 0., 100.),
                                          hist.Bin("TruthBToKMuMu_truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 500, 0., 100.))

branches = ["BToKMuMu_*", "Muon_*", "ProbeTracks_*", "TrigObj_*", "TriggerMuon_*", "GenPart_*", 
            "nBToKMuMu",  "nMuon",  "nProbeTracks",  "nTrigObj",  "nTriggerMuon",  "nGenPart",
            "HLT_Mu7_IP4", "HLT_Mu8_IP6", "HLT_Mu8_IP5", "HLT_Mu8_IP3", "HLT_Mu8p5_IP3p5", "HLT_Mu9_IP6", "HLT_Mu9_IP5", "HLT_Mu9_IP4", "HLT_Mu10p5_IP3p5", "HLT_Mu12_IP6", 
            "L1_SingleMu7er1p5", "L1_SingleMu8er1p5", "L1_SingleMu9er1p5", "L1_SingleMu10er1p5", "L1_SingleMu12er1p5", "L1_SingleMu22",             
            "event"]

def process_file(dataset, file):
    print(f"process_file({dataset}, {file})")
    tree = uproot.open(file)["Events"]
    arrays = tree.arrays(branches, namedecode='ascii')

    # Tag/probe selection
    arrays["BToKMuMu_Muon1IsTrig"] = arrays["Muon_isTriggering"][arrays["BToKMuMu_l1Idx"]] # shape=BToKMuMu
    arrays["BToKMuMu_Muon2IsTrig"] = arrays["Muon_isTriggering"][arrays["BToKMuMu_l2Idx"]] # shape=BToKMuMu
    arrays["BToKMuMu_MuonIsTrigCount"] = arrays["BToKMuMu_Muon1IsTrig"] + arrays["BToKMuMu_Muon2IsTrig"]
    arrays["Event_NTriggeringMuons"] = arrays["Muon_isTriggering"].sum()
    arrays["BToKMuMu_TagCount"] = arrays["BToKMuMu_MuonIsTrigCount"].ones_like() * arrays["Event_NTriggeringMuons"] - arrays["BToKMuMu_MuonIsTrigCount"]

    # General selection
    sv_selection = (arrays['BToKMuMu_fit_pt'] > 3.0) \
                    & (abs(arrays['BToKMuMu_l_xy'] / arrays['BToKMuMu_l_xy_unc']) > 3.0 ) \
                    & (arrays['BToKMuMu_svprob'] > 0.1) \
                    & (arrays['BToKMuMu_fit_cos2D'] > 0.9)
    l1_selection = (arrays['BToKMuMu_fit_l1_pt'] > 1.5) \
                    & (abs(arrays['BToKMuMu_fit_l1_eta']) < 2.4)
    l2_selection = (arrays['BToKMuMu_fit_l2_pt'] > 1.5) \
                    & (abs(arrays['BToKMuMu_fit_l2_eta']) < 2.4)
    k_selection = (arrays['BToKMuMu_fit_k_pt'] > 0.5) \
                    & (abs(arrays['BToKMuMu_fit_k_eta']) < 2.5)
    arrays[f"BToKMuMu_{trigger}"] = arrays["BToKMuMu_fit_pt"].ones_like() * arrays[trigger] == 1
    reco_selection = arrays[f"BToKMuMu_{trigger}"] & sv_selection & l1_selection & l2_selection & k_selection

    tag_selection = (arrays["BToKMuMu_Muon1IsTrig"] == 1) | (arrays["BToKMuMu_Muon2IsTrig"] == 1) & reco_selection
    probe_selection = (arrays["BToKMuMu_TagCount"] >= 1) & reco_selection

    # Derived variables
    arrays["BToKMuMu_l_xy_sig"] = where(arrays["BToKMuMu_l_xy_unc"] > 0, arrays["BToKMuMu_l_xy"] / arrays["BToKMuMu_l_xy_unc"], -1.e20)

    # Make a copy of the template histograms
    rhistograms = {}
    for key in histograms.keys():
        rhistograms[key] = histograms[key].copy(content=False)

    # Fill reco histograms
    rhistograms["nMuon"].fill(dataset=dataset, nMuon=arrays["nMuon"])
    rhistograms["nMuon_isTrig"].fill(dataset=dataset, nMuon_isTrig=arrays["Muon_pt"][arrays["Muon_isTriggering"]==1].count())
    rhistograms["Muon_pt"].fill(dataset=dataset, Muon_pt=arrays["Muon_pt"].flatten())
    rhistograms["Muon_pt_isTrig"].fill(dataset=dataset, Muon_pt_isTrig=arrays["Muon_pt"][arrays["Muon_isTriggering"]==1].flatten())

    rhistograms["BuToKMuMu_fit_pt"].fill(dataset=dataset, BToKMuMu_fit_pt=arrays["BToKMuMu_fit_pt"][reco_selection].flatten())
    rhistograms["BuToKMuMu_fit_eta"].fill(dataset=dataset, BToKMuMu_fit_eta=arrays["BToKMuMu_fit_eta"][reco_selection].flatten())
    rhistograms["BuToKMuMu_fit_phi"].fill(dataset=dataset, BToKMuMu_fit_phi=arrays["BToKMuMu_fit_phi"][reco_selection].flatten())
    rhistograms["BuToKMuMu_fit_mass"].fill(dataset=dataset, BToKMuMu_fit_mass=arrays["BToKMuMu_fit_mass"][reco_selection].flatten())
    rhistograms["BToKMuMu_chi2"].fill(dataset=dataset, BToKMuMu_chi2=arrays["BToKMuMu_chi2"][reco_selection].flatten())
    rhistograms["BuToKMuMu_fit_cos2D"].fill(dataset=dataset, BToKMuMu_fit_cos2D=arrays["BToKMuMu_fit_cos2D"][reco_selection].flatten())
    rhistograms["BuToKMuMu_l_xy"].fill(dataset=dataset, BToKMuMu_l_xy=arrays["BToKMuMu_l_xy"][reco_selection].flatten())
    rhistograms["BuToKMuMu_l_xy_sig"].fill(dataset=dataset, BToKMuMu_l_xy_sig=arrays["BToKMuMu_l_xy_sig"][reco_selection].flatten())

    rhistograms["BuToKMuMu_tag_fit_pt"].fill(dataset=dataset, BToKMuMu_fit_pt=arrays["BToKMuMu_fit_pt"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_fit_eta"].fill(dataset=dataset, BToKMuMu_fit_eta=arrays["BToKMuMu_fit_eta"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_fit_phi"].fill(dataset=dataset, BToKMuMu_fit_phi=arrays["BToKMuMu_fit_phi"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_fit_mass"].fill(dataset=dataset, BToKMuMu_fit_mass=arrays["BToKMuMu_fit_mass"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_fit_chi2"].fill(dataset=dataset, BToKMuMu_chi2=arrays["BToKMuMu_chi2"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_fit_cos2D"].fill(dataset=dataset, BToKMuMu_fit_cos2D=arrays["BToKMuMu_fit_cos2D"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_l_xy"].fill(dataset=dataset, BToKMuMu_l_xy=arrays["BToKMuMu_l_xy"][tag_selection].flatten())
    rhistograms["BuToKMuMu_tag_l_xy_sig"].fill(dataset=dataset, BToKMuMu_l_xy_sig=arrays["BToKMuMu_l_xy_sig"][tag_selection].flatten())

    rhistograms["BuToKMuMu_probe_fit_pt"].fill(dataset=dataset, BToKMuMu_fit_pt=arrays["BToKMuMu_fit_pt"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_fit_eta"].fill(dataset=dataset, BToKMuMu_fit_eta=arrays["BToKMuMu_fit_eta"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_fit_phi"].fill(dataset=dataset, BToKMuMu_fit_phi=arrays["BToKMuMu_fit_phi"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_fit_mass"].fill(dataset=dataset, BToKMuMu_fit_mass=arrays["BToKMuMu_fit_mass"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_fit_chi2"].fill(dataset=dataset, BToKMuMu_chi2=arrays["BToKMuMu_chi2"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_fit_cos2D"].fill(dataset=dataset, BToKMuMu_fit_cos2D=arrays["BToKMuMu_fit_cos2D"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_l_xy"].fill(dataset=dataset, BToKMuMu_l_xy=arrays["BToKMuMu_l_xy"][probe_selection].flatten())
    rhistograms["BuToKMuMu_probe_l_xy_sig"].fill(dataset=dataset, BToKMuMu_l_xy_sig=arrays["BToKMuMu_l_xy_sig"][probe_selection].flatten())

    # Truth matching
    arrays["BToKMuMu_l1_genIdx"] = arrays["Muon_genPartIdx"][arrays["BToKMuMu_l1Idx"]] 
    arrays["BToKMuMu_l2_genIdx"] = arrays["Muon_genPartIdx"][arrays["BToKMuMu_l2Idx"]] 
    arrays['BToKMuMu_k_genIdx']  = arrays['ProbeTracks_genPartIdx'][arrays['BToKMuMu_kIdx']]

    arrays['BToKMuMu_l1_genMotherIdx'] = where(arrays["BToKMuMu_l1_genIdx"] >= 0, 
                                                              arrays["GenPart_genPartIdxMother"][arrays["BToKMuMu_l1_genIdx"]], 
                                                              -1)
    arrays['BToKMuMu_l2_genMotherIdx'] = where(arrays["BToKMuMu_l2_genIdx"] >= 0, 
                                                              arrays["GenPart_genPartIdxMother"][arrays["BToKMuMu_l2_genIdx"]], 
                                                              -1)
    arrays['BToKMuMu_k_genMotherIdx'] = where(arrays["BToKMuMu_k_genIdx"] >= 0, 
                                                              arrays["GenPart_genPartIdxMother"][arrays["BToKMuMu_k_genIdx"]], 
                                                              -1)

    arrays['BToKMuMu_l1_genGrandmotherIdx'] = where(arrays['BToKMuMu_l1_genMotherIdx'] >= 0, 
                                                              arrays["GenPart_genPartIdxMother"][arrays['BToKMuMu_l1_genMotherIdx']], 
                                                              -1)
    arrays['BToKMuMu_l2_genGrandmotherIdx'] = where(arrays['BToKMuMu_l2_genMotherIdx'] >= 0, 
                                                              arrays["GenPart_genPartIdxMother"][arrays['BToKMuMu_l2_genMotherIdx']], 
                                                              -1)

    arrays['BToKMuMu_l1_genMotherPdgId'] = where(arrays['BToKMuMu_l1_genMotherIdx'] >= 0, 
                                                              arrays["GenPart_pdgId"][arrays['BToKMuMu_l1_genMotherIdx']],
                                                              -1)
    arrays['BToKMuMu_l2_genMotherPdgId'] = where(arrays['BToKMuMu_l2_genMotherIdx'] >= 0, 
                                                              arrays["GenPart_pdgId"][arrays['BToKMuMu_l2_genMotherIdx']],
                                                              -1)
    arrays['BToKMuMu_k_genMotherPdgId'] = where(arrays['BToKMuMu_k_genMotherIdx'] >= 0, 
                                                              arrays["GenPart_pdgId"][arrays['BToKMuMu_k_genMotherIdx']],
                                                              -1)

    arrays['BToKMuMu_l1_genGrandmotherPdgId'] = where(arrays['BToKMuMu_l1_genGrandmotherIdx'] >= 0, 
                                                              arrays["GenPart_pdgId"][arrays['BToKMuMu_l1_genGrandmotherIdx']],
                                                              -1)
    arrays['BToKMuMu_l2_genGrandmotherPdgId'] = where(arrays['BToKMuMu_l2_genGrandmotherIdx'] >= 0, 
                                                              arrays["GenPart_pdgId"][arrays['BToKMuMu_l2_genGrandmotherIdx']],
                                                              -1)

    arrays['BToKMuMu_mcmatch'] = (arrays['BToKMuMu_l1_genMotherPdgId'] == 443) \
                                            & (arrays['BToKMuMu_l2_genMotherPdgId'] == 443) \
                                            & (arrays['BToKMuMu_l2_genGrandmotherPdgId'] == 521) \
                                            & (arrays['BToKMuMu_l2_genGrandmotherPdgId'] == 521) \
                                            & (arrays['BToKMuMu_k_genMotherPdgId'] == 521) \
                                            & (arrays['BToKMuMu_l1_genGrandmotherIdx'] == arrays['BToKMuMu_l2_genGrandmotherIdx']) \
                                            & (arrays['BToKMuMu_l1_genGrandmotherIdx'] == arrays['BToKMuMu_k_genMotherIdx']) 

    arrays["BToKMuMu_genPartIdx"] = where(arrays['BToKMuMu_mcmatch'], arrays['BToKMuMu_l1_genGrandmotherIdx'], -1)


    # Build gen-to-reco map
    reco_genidx = arrays["BToKMuMu_genPartIdx"]
    reco_hasmatch = arrays["BToKMuMu_genPartIdx"] >= 0
    good_reco_genidx = reco_genidx[reco_hasmatch]
    good_gen_recoidx = reco_genidx.localindex[reco_hasmatch]

    # For each GenPart, default recomatch idx = -1
    gen_recoidx = arrays["GenPart_pdgId"].ones_like() * -1

    # For matched Bs, replace -1s with reco index
    gen_recoidx[good_reco_genidx] = good_gen_recoidx

    arrays["TruthBToKMuMu_recoIdx"] = gen_recoidx[abs(arrays["GenPart_pdgId"]) == 521]

    # "Selections" for truth Bs
    arrays["TruthBToKMuMu_hasRecoMatch"] = (arrays["TruthBToKMuMu_recoIdx"] >= 0)

    arrays["TruthBToKMuMu_hasRecoTagMatch"] = (arrays["TruthBToKMuMu_recoIdx"].ones_like() == -1) # Initialize to false
    arrays["TruthBToKMuMu_hasRecoTagMatch"][arrays["TruthBToKMuMu_recoIdx"] >= 0] = tag_selection[arrays["TruthBToKMuMu_recoIdx"][arrays["TruthBToKMuMu_recoIdx"] >= 0]]

    arrays["TruthBToKMuMu_hasRecoProbeMatch"] = (arrays["TruthBToKMuMu_recoIdx"].ones_like() == -1) # Initialize to false
    arrays["TruthBToKMuMu_hasRecoProbeMatch"][arrays["TruthBToKMuMu_recoIdx"] >= 0] = probe_selection[arrays["TruthBToKMuMu_recoIdx"][arrays["TruthBToKMuMu_recoIdx"] >= 0]]

    arrays["TruthBToKMuMu_pt"] = arrays["GenPart_pt"][abs(arrays["GenPart_pdgId"]) == 521]
    arrays["TruthBToKMuMu_eta"] = arrays["GenPart_eta"][abs(arrays["GenPart_pdgId"]) == 521]
    arrays["TruthBToKMuMu_phi"] = arrays["GenPart_phi"][abs(arrays["GenPart_pdgId"]) == 521]
    arrays["TruthBToKMuMu_mass"] = arrays["GenPart_mass"][abs(arrays["GenPart_pdgId"]) == 521]

    arrays["TruthBToKMuMu_recomatch_pt"] = arrays["TruthBToKMuMu_hasRecoMatch"].ones_like() * -1
    arrays["TruthBToKMuMu_recomatch_pt"][arrays["TruthBToKMuMu_hasRecoMatch"]] = arrays["BToKMuMu_fit_pt"][arrays["TruthBToKMuMu_recoIdx"][arrays["TruthBToKMuMu_hasRecoMatch"]]]

    arrays["TruthBToKMuMu_tagmatch_pt"] = arrays["TruthBToKMuMu_hasRecoMatch"].ones_like() * -1
    arrays["TruthBToKMuMu_tagmatch_pt"][arrays["TruthBToKMuMu_hasRecoTagMatch"]] = arrays["BToKMuMu_fit_pt"][arrays["TruthBToKMuMu_recoIdx"][arrays["TruthBToKMuMu_hasRecoTagMatch"]]]

    arrays["TruthBToKMuMu_probematch_pt"] = arrays["TruthBToKMuMu_hasRecoMatch"].ones_like() * -1
    arrays["TruthBToKMuMu_probematch_pt"][arrays["TruthBToKMuMu_hasRecoProbeMatch"]] = arrays["BToKMuMu_fit_pt"][arrays["TruthBToKMuMu_recoIdx"][arrays["TruthBToKMuMu_hasRecoProbeMatch"]]]

    # Fill truth histograms
    rhistograms["TruthBuToKMuMu_pt"].fill(dataset=dataset, TruthBToKMuMu_pt=arrays["TruthBToKMuMu_pt"].flatten())
    rhistograms["TruthBuToKMuMu_eta"].fill(dataset=dataset, TruthBToKMuMu_eta=arrays["TruthBToKMuMu_eta"].flatten())
    rhistograms["TruthBuToKMuMu_phi"].fill(dataset=dataset, TruthBToKMuMu_phi=arrays["TruthBToKMuMu_phi"].flatten())
    rhistograms["TruthBuToKMuMu_mass"].fill(dataset=dataset, TruthBToKMuMu_mass=arrays["TruthBToKMuMu_mass"].flatten())

    rhistograms["BuToKMuMu_truthpt_recopt_tag"].fill(dataset=dataset, TruthBToKMuMu_reco_pt=arrays["TruthBToKMuMu_tagmatch_pt"].flatten(), TruthBToKMuMu_truth_pt=arrays["TruthBToKMuMu_pt"].flatten())
    rhistograms["BuToKMuMu_truthpt_recopt_probe"].fill(dataset=dataset, TruthBToKMuMu_reco_pt=arrays["TruthBToKMuMu_probematch_pt"].flatten(), TruthBToKMuMu_truth_pt=arrays["TruthBToKMuMu_pt"].flatten())

    return dataset, tree.numentries, rhistograms

# Configure inputs
skim_directory = "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory"
in_txt = {
    "Bu2KJpsi2KMuMu_probefilter": "{}/v1_0/files_BuToKJpsi_ToMuMu_probefilter_SoftQCDnonD.txt".format(skim_directory),
    "Bu2KJpsi2KMuMu_inclusive": "{}/v1_0/files_BuToJpsiK_SoftQCDnonD.txt".format(skim_directory),
}


nworkers = 1
#fileslice = slice(None, 5)
fileslice = slice(None)
nevents = {}
#with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
    futures = set()
    for dataset, filelistpath in in_txt.items():
      with open(filelistpath) as filelist:
        files = [x.strip() for x in filelist.readlines()]
      print(files)
      futures.update(executor.submit(process_file, dataset, f) for f in files)
      nevents[dataset] = 0
    try:
        total = len(futures)
        processed = 0
        while len(futures) > 0:
            finished = set(job for job in futures if job.done())
            for job in finished:
                dataset, nentries, rhistograms = job.result()
                nevents[dataset] += nentries
                for k in rhistograms.keys():
                    histograms[k] += rhistograms[k]
                processed += 1
                print("Processing: done with % 4d / % 4d files" % (processed, total))
            futures -= finished
        del finished
    except KeyboardInterrupt:
        print("Ok quitter")
        for job in futures: job.cancel()
    except:
        for job in futures: job.cancel()
        raise

with gzip.open("Bu2KJsi2KMuMu_eff_hists.pkl.gz", "wb") as fout:
    pickle.dump(histograms, fout)
