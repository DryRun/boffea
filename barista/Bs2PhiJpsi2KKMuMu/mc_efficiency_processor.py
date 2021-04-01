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
from functools import partial

import uproot
import numpy as np
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
import awkward
import copy
from coffea.analysis_objects import JaggedCandidateArray

from brazil.aguapreta import *
import brazil.dataframereader as dataframereader
from brazil.Bcand_accumulator import Bcand_accumulator
from brazil.count_photon_children import count_photon_children

np.set_printoptions(threshold=np.inf)

def where(predicate, iftrue, iffalse):
    predicate = predicate.astype(np.bool)   # just to make sure they're 0/1
    return predicate*iftrue + (1 - predicate)*iffalse

class MCEfficencyProcessor(processor.ProcessorABC):
  def __init__(self):
    self._triggers = ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]

    # Histograms
    dataset_axis = hist.Cat("dataset", "Primary dataset")
    selection_axis_reco = hist.Cat("selection", "Selection name (reco)")
    selection_axis_truth = hist.Cat("selection", "Selection name (truth)")

    self._accumulator = processor.dict_accumulator()
    self._accumulator["nevents"] = processor.defaultdict_accumulator(int)
    self._accumulator["reco_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    self._accumulator["reco_cutflow_lowpt"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    self._accumulator["truth_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))

    self._Bcand_selections = ["recomatch"]
    for trigger in self._triggers:
      self._Bcand_selections.append(f"tagmatch_{trigger}")
      self._Bcand_selections.append(f"probematch_{trigger}")

      self._Bcand_selections.append(f"tagMaxPtmatch_{trigger}")
      self._Bcand_selections.append(f"probeMaxPtmatch_{trigger}")
      #self._Bcand_selections.append(f"probeJmatch_{trigger}")
    for selection_name in self._Bcand_selections:
      self._accumulator[f"Bcands_{selection_name}"] = processor.defaultdict_accumulator(partial(Bcand_accumulator, cols=["pt", "eta", "y", "phi", "mass"]))

    self._accumulator["Bcands_Bs_opt"] = processor.defaultdict_accumulator(
          partial(Bcand_accumulator, cols=[ "pt", "eta", "y", "phi", "mass", 
            "sv_prob", "l_xy_sig", "l_xy", "cos2D", 
            "dm_phi", "dm_jpsi", 
            "l1_pt", "l2_pt", "k1_pt", "k2_pt", 
            "l1_eta", "l2_eta", "k1_eta", "k2_eta", 
            "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", 
    ])) 

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    #self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BsToKKMuMu_fit_pt_absy_mass"] = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                            hist.Bin("fit_absy", r"$|y^{(fit)}|$", 50, 0.0, 2.5),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1)
                                                          )
    self._accumulator["BsToKKMuMu_fit_pt"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_eta"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 100, -2.5, 2.5))
    self._accumulator["BsToKKMuMu_fit_y"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_y", r"$y^{(fit)}$", 100, -2.5, 2.5))
    self._accumulator["BsToKKMuMu_fit_phi"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BsToKKMuMu_fit_mass"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    self._accumulator["BsToKKMuMu_chi2"]        = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_cos2D"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_cos2D", r"Fit cos2D", 500, 0.5, 1.))
    self._accumulator["BsToKKMuMu_fit_theta2D"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 300, 0., 0.75))
    self._accumulator["BsToKKMuMu_l_xy"]        = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BsToKKMuMu_l_xy_sig"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$", 50, -1.0, 4.0))
    self._accumulator["BsToKKMuMu_sv_prob"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("sv_prob", r"$SV prob$", 50, 0.0, 1.0))
    self._accumulator["BsToKKMuMu_jpsi_mass"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["BsToKKMuMu_phi_mass"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(\phi)$ [GeV]", 100, PHI_1020_MASS * 0.8, PHI_1020_MASS * 1.2))

    #self._accumulator["Bs_cos2D_vs_pt"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("log1mcos2D", r"log(1-cos2D)", 40, -4, 0.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_svprob_vs_pt"]  = hist.Hist("Events", dataset_axis, hist.Bin("sv_prob", r"SV prob", 100, 0., 1.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_l1_softId_vs_pt"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l1_softId", r"l1 soft ID", 2, -0.5, 1.5), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_l2_softId_vs_pt"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l2_softId", r"l2 soft ID", 2, -0.5, 1.5), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_lxysig_vs_pt"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$", 60, -1., 11.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_phimass_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("phi_m", r"$m(\phi)$ [GeV]",  50, PHI_1020_MASS*0.8, PHI_1020_MASS*1.2), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_phipt_vs_pt"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("phi_pt", r"$p_{T}(\phi)$ [GeV]",  50, 0., 25.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_k1pt_vs_pt"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("k1_pt", r"$p_{T}(k1)$ [GeV]", 60, 0., 15.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_k2pt_vs_pt"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("k2_pt", r"$p_{T}(k2)$ [GeV]", 60, 0., 15.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_l1pt_vs_l1eta_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l1_pt", r"$p_{T}(l1)$ [GeV]", 50, 0., 25.), hist.Bin("l1_eta", r"$\eta(l1)$ [GeV]", 30, 0., 3.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    #self._accumulator["Bs_l2pt_vs_l2eta_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l2_pt", r"$p_{T}(l2)$ [GeV]", 50, 0., 25.), hist.Bin("l2_eta", r"$\eta(l2)$ [GeV]", 30, 0., 3.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))

    self._accumulator["NM1_BsToKKMuMu_phi_m"]    = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("mass", r"$m_{\phi}$ [GeV]", 100, PHI_1020_MASS * 0.8, PHI_1020_MASS * 1.2))
    self._accumulator["NM1_BsToKKMuMu_jpsi_m"]   = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("mass", r"$m_{J/\psi}$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["NM1_BsToKKMuMu_l_xy_sig"] = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 9.0))
    self._accumulator["NM1_BsToKKMuMu_sv_prob"]  = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("sv_prob", r"SV prob.", 50, 0., 1.))
    self._accumulator["NM1_BsToKKMuMu_cos2D"]    = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("cos2D", r"SV cos2D.", 50, -1., 1.))

    #self._accumulator["BsToKKMuMu_tag_fit_pt"]    = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["BsToKKMuMu_tag_fit_eta"]   = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    #self._accumulator["BsToKKMuMu_tag_fit_phi"]   = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    #self._accumulator["BsToKKMuMu_tag_fit_mass"]  = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    #self._accumulator["BsToKKMuMu_tag_chi2"]      = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    #self._accumulator["BsToKKMuMu_tag_fit_cos2D"] = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    #self._accumulator["BsToKKMuMu_tag_l_xy"]      = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    #self._accumulator["BsToKKMuMu_tag_l_xy_sig"]  = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

    #self._accumulator["BsToKKMuMu_probe_fit_pt"]    = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["BsToKKMuMu_probe_fit_eta"]   = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    #self._accumulator["BsToKKMuMu_probe_fit_phi"]   = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    #self._accumulator["BsToKKMuMu_probe_fit_mass"]  = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    #self._accumulator["BsToKKMuMu_probe_chi2"]      = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    #self._accumulator["BsToKKMuMu_probe_fit_cos2D"] = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    #self._accumulator["BsToKKMuMu_probe_l_xy"]      = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    #self._accumulator["BsToKKMuMu_probe_l_xy_sig"]  = hist.Hist("Events", dataset_axis, hist.Bin("BsToKKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

    self._accumulator["nTruthMuon"]  = hist.Hist("Events", dataset_axis, hist.Bin("nTruthMuon", r"N(truth muons)", 11, -0.5, 10.5))

    self._accumulator["TruthProbeMuon_parent"]      = hist.Hist("Events", dataset_axis, hist.Bin("parentPdgId", "Parent pdgId", 1001, -0.5, 1000.5))
    self._accumulator["TruthProbeMuon_grandparent"] = hist.Hist("Events", dataset_axis, hist.Bin("grandparentPdgId", "Grandparent pdgId", 1001, -0.5, 1000.5))

    self._accumulator["TruthBsToKKMuMu_pt_absy_mass"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                       hist.Bin("pt", r"$p_{T}$ [GeV]", 100, 0.0, 100.0),
                                                       hist.Bin("absy", r"$|y|$", 50, 0.0, 2.5),
                                                       hist.Bin("mass", r"$m$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    self._accumulator["TruthBsToKKMuMu_pt"]               = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBsToKKMuMu_eta"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("eta", r"$\eta$", 100, -2.5, 2.5))
    self._accumulator["TruthBsToKKMuMu_y"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("y", r"y", 100, -2.5, 2.5))
    self._accumulator["TruthBsToKKMuMu_phi"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["TruthBsToKKMuMu_mass"]             = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("mass", r"$m$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    self._accumulator["TruthBsToKKMuMu_recopt_d_truthpt"] = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("recopt_d_truthpt", r"$p_{T}^{(reco)} / p_{T}^{(truth)}$", 200, 0., 2.))

    self._accumulator["TruthBsToKKMuMu_kp_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(K^{+})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(K^{+})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(K^{+})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(K^{+})$", 30, 0., K_MASS*2.0)
                                                  )
    self._accumulator["TruthBsToKKMuMu_km_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(K^{-})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(K^{-})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(K^{-})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(K^{-})$", 30, 0., K_MASS*2.0)
                                                  )
    self._accumulator["TruthBsToKKMuMu_mup_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{+})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{+})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{+})$", 30, 0., MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBsToKKMuMu_mum_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{-})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{-})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{-})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{-})$", 30, 0., MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBsToKKMuMu_phi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\phi)$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\phi)$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\phi)$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\phi)$", 30, 0., PHI_1020_MASS*2.0)
                                                  )
    self._accumulator["TruthBsToKKMuMu_jpsi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(J/\psi)$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(J/\psi)$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(J/\psi)$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(J/\psi)$", 30, 0., JPSI_1S_MASS*2.0)
                                                  )
    #self._accumulator["TruthBsToKKMuMu_dR_kp_km"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                                  hist.Bin("dR", r"$\Delta R$", 70, 0., 7.)
    #                                                )
    #self._accumulator["TruthBsToKKMuMu_pt_unmatched"]                   = hist.Hist("Events", dataset_axis, hist.Bin("TruthBsToKKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["TruthBsToKKMuMu_pt_matched"]                     = hist.Hist("Events", dataset_axis, hist.Bin("TruthBsToKKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["TruthBsToKKMuMu_pt_matched_tag"]                 = hist.Hist("Events", dataset_axis, hist.Bin("TruthBsToKKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["TruthBsToKKMuMu_pt_matched_probe"]               = hist.Hist("Events", dataset_axis, hist.Bin("TruthBsToKKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["TruthBsToKKMuMu_recopt_d_truthpt_matched_tag"]   = hist.Hist("Events", dataset_axis, hist.Bin("recopt_d_truthpt", r"$p_{T}^{(reco)} / p_{T}^{(truth)}$", 200, 0., 2.))
    #self._accumulator["TruthBsToKKMuMu_recopt_d_truthpt_matched_probe"] = hist.Hist("Events", dataset_axis, hist.Bin("recopt_d_truthpt", r"$p_{T}^{(reco)} / p_{T}^{(truth)}$", 200, 0., 2.))

    # One entry per truth B
    # - If truth B is not matched to reco, or if reco fails selection, fill (-1, truthpt)
    self._accumulator["TruthBsToKKMuMu_truthpt_recopt"] = hist.Hist("Events", dataset_axis, selection_axis_truth,
                                              hist.Bin("reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 500, 0., 100.),
                                              hist.Bin("truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 500, 0., 100.))



  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    output["nevents"][dataset_name] += df.size

    # Create jagged object arrays
    reco_bskkmumu = dataframereader.reco_bskkmumu(df, is_mc=True)
    reco_muons    = dataframereader.reco_muons(df, is_mc=True)
    trigger_muons = dataframereader.trigger_muons(df, is_mc=True)
    probe_tracks  = dataframereader.probe_tracks(df, is_mc=True)
    genparts      = dataframereader.genparts(df, is_mc=True)

    reco_bskkmumu.add_attributes(
      l1_softId    = reco_muons.softId[reco_bskkmumu.l1_idx],
      l1_softMvaId = reco_muons.softMvaId[reco_bskkmumu.l1_idx],
      l2_softId    = reco_muons.softId[reco_bskkmumu.l2_idx],
      l2_softMvaId = reco_muons.softMvaId[reco_bskkmumu.l2_idx],
      trk1_charge  = probe_tracks.charge[reco_bskkmumu.trk1_idx],
      trk2_charge  = probe_tracks.charge[reco_bskkmumu.trk2_idx],
      kp_pt = where(probe_tracks.charge[reco_bskkmumu.trk1_idx] > 0, 
                    probe_tracks.pt[reco_bskkmumu.trk1_idx], 
                    probe_tracks.pt[reco_bskkmumu.trk2_idx]),
      km_pt = where(probe_tracks.charge[reco_bskkmumu.trk1_idx] > 0, 
                    probe_tracks.pt[reco_bskkmumu.trk2_idx], 
                    probe_tracks.pt[reco_bskkmumu.trk1_idx]),
    )

    # Truth matching
    reco_bskkmumu.l1_genIdx = reco_muons.genPartIdx[reco_bskkmumu.l1_idx] 
    reco_bskkmumu.l2_genIdx = reco_muons.genPartIdx[reco_bskkmumu.l2_idx] 
    reco_bskkmumu.k1_genIdx = probe_tracks.genPartIdx[reco_bskkmumu.trk1_idx]
    reco_bskkmumu.k2_genIdx = probe_tracks.genPartIdx[reco_bskkmumu.trk2_idx]

    reco_bskkmumu.l1_genMotherIdx = where(reco_bskkmumu.l1_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.l1_genIdx], 
                                                              -1)
    reco_bskkmumu.l2_genMotherIdx = where(reco_bskkmumu.l2_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.l2_genIdx], 
                                                              -1)
    reco_bskkmumu.k1_genMotherIdx = where(reco_bskkmumu.k1_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.k1_genIdx], 
                                                              -1)
    reco_bskkmumu.k2_genMotherIdx = where(reco_bskkmumu.k2_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.k2_genIdx], 
                                                              -1)

    reco_bskkmumu.l1_genGrandmotherIdx = where(reco_bskkmumu.l1_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.l1_genMotherIdx], 
                                                              -1)
    reco_bskkmumu.l2_genGrandmotherIdx = where(reco_bskkmumu.l2_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.l2_genMotherIdx], 
                                                              -1)
    reco_bskkmumu.k1_genGrandmotherIdx = where(reco_bskkmumu.k1_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.k1_genMotherIdx], 
                                                              -1)
    reco_bskkmumu.k2_genGrandmotherIdx = where(reco_bskkmumu.k2_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bskkmumu.k2_genMotherIdx], 
                                                              -1)

    reco_bskkmumu.l1_genMotherPdgId = where(reco_bskkmumu.l1_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.l1_genMotherIdx],
                                                              -1)
    reco_bskkmumu.l2_genMotherPdgId = where(reco_bskkmumu.l2_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.l2_genMotherIdx],
                                                              -1)
    reco_bskkmumu.k1_genMotherPdgId = where(reco_bskkmumu.k1_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.k1_genMotherIdx],
                                                              -1)
    reco_bskkmumu.k2_genMotherPdgId = where(reco_bskkmumu.k2_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.k2_genMotherIdx],
                                                              -1)

    reco_bskkmumu.l1_genGrandmotherPdgId = where(reco_bskkmumu.l1_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.l1_genGrandmotherIdx],
                                                              -1)
    reco_bskkmumu.l2_genGrandmotherPdgId = where(reco_bskkmumu.l2_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.l2_genGrandmotherIdx],
                                                              -1)
    reco_bskkmumu.k1_genGrandmotherPdgId = where(reco_bskkmumu.k1_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.k1_genGrandmotherIdx],
                                                              -1)
    reco_bskkmumu.k2_genGrandmotherPdgId = where(reco_bskkmumu.k2_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bskkmumu.k2_genGrandmotherIdx],
                                                              -1)

    reco_bskkmumu.mcmatch = (abs(reco_bskkmumu.l1_genMotherPdgId) == 443) \
                            & (abs(reco_bskkmumu.l2_genMotherPdgId) == 443) \
                            & (abs(reco_bskkmumu.l1_genGrandmotherPdgId) == 531) \
                            & (abs(reco_bskkmumu.l2_genGrandmotherPdgId) == 531) \
                            & (abs(reco_bskkmumu.k1_genMotherPdgId) == 333) \
                            & (abs(reco_bskkmumu.k2_genMotherPdgId) == 333) \
                            & (abs(reco_bskkmumu.k1_genGrandmotherPdgId) == 531) \
                            & (abs(reco_bskkmumu.k2_genGrandmotherPdgId) == 531) \
                            & (reco_bskkmumu.l1_genGrandmotherIdx == reco_bskkmumu.l2_genGrandmotherIdx) \
                            & (reco_bskkmumu.l1_genGrandmotherIdx == reco_bskkmumu.k1_genGrandmotherIdx) \
                            & (reco_bskkmumu.l1_genGrandmotherIdx == reco_bskkmumu.k2_genGrandmotherIdx)

    reco_bskkmumu.genPartIdx = where(reco_bskkmumu.mcmatch, reco_bskkmumu.l1_genGrandmotherIdx, -1)


    # Tag/probe selection
    """ Old, before separating by trigger
    reco_bskkmumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggeringFull[reco_bskkmumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggeringFull[reco_bskkmumu.l2_idx])
    reco_bskkmumu.add_attributes(MuonIsTrigCount = reco_bskkmumu.Muon1IsTrig.astype(int) + reco_bskkmumu.Muon2IsTrig.astype(int))
    event_ntriggeringmuons = reco_muons.isTriggeringFull.astype(int).sum()
    reco_bskkmumu.add_attributes(TagCount = reco_bskkmumu.MuonIsTrigCount.ones_like() * event_ntriggeringmuons - reco_bskkmumu.MuonIsTrigCount)

    reco_bskkmumu.add_attributes(l_xy_sig = where(reco_bskkmumu.l_xy_unc > 0, reco_bskkmumu.l_xy / reco_bskkmumu.l_xy_unc, -1.e20))
    """
    tagmuon_ptcuts = {
     "HLT_Mu7_IP4": 7*1.05,
     "HLT_Mu9_IP5": 9*1.05,
     "HLT_Mu9_IP6": 9*1.05,
     "HLT_Mu12_IP6": 12*1.05,
    }
    tagmuon_ipcuts = {
     "HLT_Mu7_IP4": 4 * 1.05, 
     "HLT_Mu9_IP5": 5 * 1.05, 
     "HLT_Mu9_IP6": 5 * 1.05, 
     "HLT_Mu12_IP6": 6 * 1.05, 
    }
    for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
      reco_bskkmumu.add_attributes(**{
        f"Muon1IsTrig_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bskkmumu.l1_idx],
        f"Muon2IsTrig_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bskkmumu.l2_idx],
        f"Muon1IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bskkmumu.l1_idx] \
                                        & (reco_muons.pt[reco_bskkmumu.l1_idx] > tagmuon_ptcuts[trigger]) \
                                        & (abs(reco_muons.dxySig[reco_bskkmumu.l1_idx]) > tagmuon_ipcuts[trigger]),
        f"Muon2IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bskkmumu.l2_idx] \
                                        & (reco_muons.pt[reco_bskkmumu.l2_idx] > tagmuon_ptcuts[trigger]) \
                                        & (abs(reco_muons.dxySig[reco_bskkmumu.l2_idx]) > tagmuon_ipcuts[trigger]),
      })
      reco_bskkmumu.add_attributes(**{
        f"Muon1IsTrigMaxPt_{trigger}": getattr(reco_bskkmumu, f"Muon1IsTrigTight_{trigger}") & (reco_muons.pt[reco_bskkmumu.l1_idx] == reco_muons.pt.max()),
        f"Muon2IsTrigMaxPt_{trigger}": getattr(reco_bskkmumu, f"Muon2IsTrigTight_{trigger}") & (reco_muons.pt[reco_bskkmumu.l2_idx] == reco_muons.pt.max()),
      })

      reco_bskkmumu.add_attributes(**{
        f"MuonIsTrigCount_{trigger}": getattr(reco_bskkmumu, f"Muon1IsTrig_{trigger}").astype(int) + getattr(reco_bskkmumu, f"Muon2IsTrig_{trigger}").astype(int)
      })
      event_ntriggeringmuons = getattr(reco_muons, f"isTriggeringFull_{trigger}").astype(int).sum()
      reco_bskkmumu.add_attributes(**{
        f"TagCount_{trigger}": getattr(reco_bskkmumu, f"MuonIsTrigCount_{trigger}").ones_like() * event_ntriggeringmuons - getattr(reco_bskkmumu, f"MuonIsTrigCount_{trigger}"),
      })

      reco_bskkmumu.add_attributes(**{
        f"MuonIsTrigCountMaxPt_{trigger}": getattr(reco_bskkmumu, f"Muon1IsTrigMaxPt_{trigger}").astype(int) + getattr(reco_bskkmumu, f"Muon2IsTrigMaxPt_{trigger}").astype(int)
      })
      event_ntriggeringmuons_maxpt =  getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_muons.pt == reco_muons.pt.max()].astype(int).sum()
      reco_bskkmumu.add_attributes(**{
        f"TagCountMaxPt_{trigger}": getattr(reco_bskkmumu, f"MuonIsTrigCountMaxPt_{trigger}").ones_like() * event_ntriggeringmuons_maxpt - getattr(reco_bskkmumu, f"MuonIsTrigCountMaxPt_{trigger}"),
      })


    reco_bskkmumu.add_attributes(l_xy_sig = where(reco_bskkmumu.l_xy_unc > 0, reco_bskkmumu.l_xy / reco_bskkmumu.l_xy_unc, -1.e20))

    # General selection
    reco_bskkmumu_mask_template = reco_bskkmumu.pt.ones_like().astype(bool)
    selections = {}
    selections["sv_pt"]    = (reco_bskkmumu.pt > 3.0)
    selections["l_xy_sig"] = (abs(reco_bskkmumu.l_xy_sig) > final_cuts["Bs"]["l_xy_sig"])
    selections["sv_prob"]  = (reco_bskkmumu.sv_prob > final_cuts["Bs"]["sv_prob"])
    selections["cos2D"]    = (reco_bskkmumu.fit_cos2D > final_cuts["Bs"]["cos2D"])
    selections["Bs"]["l1"] =  (reco_bskkmumu.l1_pt > final_cuts["Bs"]["l1_pt"]) & (abs(reco_bskkmumu.l1_eta) < 2.4) & (reco_bskkmumu.l1_softId)
    selections["Bs"]["l2"] =  (reco_bskkmumu.l2_pt > final_cuts["Bs"]["l2_pt"]) & (abs(reco_bskkmumu.l2_eta) < 2.4) & (reco_bskkmumu.l2_softId)
    #selections["l1"]       = (reco_bskkmumu.l1_pt > 1.5) & (abs(reco_bskkmumu.l1_eta) < 2.4) & (reco_bskkmumu.l1_softId)
    #selections["l2"]       = (abs(reco_bskkmumu.l2_eta) < 2.4) & (reco_bskkmumu.l2_softId)
    #selections["l2"]       = (selections["l2"] & where(abs(reco_bskkmumu.l2_eta) < 1.4, 
    #                                                  (reco_bskkmumu.l2_pt > 1.5), 
    #                                                  (reco_bskkmumu.l2_pt > 1.0))).astype(bool)
    selections["k1"]       = (reco_bskkmumu.trk1_pt > final_cuts["Bs"]["k1_pt"]) & (abs(reco_bskkmumu.trk1_eta) < 2.5)
    selections["k2"]       = (reco_bskkmumu.trk2_pt > final_cuts["Bs"]["k2_pt"]) & (abs(reco_bskkmumu.trk2_eta) < 2.5)
    #selections["dR"]       = (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.trk2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.trk2_phi) > 0.03) \
    #                        & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l1_phi) > 0.03) \
    #                        & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l2_phi) > 0.03) \
    #                        & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l1_phi) > 0.03) \
    #                        & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l2_phi) > 0.03) \
    #                        & (delta_r(reco_bskkmumu.l1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.l1_phi, reco_bskkmumu.l2_phi) > 0.03)
    selections["jpsi"]       = abs(reco_bskkmumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    #(JPSI_1S_MASS - JPSI_WINDOW < reco_bskkmumu.mll_fullfit) & (reco_bskkmumu.mll_fullfit < JPSI_1S_MASS + JPSI_WINDOW)
    selections["phi"]        = (abs(reco_bskkmumu.phi_m - PHI_1020_MASS) < final_cuts["Bs"]["dm_phi"]) & (reco_bskkmumu.trk1_charge + reco_bskkmumu.trk2_charge == 0)
    #(PHI_1020_MASS - PHI_WINDOW < reco_bskkmumu.phi_m) & (reco_bskkmumu.phi_m < PHI_1020_MASS + PHI_WINDOW)
    selections["kstar_veto"] = (abs(reco_bskkmumu.Kstar1_mass - KSTAR_892_MASS) > final_cuts["Bs"]["kstar_veto"]) \
                                & (abs(reco_bskkmumu.Kstar2_mass - KSTAR_892_MASS) > final_cuts["Bs"]["kstar_veto"])
    #(reco_bskkmumu.phi_m < KSTAR_892_MASS - KSTAR_WINDOW) | (KSTAR_892_MASS + KSTAR_WINDOW < reco_bskkmumu.phi_m)
    #selections["phi_pt"]       = reco_bskkmumu.phi_pt > 1.0

    #selections["trigger"] = ((df["HLT_Mu7_IP4"] == 1) |(df["HLT_Mu9_IP5"] == 1) | (df["HLT_Mu9_IP6"] == 1)) * reco_bskkmumu_mask_template # Shape = event!

    # Final selections
    selections["inclusive"] = reco_bskkmumu.fit_pt.ones_like().astype(bool)
    selections["reco"]      = selections["sv_pt"] \
                              & selections["l_xy_sig"] \
                              & selections["sv_prob"] \
                              & selections["cos2D"] \
                              & selections["l1"] \
                              & selections["l2"] \
                              & selections["k1"] \
                              & selections["k2"] \
                              & selections["jpsi"] \
                              & selections["phi"] \
                              & selections["kstar_veto"]
                              #& selections["dR"] \
    selections["truthmatched"] = (reco_bskkmumu.genPartIdx >= 0)
    selections["recomatch"]      = selections["reco"] & selections["truthmatched"]


    for trigger in self._triggers:
      trigger_mask = ((df[trigger] == 1) & (df[l1_seeds[trigger]] == 1)) * reco_bskkmumu_mask_template # 

      selections[f"recomatch_{trigger}"]      = selections["reco"] & trigger_mask

      selections[f"tag_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bskkmumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bskkmumu, f"Muon2IsTrigTight_{trigger}"))
      selections[f"tagmatch_{trigger}"]       = selections[f"tag_{trigger}"] & selections["truthmatched"]
      selections[f"tagunmatched_{trigger}"]   = selections[f"tag_{trigger}"] & (~selections["truthmatched"])

      selections[f"probe_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bskkmumu, f"TagCount_{trigger}") >= 1)
      selections[f"probematch_{trigger}"]     = selections[f"probe_{trigger}"] & selections["truthmatched"]
      selections[f"probeunmatched_{trigger}"] = selections[f"probe_{trigger}"] & (~selections["truthmatched"])

      selections[f"tagMaxPt_{trigger}"]            = selections["reco"] & trigger_mask & \
                                                      (
                                                        (getattr(reco_bskkmumu, f"Muon1IsTrigTight_{trigger}") & (reco_muons.pt[reco_bskkmumu.l1_idx] == reco_muons.pt.max())) \
                                                        | (getattr(reco_bskkmumu, f"Muon2IsTrigTight_{trigger}") & (reco_muons.pt[reco_bskkmumu.l2_idx] == reco_muons.pt.max()))
                                                      )
      selections[f"tagMaxPtmatch_{trigger}"]       = selections[f"tagMaxPt_{trigger}"] & selections["truthmatched"]
      selections[f"tagMaxPtunmatched_{trigger}"]   = selections[f"tagMaxPt_{trigger}"] & (~selections["truthmatched"])

      selections[f"probeMaxPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bskkmumu, f"TagCountMaxPt_{trigger}") >= 1)
      selections[f"probeMaxPtmatch_{trigger}"]     = selections[f"probeMaxPt_{trigger}"] & selections["truthmatched"]
      selections[f"probeMaxPtunmatched_{trigger}"] = selections[f"probeMaxPt_{trigger}"] & (~selections["truthmatched"])

    # If more than one B is selected, choose best chi2
    selections["recomatch"] = selections["recomatch"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["recomatch"]].min())
    for trigger in self._triggers:
      selections[f"recomatch_{trigger}"] = selections[f"recomatch_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"recomatch_{trigger}"]].min()) 
      selections[f"tag_{trigger}"]            = selections[f"tag_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"tag_{trigger}"]].min())
      selections[f"tagmatch_{trigger}"]       = selections[f"tagmatch_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"tagmatch_{trigger}"]].min())
      selections[f"tagunmatched_{trigger}"]   = selections[f"tagunmatched_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"tagunmatched_{trigger}"]].min())
      selections[f"probe_{trigger}"]          = selections[f"probe_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"probe_{trigger}"]].min())
      selections[f"probematch_{trigger}"]     = selections[f"probematch_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"probematch_{trigger}"]].min())
      selections[f"probeunmatched_{trigger}"] = selections[f"probeunmatched_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"probeunmatched_{trigger}"]].min())

      selections[f"tagMaxPtmatch_{trigger}"]       = selections[f"tagMaxPtmatch_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"tagMaxPtmatch_{trigger}"]].min())
      selections[f"probeMaxPtmatch_{trigger}"]     = selections[f"probeMaxPtmatch_{trigger}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections[f"probeMaxPtmatch_{trigger}"]].min())

    # Fill cutflow
    cumulative_selection = reco_bskkmumu.pt.ones_like().astype(bool)
    output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "jpsi", "phi", "kstar_veto"]:
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow"][dataset_name]["tag"] += selections["tag_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probe"] += selections["probe_HLT_Mu7_IP4"].sum().sum()

    cumulative_selection = reco_bskkmumu.pt.ones_like().astype(bool)
    output["reco_cutflow_lowpt"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    cumulative_selection = cumulative_selection & (reco_bskkmumu.pt >= 5.0) & (reco_bskkmumu.pt <= 10.0)
    output["reco_cutflow_lowpt"][dataset_name]["lowpt"] = cumulative_selection.sum().sum()
    for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "jpsi", "phi", "kstar_veto"]:
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow_lowpt"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow_lowpt"][dataset_name]["tag"] += selections["tag_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow_lowpt"][dataset_name]["probe"] += selections["probe_HLT_Mu7_IP4"].sum().sum()

    # Fill reco histograms
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    #output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    selection_names = ["inclusive", "reco", "recomatch", "truthmatched"]
    for trigger in self._triggers:
      selection_names.extend([f"recomatch_{trigger}", f"tag_{trigger}", f"tagmatch_{trigger}", f"tagunmatched_{trigger}", f"probe_{trigger}", f"probematch_{trigger}", f"probeunmatched_{trigger}"])
    for selection_name in selection_names:
      output["BsToKKMuMu_fit_pt_absy_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_absy=np.abs(reco_bskkmumu.fit_y[selections[selection_name]].flatten()),
                                            fit_mass=reco_bskkmumu.fit_mass[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bskkmumu.fit_eta[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_y"].fill(dataset=dataset_name, selection=selection_name, fit_y=reco_bskkmumu.fit_y[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bskkmumu.fit_phi[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bskkmumu.fit_mass[selections[selection_name]].flatten())
      output["BsToKKMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bskkmumu.chi2[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bskkmumu.fit_cos2D[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bskkmumu.fit_cos2D[selections[selection_name]]).flatten())
      output["BsToKKMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bskkmumu.l_xy[selections[selection_name]].flatten())
      output["BsToKKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bskkmumu.l_xy_sig[selections[selection_name]].flatten())
      output["BsToKKMuMu_sv_prob"].fill(dataset=dataset_name, selection=selection_name, sv_prob=reco_bskkmumu.sv_prob[selections[selection_name]].flatten())
      output["BsToKKMuMu_jpsi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.mll_fullfit[selections[selection_name]].flatten())
      output["BsToKKMuMu_phi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.phi_m[selections[selection_name]].flatten())
      output["Bs_l1_softId_vs_pt"].fill(dataset=dataset_name, selection=selection_name, l1_softId=reco_bskkmumu.l1_softId[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_l2_softId_vs_pt"].fill(dataset=dataset_name, selection=selection_name, l2_softId=reco_bskkmumu.l2_softId[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
    # SV optimization
    #svprob_opt_selection = selections["sv_pt"] \
    #                          & selections["l_xy_sig"] \
    #                          & selections["cos2D"] \
    #                          & selections["l1"] \
    #                          & selections["l2"] \
    #                          & selections["k1"] \
    #                          & selections["k2"] \
    #                          & selections["jpsi"] \
    #                          & selections["phi"] \
    #                          & selections["kstar_veto"] \
    #                          #& selections["dR"] \
    #output["Bs_svprob_vs_pt"].fill(dataset=dataset_name, sv_prob=reco_bskkmumu.sv_prob[svprob_opt_selection].flatten(), pt=reco_bskkmumu.fit_pt[svprob_opt_selection].flatten())
    '''                        
    for selection_name in ["truthmatched"]:
      output["Bs_cos2D_vs_pt"].fill(dataset=dataset_name, selection=selection_name, log1mcos2D=np.log(1. - reco_bskkmumu.fit_cos2D[selections[selection_name]].flatten() + 1.e-40) / math.log(10), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_svprob_vs_pt"].fill(dataset=dataset_name, selection=selection_name, sv_prob=reco_bskkmumu.sv_prob[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_lxysig_vs_pt"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bskkmumu.l_xy_sig[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_phimass_vs_pt"].fill(dataset=dataset_name, selection=selection_name, phi_m=reco_bskkmumu.phi_m[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_phipt_vs_pt"].fill(dataset=dataset_name, selection=selection_name, phi_pt=reco_bskkmumu.phi_pt[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_l1pt_vs_l1eta_vs_pt"].fill(dataset=dataset_name, selection=selection_name, l1_pt=reco_bskkmumu.l1_pt[selections[selection_name]].flatten(), l1_eta=abs(reco_bskkmumu.l1_eta[selections[selection_name]].flatten()), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_l2pt_vs_l2eta_vs_pt"].fill(dataset=dataset_name, selection=selection_name, l2_pt=reco_bskkmumu.l2_pt[selections[selection_name]].flatten(), l2_eta=abs(reco_bskkmumu.l2_eta[selections[selection_name]].flatten()), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_k1pt_vs_pt"].fill(dataset=dataset_name, selection=selection_name, k1_pt=reco_bskkmumu.trk1_pt[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["Bs_k2pt_vs_pt"].fill(dataset=dataset_name, selection=selection_name, k2_pt=reco_bskkmumu.trk2_pt[selections[selection_name]].flatten(), pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
    '''

    # Build gen-to-reco map
    reco_genidx      = reco_bskkmumu.genPartIdx
    reco_hasmatch    = reco_genidx >= 0
    good_reco_genidx = reco_genidx[reco_hasmatch]
    good_gen_recoidx = reco_genidx.localindex[reco_hasmatch]

    # For each GenPart, default recomatch idx = -1
    gen_recoidx = genparts.pdgId.ones_like() * -1

    # For matched Bs, replace -1s with reco index
    gen_recoidx[good_reco_genidx] = good_gen_recoidx

    # Truth tree navigation: find K and mu daughters of Bs
    genpart_mother_idx = genparts.genPartIdxMother
    genpart_grandmother_idx = where(genpart_mother_idx >= 0, 
                                    genparts.genPartIdxMother[genpart_mother_idx],
                                    -1)
    genpart_mother_pdgId = where(genpart_mother_idx >= 0, genparts.pdgId[genpart_mother_idx], -1)
    genpart_grandmother_pdgId = where(genpart_grandmother_idx >= 0, genparts.pdgId[genpart_grandmother_idx], -1)

    mask_kp_frombs = (genparts.pdgId == 321) & (abs(genpart_mother_pdgId) == 333) & (abs(genpart_grandmother_pdgId) == 531)
    kp_frombs_grandmother_idx = genpart_grandmother_idx[mask_kp_frombs]
    bs_kp_idx = genparts.pt.ones_like().astype(int) * -1
    bs_kp_idx[kp_frombs_grandmother_idx] = genparts.pdgId.localindex[mask_kp_frombs]
    #for i in range(len(bs_kp_idx)):
    #  print("Event {}".format(i))
    #  print(genparts.pdgId[i])
    #  print(genpart_mother_pdgId[i])
    #  print(genpart_grandmother_pdgId[i])
    #  print(bs_kp_idx[i])

    mask_km_frombs = (genparts.pdgId == -321) & (abs(genpart_mother_pdgId) == 333) & (abs(genpart_grandmother_pdgId) == 531)
    km_frombs_grandmother_idx = genpart_grandmother_idx[mask_km_frombs]
    bs_km_idx = genparts.pt.ones_like().astype(int) * -1
    bs_km_idx[km_frombs_grandmother_idx] = genparts.pdgId.localindex[mask_km_frombs]

    mask_mup_frombs = (genparts.pdgId == -13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 531)
    mup_frombs_grandmother_idx = genpart_grandmother_idx[mask_mup_frombs]
    bs_mup_idx = genparts.pt.ones_like().astype(int) * -1
    bs_mup_idx[mup_frombs_grandmother_idx] = genparts.pdgId.localindex[mask_mup_frombs]

    mask_mum_frombs = (genparts.pdgId == 13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 531)
    mum_frombs_grandmother_idx = genpart_grandmother_idx[mask_mum_frombs]
    bs_mum_idx = genparts.pt.ones_like().astype(int) * -1
    bs_mum_idx[mum_frombs_grandmother_idx] = genparts.pdgId.localindex[mask_mum_frombs]
    
    mask_phi_frombs = (abs(genparts.pdgId) == 333) & (abs(genpart_mother_pdgId) == 531)
    phi_frombs_mother_idx = genpart_mother_idx[mask_phi_frombs]
    bs_phi_idx = genparts.pt.ones_like().astype(int) * -1
    bs_phi_idx[phi_frombs_mother_idx] = genparts.pdgId.localindex[phi_frombs_mother_idx]

    mask_jpsi_frombs = (abs(genparts.pdgId) == 443) & (abs(genpart_mother_pdgId) == 531)
    jpsi_frombs_mother_idx = genpart_mother_idx[mask_jpsi_frombs]
    bs_jpsi_idx = genparts.pt.ones_like().astype(int) * -1
    bs_jpsi_idx[jpsi_frombs_mother_idx] = genparts.pdgId.localindex[jpsi_frombs_mother_idx]

    # Count number of soft photon daughters
    nChildrenNoPhoton = genparts.nChildren - count_photon_children(genparts.genPartIdxMother, genparts.pdgId, genparts.pt)

    # Jagged array of truth BToKMuMus
    mask531 = (abs(genparts.pdgId)==531) & (bs_jpsi_idx >= 0) & (bs_phi_idx >= 0) & (bs_mup_idx >= 0) & (bs_mum_idx >= 0) & (bs_kp_idx >= 0) & (bs_km_idx >= 0) & (nChildrenNoPhoton == 2)
    #mask531 = mask531 & (mask531.sum() <= 1)
    truth_bskkmumu = JaggedCandidateArray.candidatesfromcounts(
      genparts.pt[mask531].count(),
      pt           = genparts.pt[mask531].flatten(),
      eta          = genparts.eta[mask531].flatten(),
      phi          = genparts.phi[mask531].flatten(),
      mass         = genparts.mass[mask531].flatten(),
      reco_idx     = gen_recoidx[mask531].flatten(),
      gen_idx      = genparts.localindex[mask531].flatten(),
      kp_idx       = bs_kp_idx[mask531].flatten(),
      km_idx       = bs_km_idx[mask531].flatten(),
      mup_idx      = bs_mup_idx[mask531].flatten(),
      mum_idx      = bs_mum_idx[mask531].flatten(),
      phi_idx      = bs_phi_idx[mask531].flatten(),
      jpsi_idx     = bs_jpsi_idx[mask531].flatten(),
      recomatch_pt = genparts.pt[mask531].ones_like().flatten() * -1,
      status       = genparts.status[mask531].flatten(),
    )
    truth_bskkmumu.add_attributes(
      y = np.log((np.sqrt(truth_bskkmumu.mass**2 
        + truth_bskkmumu.pt**2*np.cosh(truth_bskkmumu.eta)**2) 
        + truth_bskkmumu.pt*np.sinh(truth_bskkmumu.eta)) / np.sqrt(truth_bskkmumu.mass**2 
        + truth_bskkmumu.pt**2))
      )

    #print("Fraction with 2 children:")
    #print(truth_bskkmumu.pt[truth_bskkmumu.nChildren == 2].flatten().size / truth_bskkmumu.pt.flatten().size)
    #for i in range(min(truth_bskkmumu.size, 1.e6)):
    #  if truth_bskkmumu.pt[i].size >= 2:
    #    print(f"i={i}, pt={truth_bskkmumu.pt[i]}, status={truth_bskkmumu.status[i]}")

    # Truth selections
    truth_selections = {}
    truth_selections["inclusive"] = truth_bskkmumu.pt.ones_like().astype(bool)

    # Fiducial selection: match reco cuts
    truth_selections["fiducial"] =  (genparts.pt[truth_bskkmumu.gen_idx] > 3.0) \
                                    & (genparts.pt[truth_bskkmumu.kp_idx] > 0.5) & (abs(genparts.eta[truth_bskkmumu.kp_idx]) < 2.5) \
                                    & (genparts.pt[truth_bskkmumu.km_idx] > 0.5) & (abs(genparts.eta[truth_bskkmumu.km_idx]) < 2.5) \
                                    & (genparts.pt[truth_bskkmumu.mup_idx] > 1.0) & (abs(genparts.eta[truth_bskkmumu.mup_idx]) < 2.4) \
                                    & (genparts.pt[truth_bskkmumu.mum_idx] > 1.0) & (abs(genparts.eta[truth_bskkmumu.mum_idx]) < 2.4) \

    truth_selections["matched"] = (truth_bskkmumu.reco_idx >= 0)
    truth_selections["unmatched"] = ~truth_selections["matched"]
    truth_selections["matched_sel"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
    truth_selections["matched_sel"][truth_selections["matched"]] = selections["reco"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]

    for trigger in self._triggers:
      truth_selections[f"matched_tag_{trigger}"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
      truth_selections[f"matched_tag_{trigger}"][truth_selections["matched"]] = selections[f"tag_{trigger}"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]
      
      truth_selections[f"matched_probe_{trigger}"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
      truth_selections[f"matched_probe_{trigger}"][truth_selections["matched"]] = selections[f"probe_{trigger}"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]

    truth_bskkmumu.recomatch_pt[truth_selections["matched"]] = reco_bskkmumu.fit_pt[truth_bskkmumu.reco_idx[truth_selections["matched"]]]

    # Truth "cutflow"
    truth_selection_names = ["inclusive", "fiducial", "matched", "unmatched", "matched_sel"]
    for trigger in self._triggers:
      truth_selection_names.extend([f"matched_tag_{trigger}", f"matched_probe_{trigger}"])
    for selection_name in truth_selection_names:
      output["truth_cutflow"][dataset_name][selection_name] = truth_selections[selection_name].sum().sum()

    # Probe filter (pythia)
    # probefilter=cms.EDFilter("PythiaProbeFilter",  # bachelor muon with kinematic cuts.
    #   MaxEta = cms.untracked.double(2.5),
    #   MinEta = cms.untracked.double(-2.5),
    #   MinPt = cms.untracked.double(5.), # third Mu with Pt > 5
    #   ParticleID = cms.untracked.int32(13),
    #   MomID=cms.untracked.int32(443),
    #   GrandMomID = cms.untracked.int32(521),
    #   NumberOfSisters= cms.untracked.int32(1),
    #   NumberOfAunts= cms.untracked.int32(1),
    #   SisterIDs=cms.untracked.vint32(-13),
    #   AuntIDs=cms.untracked.vint32(321),)
    truth_muons = genparts[abs(genparts.pdgId) == 13]
    truth_muons_probefilter = (abs(truth_muons.eta) <= 2.5) \
                              & (truth_muons.pt >= 5.0) \
                              & ~(
                                  (abs(genparts.pdgId[truth_muons.genPartIdxMother]) == 443) \
                                  & (abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons.genPartIdxMother]]) == 531) \
                                  & (bs_phi_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (bs_kp_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (bs_km_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (nChildrenNoPhoton[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] == 2) \
                                  )
    #event_probefilter = (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3)
    #for x in zip(truth_muons[~event_probefilter].pt.count(), truth_muons[~event_probefilter].pdgId, truth_muons[~event_probefilter].pt, truth_muons[~event_probefilter].eta, genparts[~event_probefilter].pdgId[truth_muons[~event_probefilter].genPartIdxMother]):
    #  print("Muon info in event failing probefilter:")
    #  print(x)
    truth_selections["probefilter"] = (truth_bskkmumu.pt.ones_like() * ( \
                                        (truth_muons_probefilter.sum() >= 1) \
                                      )).astype(bool) # & (truth_muons.pt.count() >= 3)

    # Print out events failing the probefilter closure test
    if False:
      if "probefilter" in dataset_name:
        print("Printing probefilter failures")
        counter = 0
        fail_eventmask = (truth_selections["probefilter"].sum() == 0)
        print(truth_selections["probefilter"])
        print(fail_eventmask)
        for ievt in range(genparts.pt.size):
          if fail_eventmask[ievt]:
            print("Event")
            print(truth_selections["probefilter"][ievt])
            print(truth_selections["probefilter"].sum()[ievt])
            print(truth_muons_probefilter[ievt])
            for ipart in range(genparts.pt[ievt].size):
              print(f"i={ipart}, pdgId={genparts.pdgId[ievt][ipart]}, pt={genparts.pt[ievt][ipart]}, eta={genparts.eta[ievt][ipart]}, phi={genparts.phi[ievt][ipart]}, status={genparts.status[ievt][ipart]}, \
  iparent={genparts.genPartIdxMother[ievt][ipart]}, \
  parent={genparts.pdgId[ievt][genparts.genPartIdxMother[ievt][ipart]]}, \
  igrandparent={genparts.genPartIdxMother[ievt][genparts.genPartIdxMother[ievt][ipart]]}, \
  grandparent={genparts.pdgId[ievt][genparts.genPartIdxMother[ievt][genparts.genPartIdxMother[ievt][ipart]]]}")

              if abs(genparts.pdgId[ievt][ipart]) == 13:
                print(f"Muon pt={genparts.pt[ievt][ipart]}, eta={genparts.eta[ievt][ipart]}, mother={genparts.pdgId[ievt][genparts.genPartIdxMother[ievt][ipart]]}")
            counter += 1
            if counter > 5:
              break

    self._accumulator["TruthProbeMuon_parent"].fill(
      dataset=dataset_name, 
      parentPdgId=abs(genparts.pdgId[truth_muons[truth_muons_probefilter].genPartIdxMother].flatten())
    )
    self._accumulator["TruthProbeMuon_grandparent"].fill(
      dataset=dataset_name, 
      grandparentPdgId=abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons[truth_muons_probefilter].genPartIdxMother]].flatten())
    )

    # Fill truth histograms
    for selection_name in truth_selection_names + ["probefilter"]:
      output["TruthBsToKKMuMu_pt_absy_mass"].fill(dataset=dataset_name, 
        selection=selection_name, 
        pt=truth_bskkmumu.pt[truth_selections[selection_name]].flatten(),
        absy=np.abs(truth_bskkmumu.y[truth_selections[selection_name]].flatten()),
        mass=truth_bskkmumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_pt"].fill(dataset=dataset_name, selection=selection_name, pt=truth_bskkmumu.pt[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_eta"].fill(dataset=dataset_name, selection=selection_name, eta=truth_bskkmumu.eta[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_y"].fill(dataset=dataset_name, selection=selection_name, y=truth_bskkmumu.y[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_phi"].fill(dataset=dataset_name, selection=selection_name, phi=truth_bskkmumu.phi[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_mass"].fill(dataset=dataset_name, selection=selection_name, mass=truth_bskkmumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_recopt_d_truthpt"].fill(dataset=dataset_name, selection=selection_name, 
          recopt_d_truthpt=where(truth_selections["matched"].flatten(), 
                                  ((truth_bskkmumu.recomatch_pt / truth_bskkmumu.pt)).flatten(),
                                  -1.0)[truth_selections[selection_name].flatten()])
      output["TruthBsToKKMuMu_truthpt_recopt"].fill(dataset=dataset_name, selection=selection_name,
          reco_pt=truth_bskkmumu.recomatch_pt[truth_selections[selection_name]].flatten(), 
          truth_pt=truth_bskkmumu.pt[truth_selections[selection_name]].flatten())

      output["TruthBsToKKMuMu_kp_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.kp_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.kp_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.kp_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.kp_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBsToKKMuMu_km_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.km_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.km_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.km_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.km_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBsToKKMuMu_mup_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.mup_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBsToKKMuMu_mum_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.mum_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBsToKKMuMu_phi_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.phi_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.phi_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.phi_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.phi_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBsToKKMuMu_jpsi_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bskkmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bskkmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bskkmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bskkmumu.jpsi_idx[truth_selections[selection_name]]].flatten()
                                          )
      #output["TruthBsToKKMuMu_dR_kp_km"].fill(dataset=dataset_name, selection=selection_name, 
      #                                      dR=genparts.p4[truth_bskkmumu.kp_idx].delta_r(genparts.p4[truth_bskkmumu.km_idx]).flatten())


    output["nTruthMuon"].fill(dataset=dataset_name, nTruthMuon=genparts[abs(genparts.pdgId)==13].pt.count())

    # N-1 histograms
    all_cuts = ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "jpsi", "phi", "kstar_veto"] # dR
    nm1_selections = {}
    for xcut in ["jpsi", "phi", "l_xy_sig", "sv_prob", "cos2D"]:
      nm1_selections[f"{xcut}_tag"] = reco_bskkmumu_mask_template
      for cut in all_cuts:
        if not cut == xcut:
          nm1_selections[f"{xcut}_tag"] = nm1_selections[f"{xcut}_tag"] & selections[cut]
      nm1_selections[f"{xcut}_probe"] = copy.deepcopy(nm1_selections[f"{xcut}_tag"])

      nm1_selections[f"{xcut}_tag"] = nm1_selections[f"{xcut}_tag"] & (getattr(reco_bskkmumu, f"Muon1IsTrig_HLT_Mu7_IP4") | getattr(reco_bskkmumu, f"Muon2IsTrig_HLT_Mu7_IP4"))
      nm1_selections[f"{xcut}_tag"] = nm1_selections[f"{xcut}_tag"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[nm1_selections[f"{xcut}_tag"]].min())

      nm1_selections[f"{xcut}_probe"] = nm1_selections[f"{xcut}_probe"] & (getattr(reco_bskkmumu, "TagCount_HLT_Mu7_IP4") >= 1)
      nm1_selections[f"{xcut}_probe"] = nm1_selections[f"{xcut}_probe"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[nm1_selections[f"{xcut}_tag"]].min())
    for selection in ["tag", "probe"]:
      output["NM1_BsToKKMuMu_jpsi_m"].fill(dataset=dataset_name, selection="f{selection}", mass=reco_bskkmumu.mll_fullfit[nm1_selections[f"jpsi_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_phi_m"].fill(dataset = dataset_name, selection="f{selection}", mass=reco_bskkmumu.phi_m[nm1_selections[f"phi_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection="f{selection}", l_xy_sig=reco_bskkmumu.l_xy_sig[nm1_selections[f"l_xy_sig_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_sv_prob"].fill(dataset=dataset_name, selection="f{selection}", sv_prob=reco_bskkmumu.sv_prob[nm1_selections[f"sv_prob_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_cos2D"].fill(dataset=dataset_name, selection="f{selection}", cos2D=reco_bskkmumu.fit_cos2D[nm1_selections[f"cos2D_{selection}"]].flatten())


    # Tree outputs
    for selection_name in self._Bcand_selections:
      output[f"Bcands_{selection_name}"][dataset_name].extend({
        "pt": reco_bskkmumu.fit_pt[selections[selection_name]].flatten(),
        "eta": reco_bskkmumu.fit_eta[selections[selection_name]].flatten(),
        "y": reco_bskkmumu.fit_y[selections[selection_name]].flatten(),
        "phi": reco_bskkmumu.fit_phi[selections[selection_name]].flatten(),
        "mass": reco_bskkmumu.fit_mass[selections[selection_name]].flatten(),
        #"l_xy": reco_bskkmumu.l_xy[selections[selection_name]].flatten(),
        #"l_xy_unc": reco_bskkmumu.l_xy_unc[selections[selection_name]].flatten(),
        #"sv_prob": reco_bskkmumu.sv_prob[selections[selection_name]].flatten(),
        #"cos2D": reco_bskkmumu.fit_cos2D[selections[selection_name]].flatten(),
      })

    optz_selection = (reco_bskkmumu.fit_pt > 5.0) & (reco_bskkmumu.fit_pt < 25.0) \
                                & (reco_bskkmumu.sv_prob > 0.01) \
                                & (reco_bskkmumu.l_xy_sig > 1.5) \
                                & (reco_bskkmumu.cos2D > 0.95) \
                                & (np.abs(reco_bskkmumu.phi_m - PHI_1020_MASS) < 0.05) \
                                & (np.abs(reco_bskkmumu.mll_fullfit - JPSI_1S_MASS) < 0.5) \
                                & (reco_bskkmumu.l1_pt > 1.0) \
                                & (reco_bskkmumu.l2_pt > 1.0) \
                                & (reco_bskkmumu.trk1_pt > 0.5) \
                                & (reco_bskkmumu.trk2_pt > 0.5) \
                                & selections["truthmatched"]
    output[f"Bcands_Bs_opt"][dataset_name].extend({
      "pt": reco_bskkmumu.fit_pt[optz_selection].flatten(),
      "eta": reco_bskkmumu.fit_eta[optz_selection].flatten(),
      "y": reco_bskkmumu.fit_y[optz_selection].flatten(),
      "phi": reco_bskkmumu.fit_phi[optz_selection].flatten(),
      "mass": reco_bskkmumu.fit_mass[optz_selection].flatten(),
      "l_xy": reco_bskkmumu.l_xy[optz_selection].flatten(),
      "l_xy_sig": reco_bskkmumu.l_xy_sig[optz_selection].flatten(),
      "sv_prob": reco_bskkmumu.sv_prob[optz_selection].flatten(),
      "cos2D": reco_bskkmumu.fit_cos2D[optz_selection].flatten(),
      "dm_phi": np.abs(reco_bskkmumu.phi_m[optz_selection].flatten() - PHI_1020_MASS),
      "dm_jpsi": np.abs(reco_bskkmumu.mll_fullfit[optz_selection].flatten() - JPSI_1S_MASS),
      "l1_pt": reco_bskkmumu.l1_pt[optz_selection].flatten(),
      "l2_pt": reco_bskkmumu.l2_pt[optz_selection].flatten(),
      "k1_pt": reco_bskkmumu.trk1_pt[optz_selection].flatten(),
      "k2_pt": reco_bskkmumu.trk2_pt[optz_selection].flatten(),
      "l1_eta": reco_bskkmumu.l1_pt[optz_selection].flatten(),
      "l2_eta": reco_bskkmumu.l2_pt[optz_selection].flatten(),
      "k1_eta": reco_bskkmumu.trk1_pt[optz_selection].flatten(),
      "k2_eta": reco_bskkmumu.trk2_pt[optz_selection].flatten(),
      "HLT_Mu7_IP4":  ((df["HLT_Mu7_IP4"]  == 1) * reco_bskkmumu_mask_template[optz_selection]).flatten().astype(int),
      "HLT_Mu9_IP5":  ((df["HLT_Mu9_IP5"]  == 1) * reco_bskkmumu_mask_template[optz_selection]).flatten().astype(int),
      "HLT_Mu9_IP6":  ((df["HLT_Mu9_IP6"]  == 1) * reco_bskkmumu_mask_template[optz_selection]).flatten().astype(int),
      "HLT_Mu12_IP6": ((df["HLT_Mu12_IP6"] == 1) * reco_bskkmumu_mask_template[optz_selection]).flatten().astype(int),
    })

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  # Inputs
  in_txt = {
      #"Bs2PhiJpsi2KKMuMu_probefilter_noconstr": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v1_7/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      #"Bs2PhiJpsi2KKMuMu_inclusive_noconstr": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v1_6/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      #"Bs2PhiJpsi2KKMuMu_probefilter_lzma6": "/home/dryu/BFrag/boffea/barista/filelists/v1_6/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen_lzma.txt",
      #"Bs2PhiJpsi2KKMuMu_probefilter_zlib6": "/home/dryu/BFrag/boffea/barista/filelists/v1_6/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen_zlib.txt",
      #"Bs2PhiJpsi2KKMuMu_probefilter_veryloose": "/home/dryu/BFrag/boffea/barista/filelists/v1_5_veryloose/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
      #"Bs2PhiJpsi2KKMuMu_probefilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v2_6/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      #"Bs2PhiJpsi2KKMuMu_inclusive": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v2_6/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      "Bs2PhiJpsi2KKMuMu_probefilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      "Bs2PhiJpsi2KKMuMu_inclusive": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      "Bs2PhiJpsi2KKMuMu_mufilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToPhiJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
  }
  dataset_files = {}
  for dataset_name, filelistpath in in_txt.items():
    with open(filelistpath, 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]
  ts_start = time.time()
  print(dataset_files)
  output = processor.run_uproot_job(dataset_files,
                                treename='Events',
                                processor_instance=MCEfficencyProcessor(),
                                executor=processor.futures_executor,
                                executor_args={'workers': 12, 'flatten': False},
                                chunksize=50000,
                                # maxchunks=1,
                            )
  ts_end = time.time()
  total_events = 0
  dataset_nevents = {}
  for k, v in output['nevents'].items():
    if k in dataset_nevents:
      dataset_nevents[k] += v
    else:
      dataset_nevents[k] = v
    total_events += v

  print("Reco cutflow:")
  for dataset, d1 in output["reco_cutflow"].items():
    print(f"\tDataset={dataset}")
    print(f"\t\tnevents => {dataset_nevents[dataset]}")
    for cut_name, cut_npass in d1.items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / dataset_nevents[dataset]}")

  print("Reco cutflow, 5<pT<10:")
  for dataset, d1 in output["reco_cutflow_lowpt"].items():
    print(f"\tDataset={dataset}")
    print(f"\t\tnevents => {dataset_nevents[dataset]}")
    for cut_name, cut_npass in d1.items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / dataset_nevents[dataset]}")

  print("Truth cutflow:")
  for dataset, d1 in output["truth_cutflow"].items():
    print(f"\tDataset={dataset}")
    print(f"\t\tnevents => {dataset_nevents[dataset]}")
    for cut_name, cut_npass in d1.items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / d1['inclusive']}")
  util.save(output, f"MCEfficiencyHistograms.coffea")

  print("Total nevents: {}".format(total_events))
  print("Total time: {} seconds".format(ts_end - ts_start))
  print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))
