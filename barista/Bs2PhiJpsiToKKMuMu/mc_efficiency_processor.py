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

np.set_printoptions(threshold=np.inf)

def where(predicate, iftrue, iffalse):
    predicate = predicate.astype(np.bool)   # just to make sure they're 0/1
    return predicate*iftrue + (1 - predicate)*iffalse

class MCEfficencyProcessor(processor.ProcessorABC):
  def __init__(self):
    #self._trigger = "HLT_Mu9_IP6"

    # Histograms
    dataset_axis = hist.Cat("dataset", "Primary dataset")
    selection_axis_reco = hist.Cat("selection", "Selection name (reco)")
    selection_axis_truth = hist.Cat("selection", "Selection name (truth)")

    self._accumulator = processor.dict_accumulator()
    self._accumulator["nevents"] = processor.defaultdict_accumulator(int)
    self._accumulator["reco_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    self._accumulator["truth_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    for selection_name in ["tagmatch", "probematch"]:
      self._accumulator[f"Bcands_{selection_name}"] = processor.defaultdict_accumulator(Bcand_accumulator)

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BsToKKMuMu_fit_pt_y_mass"] = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0),
                                                            hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1)
                                                          )
    self._accumulator["BsToKKMuMu_fit_pt"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_eta"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BsToKKMuMu_fit_phi"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BsToKKMuMu_fit_mass"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1))
    self._accumulator["BsToKKMuMu_chi2"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_cos2D"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BsToKKMuMu_fit_theta2D"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, 0., math.pi))
    self._accumulator["BsToKKMuMu_l_xy"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BsToKKMuMu_l_xy_sig"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$", 50, -1.0, 4.0))
    self._accumulator["BsToKKMuMu_jpsi_mass"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["BsToKKMuMu_phi_mass"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(\phi)$ [GeV]", 100, PHI_1020_MASS * 0.8, PHI_1020_MASS * 1.2))

    self._accumulator["Bs_cos2D_vs_pt"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("log1mcos2D", r"log(1-cos2D)", 40, -4, 0.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_svprob_vs_pt"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("sv_prob", r"SV prob", 100, 0., 1.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_lxysig_vs_pt"]  = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$", 60, -1., 11.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_phimass_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("phi_m", r"$m(\phi)$ [GeV]",  50, PHI_1020_MASS*0.8, PHI_1020_MASS*1.2), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_phipt_vs_pt"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("phi_pt", r"$p_{T}(\phi)$ [GeV]",  50, 0., 25.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_k1pt_vs_pt"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("k1_pt", r"$p_{T}(k1)$ [GeV]", 60, 0., 15.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_k2pt_vs_pt"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("k2_pt", r"$p_{T}(k2)$ [GeV]", 60, 0., 15.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_l1pt_vs_l1eta_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l1_pt", r"$p_{T}(l1)$ [GeV]", 50, 0., 25.), hist.Bin("l1_eta", r"$\eta(l1)$ [GeV]", 30, 0., 3.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))
    self._accumulator["Bs_l2pt_vs_l2eta_vs_pt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l2_pt", r"$p_{T}(l2)$ [GeV]", 50, 0., 25.), hist.Bin("l2_eta", r"$\eta(l2)$ [GeV]", 30, 0., 3.), hist.Bin("pt", r"$p_{T}^{(fit)}$ [GeV]", 20, 0., 50.))

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

    self._accumulator["TruthBsToKKMuMu_pt"]               = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBsToKKMuMu_eta"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("eta", r"$\eta$", 50, -5.0, 5.0))
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
    reco_bskkmumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggering[reco_bskkmumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggering[reco_bskkmumu.l2_idx])
    reco_bskkmumu.add_attributes(MuonIsTrigCount = reco_bskkmumu.Muon1IsTrig.astype(int) + reco_bskkmumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bskkmumu.add_attributes(TagCount = reco_bskkmumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bskkmumu.MuonIsTrigCount)

    reco_bskkmumu.add_attributes(l_xy_sig = where(reco_bskkmumu.l_xy_unc > 0, reco_bskkmumu.l_xy / reco_bskkmumu.l_xy_unc, -1.e20))

    # General selection
    reco_bskkmumu_mask_template = reco_bskkmumu.pt.ones_like().astype(bool)
    selections = {}
    selections["sv_pt"]    = (reco_bskkmumu.pt > 3.0)
    selections["l_xy_sig"] = (abs(reco_bskkmumu.l_xy_sig) > 3.5)
    selections["sv_prob"]  = (reco_bskkmumu.sv_prob > 0.1)
    selections["cos2D"]    = (reco_bskkmumu.fit_cos2D > 0.999)
    selections["l1"]       = (reco_bskkmumu.l1_pt > 1.5) & (abs(reco_bskkmumu.l1_eta) < 2.4)
    selections["l2"]       = (abs(reco_bskkmumu.l2_eta) < 2.4)
    selections["l2"]       = (selections["l2"] & where(abs(reco_bskkmumu.l2_eta) < 1.4, 
                                                      (reco_bskkmumu.l2_pt > 1.5), 
                                                      (reco_bskkmumu.l2_pt > 1.0))).astype(bool)
    selections["k1"]       = (reco_bskkmumu.trk1_pt > 0.5) & (abs(reco_bskkmumu.trk1_eta) < 2.5)
    selections["k2"]       = (reco_bskkmumu.trk2_pt > 0.5) & (abs(reco_bskkmumu.trk2_eta) < 2.5)
    selections["dR"]       = (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.trk2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.trk2_phi) > 0.03) \
                            & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l1_phi) > 0.03) \
                            & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l2_phi) > 0.03) \
                            & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l1_phi) > 0.03) \
                            & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l2_phi) > 0.03) \
                            & (delta_r(reco_bskkmumu.l1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.l1_phi, reco_bskkmumu.l2_phi) > 0.03)
    selections["jpsi"]       = abs(reco_bskkmumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    #(JPSI_1S_MASS - JPSI_WINDOW < reco_bskkmumu.mll_fullfit) & (reco_bskkmumu.mll_fullfit < JPSI_1S_MASS + JPSI_WINDOW)
    selections["phi"]        = (abs(reco_bskkmumu.phi_m - PHI_1020_MASS) < PHI_WINDOW) # & (reco_bskkmumu.phi_pt > 1.0)
    #(PHI_1020_MASS - PHI_WINDOW < reco_bskkmumu.phi_m) & (reco_bskkmumu.phi_m < PHI_1020_MASS + PHI_WINDOW)
    selections["kstar_veto"] = (abs(reco_bskkmumu.Kstar1_mass - KSTAR_892_MASS) > BS_KSTAR_VETO_WINDOW) \
                                & (abs(reco_bskkmumu.Kstar2_mass - KSTAR_892_MASS) > BS_KSTAR_VETO_WINDOW)
    #(reco_bskkmumu.phi_m < KSTAR_892_MASS - KSTAR_WINDOW) | (KSTAR_892_MASS + KSTAR_WINDOW < reco_bskkmumu.phi_m)

    selections["trigger"] = ((df["HLT_Mu7_IP4"] == 1) |(df["HLT_Mu9_IP5"] == 1) | (df["HLT_Mu9_IP6"] == 1)) * reco_bskkmumu_mask_template # Shape = event!

    # Final selections
    selections["inclusive"] = reco_bskkmumu.fit_pt.ones_like().astype(bool)
    selections["reco"]      = selections["trigger"] \
                              & selections["sv_pt"] \
                              & selections["l_xy_sig"] \
                              & selections["sv_prob"] \
                              & selections["cos2D"] \
                              & selections["l1"] \
                              & selections["l2"] \
                              & selections["k1"] \
                              & selections["k2"] \
                              & selections["dR"] \
                              & selections["jpsi"] \
                              & selections["phi"] \
                              & selections["kstar_veto"]
    selections["truthmatched"] = (reco_bskkmumu.genPartIdx >= 0)
    selections["tag"]            = selections["reco"] & (reco_bskkmumu.Muon1IsTrig | reco_bskkmumu.Muon2IsTrig)
    selections["tagmatch"]       = selections["tag"] & selections["truthmatched"]
    selections["tagunmatched"]   = selections["tag"] & (~selections["truthmatched"])
    selections["probe"]          = selections["reco"] & (reco_bskkmumu.TagCount >= 1)
    selections["probematch"]     = selections["probe"] & selections["truthmatched"]
    selections["probeunmatched"] = selections["probe"] & (~selections["truthmatched"])

    # If more than one B is selected, choose best chi2
    selections["tag"]            = selections["tag"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["tag"]].min())
    selections["tagmatch"]       = selections["tagmatch"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["tagmatch"]].min())
    selections["tagunmatched"]   = selections["tagunmatched"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["tagunmatched"]].min())
    selections["probe"]          = selections["probe"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["probe"]].min())
    selections["probematch"]     = selections["probematch"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["probematch"]].min())
    selections["probeunmatched"] = selections["probeunmatched"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["probeunmatched"]].min())

    # Fill cutflow
    cumulative_selection = reco_bskkmumu.pt.ones_like().astype(bool)
    output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    for cut_name in ["trigger", "sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "dR", "jpsi", "phi", "kstar_veto"]:
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow"][dataset_name]["tag"] += selections["tag"].sum().sum()
    output["reco_cutflow"][dataset_name]["probe"] += selections["probe"].sum().sum()

    # Fill reco histograms
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    for selection_name in ["inclusive", "reco", "truthmatched", "tag", "tagmatch", "probe", "probematch", "tagunmatched", "probeunmatched"]:
      output["BsToKKMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_y=reco_bskkmumu.fit_y[selections[selection_name]].flatten(),
                                            fit_mass=reco_bskkmumu.fit_mass[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bskkmumu.fit_pt[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bskkmumu.fit_eta[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bskkmumu.fit_phi[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bskkmumu.fit_mass[selections[selection_name]].flatten())
      output["BsToKKMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bskkmumu.chi2[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bskkmumu.fit_cos2D[selections[selection_name]].flatten())
      output["BsToKKMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bskkmumu.fit_cos2D[selections[selection_name]]).flatten())
      output["BsToKKMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bskkmumu.l_xy[selections[selection_name]].flatten())
      output["BsToKKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bskkmumu.l_xy_sig[selections[selection_name]].flatten())
      output["BsToKKMuMu_jpsi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.mll_fullfit[selections[selection_name]].flatten())
      output["BsToKKMuMu_phi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.phi_m[selections[selection_name]].flatten())

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

    # Jagged array of truth BToKMuMus
    mask531 = (abs(genparts.pdgId)==531)
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
    )

    # Truth selections
    truth_selections = {}
    truth_selections["inclusive"] = truth_bskkmumu.pt.ones_like().astype(bool)

    # Fiducial selection: match reco cuts
    truth_selections["fiducial"] =  (genparts.pt[truth_bskkmumu.gen_idx] > 3.0) \
                                    & (genparts.pt[truth_bskkmumu.kp_idx] > 0.5) & (abs(genparts.eta[truth_bskkmumu.kp_idx]) < 2.5) \
                                    & (genparts.pt[truth_bskkmumu.km_idx] > 0.5) & (abs(genparts.eta[truth_bskkmumu.km_idx]) < 2.5) \
                                    & (genparts.pt[truth_bskkmumu.mup_idx] > 1.5) & (abs(genparts.eta[truth_bskkmumu.mup_idx]) < 2.4) \
                                    & (genparts.pt[truth_bskkmumu.mum_idx] > 1.5) & (abs(genparts.eta[truth_bskkmumu.mum_idx]) < 2.4) \
                                    & (genparts.p4[truth_bskkmumu.kp_idx].delta_r(genparts.p4[truth_bskkmumu.km_idx]) > 0.03) \
                                    & (genparts.p4[truth_bskkmumu.kp_idx].delta_r(genparts.p4[truth_bskkmumu.mup_idx]) > 0.03) \
                                    & (genparts.p4[truth_bskkmumu.kp_idx].delta_r(genparts.p4[truth_bskkmumu.mum_idx]) > 0.03) \
                                    & (genparts.p4[truth_bskkmumu.km_idx].delta_r(genparts.p4[truth_bskkmumu.mup_idx]) > 0.03) \
                                    & (genparts.p4[truth_bskkmumu.km_idx].delta_r(genparts.p4[truth_bskkmumu.mum_idx]) > 0.03) \
                                    & (genparts.p4[truth_bskkmumu.mup_idx].delta_r(genparts.p4[truth_bskkmumu.mum_idx]) > 0.03)

    truth_selections["matched"] = (truth_bskkmumu.reco_idx >= 0)
    truth_selections["unmatched"] = ~truth_selections["matched"]
    truth_selections["matched_sel"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
    truth_selections["matched_sel"][truth_selections["matched"]] = selections["reco"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]
    truth_selections["matched_tag"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
    truth_selections["matched_tag"][truth_selections["matched"]] = selections["tag"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]
    truth_selections["matched_probe"] = truth_bskkmumu.reco_idx.zeros_like().astype(bool)
    truth_selections["matched_probe"][truth_selections["matched"]] = selections["probe"][truth_bskkmumu.reco_idx[truth_selections["matched"]]]

    truth_bskkmumu.recomatch_pt[truth_selections["matched"]] = reco_bskkmumu.fit_pt[truth_bskkmumu.reco_idx[truth_selections["matched"]]]

    # Truth "cutflow"
    for cut_name in ["inclusive", "fiducial", "matched", "unmatched", "matched_sel", "matched_tag", "matched_probe"]:
      output["truth_cutflow"][dataset_name][cut_name] = truth_selections[cut_name].sum().sum()

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
    truth_muons_probefilter = (abs(truth_muons.eta) < 2.5) \
                              & (truth_muons.pt > 5.0) \
                              & ~(
                                  (abs(genparts.pdgId[truth_muons.genPartIdxMother]) == 443) \
                                  & (abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons.genPartIdxMother]]) == 531)
                                  )
    event_probefilter = (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3)
    #for x in zip(truth_muons[~event_probefilter].pt.count(), truth_muons[~event_probefilter].pdgId, truth_muons[~event_probefilter].pt, truth_muons[~event_probefilter].eta, genparts[~event_probefilter].pdgId[truth_muons[~event_probefilter].genPartIdxMother]):
    #  print("Muon info in event failing probefilter:")
    #  print(x)
    truth_selections["probefilter"] = (truth_bskkmumu.pt.ones_like() * ( \
                                        (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3) \
                                      )).astype(bool)

    # Fill truth histograms
    for selection_name in ["inclusive", "matched", "unmatched", "matched_sel", "matched_tag", "matched_probe", "probefilter"]:
      output["TruthBsToKKMuMu_pt"].fill(dataset=dataset_name, selection=selection_name, pt=truth_bskkmumu.pt[truth_selections[selection_name]].flatten())
      output["TruthBsToKKMuMu_eta"].fill(dataset=dataset_name, selection=selection_name, eta=truth_bskkmumu.eta[truth_selections[selection_name]].flatten())
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

    # Tree outputs
    for selection_name in ["tagmatch", "probematch"]:
      output[f"Bcands_{selection_name}"][dataset_name].extend({
        "pt": reco_bskkmumu.fit_pt[selections[selection_name]].flatten(),
        "eta": reco_bskkmumu.fit_eta[selections[selection_name]].flatten(),
        "y": reco_bskkmumu.fit_y[selections[selection_name]].flatten(),
        "phi": reco_bskkmumu.fit_phi[selections[selection_name]].flatten(),
        "mass": reco_bskkmumu.fit_mass[selections[selection_name]].flatten(),
        "l_xy": reco_bskkmumu.l_xy[selections[selection_name]].flatten(),
        "l_xy_unc": reco_bskkmumu.l_xy_unc[selections[selection_name]].flatten(),
        "sv_prob": reco_bskkmumu.sv_prob[selections[selection_name]].flatten(),
        "cos2D": reco_bskkmumu.fit_cos2D[selections[selection_name]].flatten(),
      })

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  # Inputs
  in_txt = {
      #"Bs2PhiJpsi2KKMuMu_probefilter_noconstr": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v1_7/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      #"Bs2PhiJpsi2KKMuMu_inclusive_noconstr": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v1_6/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
      "Bs2PhiJpsi2KKMuMu_probefilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v2_5_4/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.dat"),
      "Bs2PhiJpsi2KKMuMu_inclusive": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/v2_5_4/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.dat"),
      #"Bs2PhiJpsi2KKMuMu_probefilter_lzma6": "/home/dryu/BFrag/boffea/barista/filelists/v1_6/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen_lzma.txt",
      #"Bs2PhiJpsi2KKMuMu_probefilter_zlib6": "/home/dryu/BFrag/boffea/barista/filelists/v1_6/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen_zlib.txt",
      #"Bs2PhiJpsi2KKMuMu_probefilter_veryloose": "/home/dryu/BFrag/boffea/barista/filelists/v1_5_veryloose/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
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
                                executor_args={'workers': 8, 'flatten': False},
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
