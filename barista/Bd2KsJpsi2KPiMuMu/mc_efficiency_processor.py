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
    self._trigger = "HLT_Mu7_IP4"

    # Histograms
    dataset_axis = hist.Cat("dataset", "Primary dataset")
    selection_axis_reco = hist.Cat("selection", "Selection name (reco)")
    selection_axis_truth = hist.Cat("selection", "Selection name (truth)")

    self._accumulator = processor.dict_accumulator()
    self._accumulator["nevents"] = processor.defaultdict_accumulator(int)
    self._accumulator["reco_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    self._accumulator["truth_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))
    
    for selection_name in ["tagmatch", "tagmatchswap", "probematch", "probematchswap"]:
      self._accumulator[f"Bcands_{selection_name}"] = processor.defaultdict_accumulator(Bcand_accumulator) #, outputfile=f"tree_{selection_name}.root"))

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BdToKPiMuMu_fit_pt_y_mass"]             = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0),
                                                            hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1)
                                                          )
    self._accumulator["BdToKPiMuMu_fit_pt"]             = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_eta"]            = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BdToKPiMuMu_fit_phi"]            = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BdToKPiMuMu_fit_best_mass"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_fit_best_barmass"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_barmass", r"Swap $m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_chi2"]               = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_cos2D"]          = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BdToKPiMuMu_fit_theta2D"]        = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, 0., math.pi))
    self._accumulator["BdToKPiMuMu_l_xy"]               = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BdToKPiMuMu_l_xy_sig"]           = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))
    self._accumulator["BdToKPiMuMu_fit_best_mkstar"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mkstar", r"$m_{K^{*}}^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_fit_best_barmkstar"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_barmkstar", r"Swap $m_{K^{*}}^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_jpsi_mass"]          = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))


    self._accumulator["nTruthMuon"]  = hist.Hist("Events", dataset_axis, hist.Bin("nTruthMuon", r"N(truth muons)", 11, -0.5, 10.5))
    self._accumulator["TruthBdToKPiMuMu_pt"]               = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBdToKPiMuMu_eta"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("eta", r"$\eta$", 50, -5.0, 5.0))
    self._accumulator["TruthBdToKPiMuMu_phi"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["TruthBdToKPiMuMu_mass"]             = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("mass", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBdToKPiMuMu_recopt_d_truthpt"] = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("recopt_d_truthpt", r"$m$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["TruthBdToKPiMuMu_k_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(K^{\pm})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(K^{\pm})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(K^{\pm})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(K^{\pm})$", 30, 0., K_MASS*2.0)
    )
    self._accumulator["TruthBdToKPiMuMu_pi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\pi^{\pm})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\pi^{\pm})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\pi^{\pm})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\pi^{\pm})$", 30, 0., PI_MASS*2.0)
    )
    self._accumulator["TruthBdToKPiMuMu_mup_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{+})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{+})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{+})$", 30, 0., MUON_MASS*2.0)
    )
    self._accumulator["TruthBdToKPiMuMu_mum_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{+})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{+})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{+})$", 30, 0., MUON_MASS*2.0)
    )
    self._accumulator["TruthBdToKPiMuMu_kstar_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\phi)$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\phi)$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\phi)$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\phi)$", 30, 0., KSTAR_892_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_jpsi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(J/\psi)$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(J/\psi)$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(J/\psi)$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(J/\psi)$", 30, 0., JPSI_1S_MASS*2.0)
                                                  )

    # One entry per truth B
    # - If truth B is not matched to reco, or if reco fails selection, fill (-1, truthpt)
    self._accumulator["TruthBdToKPiMuMu_truthpt_recopt"] = hist.Hist("Events", dataset_axis, selection_axis_truth,
                                              hist.Bin("reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 500, 0., 100.),
                                              hist.Bin("truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 500, 0., 100.))

    self._accumulator["TruthBdToKPiMuMu_k_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(K^{\pm})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(K^{\pm})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(K^{\pm})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(K^{\pm})$", 30, K_MASS*0.5, K_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_pi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\pi^{\pm})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\pi^{\pm})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\pi^{\pm})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\pi^{\pm})$", 30, PI_MASS*0.5, PI_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_mup_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{+})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{+})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{+})$", 30, MUON_MASS*0.5, MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_mum_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(\mu^{-})$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(\mu^{-})$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(\mu^{-})$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(\mu^{-})$", 30, MUON_MASS*0.5, MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_kstar_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(K^{*}(892))$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(K^{*}(892))$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(K^{*}(892))$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(K^{*}(892))$", 30, PHI_1020_MASS*0.5, PHI_1020_MASS*2.0)
                                                  )
    self._accumulator["TruthBdToKPiMuMu_jpsi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                  hist.Bin("pt", r"$p_{T}(J/\psi)$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("eta", r"$\eta(J/\psi)$", 20, -5., 5.),
                                                  hist.Bin("phi", r"$\phi(J/\psi)$", 20, -1*math.pi, math.pi),
                                                  hist.Bin("mass", r"$m(J/\psi)$", 30, JPSI_1S_MASS*0.5, JPSI_1S_MASS*2.0)
                                                  )


  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    output["nevents"][dataset_name] += df.size

    # Create jagged object arrays
    reco_bdkpimumu = dataframereader.reco_bdkpimumu(df, is_mc=True)
    reco_muons     = dataframereader.reco_muons(df, is_mc=True)
    trigger_muons  = dataframereader.trigger_muons(df, is_mc=True)
    probe_tracks   = dataframereader.probe_tracks(df, is_mc=True)
    genparts       = dataframereader.genparts(df, is_mc=True)

    # Truth matching
    reco_bdkpimumu.l1_genIdx   = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.l1_genIdx[reco_bdkpimumu.l1_idx >= 0] = reco_muons.genPartIdx[reco_bdkpimumu.l1_idx] 
    reco_bdkpimumu.l2_genIdx   = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.l2_genIdx[reco_bdkpimumu.l2_idx >= 0] = reco_muons.genPartIdx[reco_bdkpimumu.l2_idx]
    reco_bdkpimumu.trk1_genIdx = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.trk1_genIdx[reco_bdkpimumu.trk1_idx >= 0] = probe_tracks.genPartIdx[reco_bdkpimumu.trk1_idx]
    #print(reco_bdkpimumu[reco_bdkpimumu.trk2_idx.count() > 0].trk2_idx)
    reco_bdkpimumu.trk2_genIdx = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.trk2_genIdx[reco_bdkpimumu.trk2_idx >= 0] = probe_tracks.genPartIdx[reco_bdkpimumu.trk2_idx]

    reco_bdkpimumu.trk1_pdgId = where(reco_bdkpimumu.trk1_genIdx >= 0, 
                                     genparts.pdgId[reco_bdkpimumu.trk1_genIdx], 
                                     -1)
    reco_bdkpimumu.trk2_pdgId = where(reco_bdkpimumu.trk2_genIdx >= 0, 
                                     genparts.pdgId[reco_bdkpimumu.trk2_genIdx], 
                                     -1)

    reco_bdkpimumu.l1_genMotherIdx = where(reco_bdkpimumu.l1_genIdx >= 0, 
                                            genparts.genPartIdxMother[reco_bdkpimumu.l1_genIdx], 
                                            -1)
    reco_bdkpimumu.l2_genMotherIdx = where(reco_bdkpimumu.l2_genIdx >= 0, 
                                            genparts.genPartIdxMother[reco_bdkpimumu.l2_genIdx], 
                                            -1)
    reco_bdkpimumu.trk1_genMotherIdx = where(reco_bdkpimumu.trk1_genIdx >= 0, 
                                            genparts.genPartIdxMother[reco_bdkpimumu.trk1_genIdx], 
                                            -1)
    reco_bdkpimumu.trk2_genMotherIdx = where(reco_bdkpimumu.trk2_genIdx >= 0, 
                                            genparts.genPartIdxMother[reco_bdkpimumu.trk2_genIdx], 
                                            -1)

    reco_bdkpimumu.l1_genGrandmotherIdx = where(reco_bdkpimumu.l1_genMotherIdx >= 0, 
                                                genparts.genPartIdxMother[reco_bdkpimumu.l1_genMotherIdx], 
                                                -1)
    reco_bdkpimumu.l2_genGrandmotherIdx = where(reco_bdkpimumu.l2_genMotherIdx >= 0, 
                                                genparts.genPartIdxMother[reco_bdkpimumu.l2_genMotherIdx], 
                                                -1)
    reco_bdkpimumu.trk1_genGrandmotherIdx = where(reco_bdkpimumu.trk1_genMotherIdx >= 0, 
                                                  genparts.genPartIdxMother[reco_bdkpimumu.trk1_genMotherIdx], 
                                                  -1)
    reco_bdkpimumu.trk2_genGrandmotherIdx = where(reco_bdkpimumu.trk2_genMotherIdx >= 0, 
                                                  genparts.genPartIdxMother[reco_bdkpimumu.trk2_genMotherIdx], 
                                                  -1)

    reco_bdkpimumu.l1_genMotherPdgId = where(reco_bdkpimumu.l1_genMotherIdx >= 0, 
                                              genparts.pdgId[reco_bdkpimumu.l1_genMotherIdx],
                                              -1)
    reco_bdkpimumu.l2_genMotherPdgId = where(reco_bdkpimumu.l2_genMotherIdx >= 0, 
                                              genparts.pdgId[reco_bdkpimumu.l2_genMotherIdx],
                                              -1)
    reco_bdkpimumu.trk1_genMotherPdgId = where(reco_bdkpimumu.trk1_genMotherIdx >= 0, 
                                                genparts.pdgId[reco_bdkpimumu.trk1_genMotherIdx],
                                                -1)
    reco_bdkpimumu.trk2_genMotherPdgId = where(reco_bdkpimumu.trk2_genMotherIdx >= 0, 
                                                genparts.pdgId[reco_bdkpimumu.trk2_genMotherIdx],
                                                -1)

    reco_bdkpimumu.l1_genGrandmotherPdgId = where(reco_bdkpimumu.l1_genGrandmotherIdx >= 0, 
                                                  genparts.pdgId[reco_bdkpimumu.l1_genGrandmotherIdx],
                                                  -1)
    reco_bdkpimumu.l2_genGrandmotherPdgId = where(reco_bdkpimumu.l2_genGrandmotherIdx >= 0, 
                                                  genparts.pdgId[reco_bdkpimumu.l2_genGrandmotherIdx],
                                                  -1)
    reco_bdkpimumu.trk1_genGrandmotherPdgId = where(reco_bdkpimumu.trk1_genGrandmotherIdx >= 0, 
                                                  genparts.pdgId[reco_bdkpimumu.trk1_genGrandmotherIdx],
                                                  -1)
    reco_bdkpimumu.trk2_genGrandmotherPdgId = where(reco_bdkpimumu.trk2_genGrandmotherIdx >= 0, 
                                                  genparts.pdgId[reco_bdkpimumu.trk2_genGrandmotherIdx],
                                                  -1)

    # MC matching assuming trk1 = K, trk2 = pi
    mcmatch_hypo1 = (abs(reco_bdkpimumu.l1_genMotherPdgId) == 443) \
                        & (abs(reco_bdkpimumu.l2_genMotherPdgId) == 443) \
                        & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                        & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                        & where(reco_bdkpimumu.nominal_kpi, 
                            (abs(reco_bdkpimumu.trk1_pdgId) == 321) & (abs(reco_bdkpimumu.trk2_pdgId) == 211),
                            (abs(reco_bdkpimumu.trk2_pdgId) == 321) & (abs(reco_bdkpimumu.trk1_pdgId) == 211),
                          ) \
                        & (abs(reco_bdkpimumu.trk1_genMotherPdgId) == 313) \
                        & (abs(reco_bdkpimumu.trk2_genMotherPdgId) == 313) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.l2_genGrandmotherIdx) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk1_genGrandmotherIdx) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk2_genGrandmotherIdx) 
    # MC matching assuming trk1=pi, trk2=K
    mcmatch_hypo2 = (abs(reco_bdkpimumu.l1_genMotherPdgId) == 443) \
                       & (abs(reco_bdkpimumu.l2_genMotherPdgId) == 443) \
                       & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                       & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                       & where(reco_bdkpimumu.nominal_kpi, 
                           (abs(reco_bdkpimumu.trk1_pdgId) == 211) & (abs(reco_bdkpimumu.trk2_pdgId) == 321),
                           (abs(reco_bdkpimumu.trk2_pdgId) == 211) & (abs(reco_bdkpimumu.trk1_pdgId) == 321),
                         ) \
                       & (abs(reco_bdkpimumu.trk1_genMotherPdgId) == 313) \
                       & (abs(reco_bdkpimumu.trk2_genMotherPdgId) == 313) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.l2_genGrandmotherIdx) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk1_genGrandmotherIdx) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk2_genGrandmotherIdx) 

    reco_bdkpimumu.mcmatch = where(reco_bdkpimumu.nominal_kpi, mcmatch_hypo1, mcmatch_hypo2).astype(np.bool)
    reco_bdkpimumu.mcmatch_swap = where(reco_bdkpimumu.nominal_kpi, mcmatch_hypo2, mcmatch_hypo1).astype(np.bool)

    reco_bdkpimumu.genPartIdx = where((reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap), reco_bdkpimumu.l1_genGrandmotherIdx, -1)

    # Tag/probe selection
    reco_bdkpimumu.add_attributes(Muon1IsTrig = reco_muons.isTriggering[reco_bdkpimumu.l1_idx], 
                                  Muon2IsTrig = reco_muons.isTriggering[reco_bdkpimumu.l2_idx])
    reco_bdkpimumu.add_attributes(MuonIsTrigCount = reco_bdkpimumu.Muon1IsTrig.astype(int) + reco_bdkpimumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bdkpimumu.add_attributes(TagCount = reco_bdkpimumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bdkpimumu.MuonIsTrigCount)

    reco_bdkpimumu.add_attributes(l_xy_sig = where(reco_bdkpimumu.l_xy_unc > 0, reco_bdkpimumu.l_xy / reco_bdkpimumu.l_xy_unc, -1.e20))

    # General selection
    reco_bdkpimumu_mask_template = reco_bdkpimumu.pt.ones_like().astype(bool)
    selections = {}
    selections["sv_pt"]    = (reco_bdkpimumu.fit_pt > 3.0)
    selections["l_xy_sig"] = (abs(reco_bdkpimumu.l_xy_sig) > 3.5)
    selections["sv_prob"]  = (reco_bdkpimumu.sv_prob > 0.1)
    selections["cos2D"]    = (reco_bdkpimumu.fit_cos2D > 0.999)
    selections["l1"]       = (reco_bdkpimumu.lep1pt_fullfit > 1.5) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4)
    selections["l2"]       = (reco_bdkpimumu.lep2pt_fullfit > 1.5) & (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4)
    selections["trk1"]     = (reco_bdkpimumu.trk1pt_fullfit > 0.5) & (abs(reco_bdkpimumu.trk1eta_fullfit) < 2.5)
    selections["trk2"]     = (reco_bdkpimumu.trk2pt_fullfit > 0.5) & (abs(reco_bdkpimumu.trk2eta_fullfit) < 2.5)
    selections["dR"]       = (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.trk2phi_fullfit) > 0.03) \
                                & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
                                & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
                                & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
                                & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
                                & (delta_r(reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.lep1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03)
    selections["jpsi"]     = abs(reco_bdkpimumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    selections["kstar"]    = abs(reco_bdkpimumu.mkstar_best_fullfit - KSTAR_892_MASS) < KSTAR_WINDOW
    selections["phi_veto"] = (abs(reco_bdkpimumu.mkstar_best_fullfit - PHI_1020_MASS) > B0_PHI_VETO_WINDOW) \
                              & (abs(reco_bdkpimumu.barmkstar_best_fullfit - PHI_1020_MASS) > B0_PHI_VETO_WINDOW)
    selections["trigger"] = (df[self._trigger] == 1) # Shape = event!

    # Final selections
    selections["inclusive"]  = reco_bdkpimumu.fit_pt.ones_like().astype(bool)
    selections["reco"]       = selections["trigger"] \
                              & selections["sv_pt"] \
                              & selections["l_xy_sig"] \
                              & selections["sv_prob"] \
                              & selections["cos2D"] \
                              & selections["l1"] \
                              & selections["l2"] \
                              & selections["trk1"] \
                              & selections["trk2"] \
                              & selections["dR"] \
                              & selections["jpsi"] \
                              & selections["kstar"] \
                              & selections["phi_veto"]
    selections["tag"]            = selections["reco"] & (reco_bdkpimumu.Muon1IsTrig | reco_bdkpimumu.Muon2IsTrig)
    selections["tagmatch"]       = selections["tag"] & (reco_bdkpimumu.mcmatch)
    selections["tagmatchswap"]   = selections["tag"] & (reco_bdkpimumu.mcmatch_swap)
    selections["tagunmatched"]   = selections["tag"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))
    selections["probe"]          = selections["reco"] & (reco_bdkpimumu.TagCount >= 1)
    selections["probematch"]     = selections["probe"] & (reco_bdkpimumu.mcmatch)
    selections["probematchswap"] = selections["probe"] & (reco_bdkpimumu.mcmatch_swap)
    selections["probeunmatched"]   = selections["probe"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

    # If more than one B is selected, choose best chi2
    selections["tag"]            = selections["tag"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["tag"]].chi2.min())
    selections["tagmatch"]       = selections["tagmatch"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["tagmatch"]].chi2.min())
    selections["tagmatchswap"]   = selections["tagmatchswap"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["tagmatchswap"]].chi2.min())
    selections["tagunmatched"]   = selections["tagunmatched"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["tagunmatched"]].chi2.min())
    selections["probe"]          = selections["probe"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["probe"]].chi2.min())
    selections["probematch"]     = selections["probematch"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["probematch"]].chi2.min())
    selections["probematchswap"] = selections["probematchswap"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["probematchswap"]].chi2.min())
    selections["probeunmatched"]   = selections["probeunmatched"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections["probeunmatched"]].chi2.min())

    # Fill cutflow
    cumulative_selection = reco_bdkpimumu.pt.ones_like().astype(bool)
    output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    for cut_name in ["trigger", "sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "trk1", "trk2", "dR", "jpsi", "kstar", "phi_veto"]:
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow"][dataset_name]["tag"] += selections["tag"].sum().sum()
    output["reco_cutflow"][dataset_name]["tagmatch"] += selections["tagmatch"].sum().sum()
    output["reco_cutflow"][dataset_name]["tagmatchswap"] += selections["tagmatchswap"].sum().sum()
    output["reco_cutflow"][dataset_name]["probe"] += selections["probe"].sum().sum()
    output["reco_cutflow"][dataset_name]["probematch"] += selections["probematch"].sum().sum()
    output["reco_cutflow"][dataset_name]["probematchswap"] += selections["probematchswap"].sum().sum()

    # Fill reco histograms
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    for selection_name in ["inclusive", "reco", "tag", "tagmatch", "tagmatchswap", "probe", "probematch", "probematchswap", "tagunmatched", "probeunmatched"]:
      output["BdToKPiMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bdkpimumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_y=reco_bdkpimumu.fit_y[selections[selection_name]].flatten(),
                                            fit_mass=reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bdkpimumu.fit_pt[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bdkpimumu.fit_eta[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bdkpimumu.fit_phi[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_barmass"].fill(dataset=dataset_name, selection=selection_name, fit_barmass=reco_bdkpimumu.fit_best_barmass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bdkpimumu.chi2[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten())
      output["BdToKPiMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bdkpimumu.l_xy[selections[selection_name]].flatten())
      output["BdToKPiMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bdkpimumu.l_xy_sig[selections[selection_name]].flatten())
      output["BdToKPiMuMu_jpsi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.mll_fullfit[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_mkstar"].fill(dataset=dataset_name, selection=selection_name, fit_mkstar=reco_bdkpimumu.mkstar_best_fullfit[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_barmkstar"].fill(dataset=dataset_name, selection=selection_name, fit_barmkstar=reco_bdkpimumu.barmkstar_best_fullfit[selections[selection_name]].flatten())

    # Build gen-to-reco map
    reco_genidx = reco_bdkpimumu.genPartIdx
    reco_hasmatch = reco_bdkpimumu.genPartIdx >= 0
    good_reco_genidx = reco_genidx[reco_hasmatch]
    good_gen_recoidx = reco_genidx.localindex[reco_hasmatch]

    # For each GenPart, default recomatch idx = -1
    gen_recoidx = genparts.pdgId.ones_like() * -1

    # For matched Bs, replace -1s with reco index
    gen_recoidx[good_reco_genidx] = good_gen_recoidx

    # Truth tree navigation: find K, pi, and mu daughters of Bds
    genpart_mother_idx = genparts.genPartIdxMother
    genpart_grandmother_idx = where(genpart_mother_idx >= 0, 
                                    genparts.genPartIdxMother[genpart_mother_idx],
                                    -1)
    genpart_mother_pdgId = where(genpart_mother_idx >= 0, genparts.pdgId[genpart_mother_idx], -1)
    genpart_grandmother_pdgId = where(genpart_grandmother_idx >= 0, genparts.pdgId[genpart_grandmother_idx], -1)

    mask_k_frombd = (abs(genparts.pdgId) == 321) & (abs(genpart_mother_pdgId) == 313) & (abs(genpart_grandmother_pdgId) == 511)
    k_frombd_grandmother_idx = genpart_grandmother_idx[mask_k_frombd]
    bd_k_idx = genparts.pt.ones_like().astype(int) * -1
    bd_k_idx[k_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_k_frombd]

    mask_pi_frombd = (abs(genparts.pdgId) == 211) & (abs(genpart_mother_pdgId) == 313) & (abs(genpart_grandmother_pdgId) == 511)
    pi_frombd_grandmother_idx = genpart_grandmother_idx[mask_pi_frombd]
    bd_pi_idx = genparts.pt.ones_like().astype(int) * -1
    bd_pi_idx[pi_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_pi_frombd]

    mask_mup_frombd = (genparts.pdgId == -13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 511)
    mup_frombd_grandmother_idx = genpart_grandmother_idx[mask_mup_frombd]
    bd_mup_idx = genparts.pt.ones_like().astype(int) * -1
    bd_mup_idx[mup_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_mup_frombd]

    mask_mum_frombd = (genparts.pdgId == 13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 511)
    mum_frombd_grandmother_idx = genpart_grandmother_idx[mask_mum_frombd]
    bd_mum_idx = genparts.pt.ones_like().astype(int) * -1
    bd_mum_idx[mum_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_mum_frombd]
    
    mask_kstar_frombd = (abs(genparts.pdgId) == 313) & (abs(genpart_mother_pdgId) == 511)
    kstar_frombd_mother_idx = genpart_mother_idx[mask_kstar_frombd]
    bd_kstar_idx = genparts.pt.ones_like().astype(int) * -1
    bd_kstar_idx[kstar_frombd_mother_idx] = genparts.pdgId.localindex[kstar_frombd_mother_idx]

    mask_jpsi_frombd = (abs(genparts.pdgId) == 443) & (abs(genpart_mother_pdgId) == 511)
    jpsi_frombd_mother_idx = genpart_mother_idx[mask_jpsi_frombd]
    bd_jpsi_idx = genparts.pt.ones_like().astype(int) * -1
    bd_jpsi_idx[jpsi_frombd_mother_idx] = genparts.pdgId.localindex[jpsi_frombd_mother_idx]

    # Jagged array of truth BdToKPiMuMus
    mask511 = (abs(genparts.pdgId)==511)
    truth_bdkpimumu = JaggedCandidateArray.candidatesfromcounts(
      genparts.pt[mask511].count(),
      pt           = genparts.pt[mask511].flatten(),
      eta          = genparts.eta[mask511].flatten(),
      phi          = genparts.phi[mask511].flatten(),
      mass         = genparts.mass[mask511].flatten(),
      recoIdx      = gen_recoidx[mask511].flatten(),
      gen_idx      = genparts.localindex[mask511].flatten(),
      k_idx        = bd_k_idx[mask511].flatten(),
      pi_idx       = bd_pi_idx[mask511].flatten(),
      mup_idx      = bd_mup_idx[mask511].flatten(),
      mum_idx      = bd_mum_idx[mask511].flatten(),
      kstar_idx    = bd_kstar_idx[mask511].flatten(),
      jpsi_idx     = bd_jpsi_idx[mask511].flatten(),
      recomatch_pt = genparts.pt[mask511].ones_like().flatten() * -1,
    )

    # Truth selections
    truth_selections = {}
    truth_selections["inclusive"] = truth_bdkpimumu.pt.ones_like().astype(bool)

    # Fiducial selection: match reco cuts
    truth_selections["fiducial"] =  (genparts.pt[truth_bdkpimumu.gen_idx] > 3.0) \
                                    & (genparts.pt[truth_bdkpimumu.k_idx] > 0.5) & (abs(genparts.eta[truth_bdkpimumu.k_idx]) < 2.5) \
                                    & (genparts.pt[truth_bdkpimumu.pi_idx] > 0.5) & (abs(genparts.eta[truth_bdkpimumu.pi_idx]) < 2.5) \
                                    & (genparts.pt[truth_bdkpimumu.mup_idx] > 1.5) & (abs(genparts.eta[truth_bdkpimumu.mup_idx]) < 2.4) \
                                    & (genparts.pt[truth_bdkpimumu.mum_idx] > 1.5) & (abs(genparts.eta[truth_bdkpimumu.mum_idx]) < 2.4) \
                                    & (genparts.p4[truth_bdkpimumu.k_idx].delta_r(genparts.p4[truth_bdkpimumu.pi_idx]) > 0.03) \
                                    & (genparts.p4[truth_bdkpimumu.k_idx].delta_r(genparts.p4[truth_bdkpimumu.mup_idx]) > 0.03) \
                                    & (genparts.p4[truth_bdkpimumu.k_idx].delta_r(genparts.p4[truth_bdkpimumu.mum_idx]) > 0.03) \
                                    & (genparts.p4[truth_bdkpimumu.pi_idx].delta_r(genparts.p4[truth_bdkpimumu.mup_idx]) > 0.03) \
                                    & (genparts.p4[truth_bdkpimumu.pi_idx].delta_r(genparts.p4[truth_bdkpimumu.mum_idx]) > 0.03) \
                                    & (genparts.p4[truth_bdkpimumu.mup_idx].delta_r(genparts.p4[truth_bdkpimumu.mum_idx]) > 0.03)

    # Matching: a bit more complicated for Bd
    truth_selections["matched"] = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched"][truth_bdkpimumu.recoIdx >= 0] = reco_bdkpimumu.mcmatch[truth_bdkpimumu.recoIdx[truth_bdkpimumu.recoIdx >= 0]]
    truth_selections["matched_swap"] = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap"][truth_bdkpimumu.recoIdx >= 0] = reco_bdkpimumu.mcmatch_swap[truth_bdkpimumu.recoIdx[truth_bdkpimumu.recoIdx >= 0]]
    truth_selections["unmatched"]       = ~(truth_selections["matched"] | truth_selections["matched_swap"])

    truth_selections["matched_sel"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_sel"][truth_selections["matched"]]   = selections["reco"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
    truth_selections["matched_tag"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_tag"][truth_selections["matched"]]   = selections["tag"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
    truth_selections["matched_probe"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_probe"][truth_selections["matched"]] = selections["probe"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

    truth_selections["matched_swap_sel"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap_sel"][truth_selections["matched_swap"]]   = selections["reco"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
    truth_selections["matched_swap_tag"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap_tag"][truth_selections["matched_swap"]]   = selections["tag"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
    truth_selections["matched_swap_probe"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap_probe"][truth_selections["matched_swap"]] = selections["probe"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

    truth_bdkpimumu.recomatch_pt[truth_selections["matched"] | truth_selections["matched_swap"]] = reco_bdkpimumu.fit_pt[truth_bdkpimumu.recoIdx[truth_selections["matched"] | truth_selections["matched_swap"]]]

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
    #   GrandMomID = cms.untracked.int32(511),
    #   NumberOfSisters= cms.untracked.int32(1),
    #   NumberOfAunts= cms.untracked.int32(1),
    #   SisterIDs=cms.untracked.vint32(-13),
    #   AuntIDs=cms.untracked.vint32(321),)
    truth_muons = genparts[abs(genparts.pdgId) == 13]
    truth_muons_probefilter = (abs(truth_muons.eta) < 2.5) \
                              & (truth_muons.pt > 5.0) \
                              & ~(
                                  (abs(genparts.pdgId[truth_muons.genPartIdxMother]) == 443) \
                                  & (abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons.genPartIdxMother]]) == 511)
                                  )
    event_probefilter = (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3)
    #for x in zip(truth_muons[~event_probefilter].pt.count(), truth_muons[~event_probefilter].pdgId, truth_muons[~event_probefilter].pt, truth_muons[~event_probefilter].eta, genparts[~event_probefilter].pdgId[truth_muons[~event_probefilter].genPartIdxMother]):
    #  print("Muon info in event failing probefilter:")
    #  print(x)
    truth_selections["probefilter"] = (truth_bdkpimumu.pt.ones_like() * ( \
                                        (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3) \
                                      )).astype(bool)


    # Fill truth histograms
    for selection_name in truth_selections.keys():
      output["TruthBdToKPiMuMu_pt"].fill(dataset=dataset_name, selection=selection_name, pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_eta"].fill(dataset=dataset_name, selection=selection_name, eta=truth_bdkpimumu.eta[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_phi"].fill(dataset=dataset_name, selection=selection_name, phi=truth_bdkpimumu.phi[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_mass"].fill(dataset=dataset_name, selection=selection_name, mass=truth_bdkpimumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_recopt_d_truthpt"].fill(dataset=dataset_name, selection=selection_name, 
          recopt_d_truthpt=where(truth_selections["matched"].flatten(), 
                                  ((truth_bdkpimumu.recomatch_pt / truth_bdkpimumu.pt)).flatten(),
                                  -1.0)[truth_selections[selection_name].flatten()])
      output["TruthBdToKPiMuMu_truthpt_recopt"].fill(dataset=dataset_name, selection=selection_name,
          reco_pt=truth_bdkpimumu.recomatch_pt[truth_selections[selection_name]].flatten(), 
          truth_pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten())

      output["TruthBdToKPiMuMu_k_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.k_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBdToKPiMuMu_pi_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.pi_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.pi_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.pi_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.pi_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBdToKPiMuMu_mup_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.mup_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBdToKPiMuMu_mum_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.mum_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBdToKPiMuMu_kstar_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.kstar_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.kstar_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.kstar_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.kstar_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBdToKPiMuMu_jpsi_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bdkpimumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bdkpimumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bdkpimumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bdkpimumu.jpsi_idx[truth_selections[selection_name]]].flatten()
                                          )

    output["nTruthMuon"].fill(dataset=dataset_name, nTruthMuon=genparts[abs(genparts.pdgId)==13].pt.count())

    # Tree outputs
    for selection_name in ["tagmatch", "tagmatchswap", "probematch", "probematchswap"]:
      output[f"Bcands_{selection_name}"][dataset_name].extend({
        "pt": reco_bdkpimumu.fit_pt[selections[selection_name]].flatten(),
        "eta": reco_bdkpimumu.fit_eta[selections[selection_name]].flatten(),
        "y": reco_bdkpimumu.fit_y[selections[selection_name]].flatten(),
        "phi": reco_bdkpimumu.fit_phi[selections[selection_name]].flatten(),
        "mass": reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten(),
        "l_xy": reco_bdkpimumu.l_xy[selections[selection_name]].flatten(),
        "l_xy_unc": reco_bdkpimumu.l_xy_unc[selections[selection_name]].flatten(),
        "sv_prob": reco_bdkpimumu.sv_prob[selections[selection_name]].flatten(),
        "cos2D": reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten(),
      })

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  # Inputs
  in_txt = {
    "Bd2KstarJpsi2KPiMuMu_probefilter":"/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_6/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KstarJpsi2KPiMuMu_inclusive":"/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_6/files_BdToJpsiKstar_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"
  }
  dataset_files = {}
  for dataset_name, filelistpath in in_txt.items():
    with open(filelistpath, 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

  ts_start = time.time()
  output = processor.run_uproot_job(dataset_files,
                                treename='Events',
                                processor_instance=MCEfficencyProcessor(),
                                executor=processor.futures_executor,
                                executor_args={'workers': 32, 'flatten': False},
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

  print("Total time: {} seconds".format(ts_end - ts_start))
  print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))
  print("Total nevents: {}".format(total_events))
