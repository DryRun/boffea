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
    self._accumulator["truth_cutflow"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))

    self._Bcand_selections = ["recomatch"]
    for trigger in self._triggers:
      self._Bcand_selections.append(f"tagmatch_{trigger}")
      self._Bcand_selections.append(f"probematch_{trigger}")
    for selection_name in self._Bcand_selections:
      self._accumulator[f"Bcands_{selection_name}"] = processor.defaultdict_accumulator(partial(Bcand_accumulator, cols=["pt", "eta", "y", "phi", "mass"]))

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BuToKMuMu_fit_pt_y_mass"] = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0),
                                                            hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1)
                                                          )
    self._accumulator["BuToKMuMu_fit_pt"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BuToKMuMu_fit_eta"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BuToKMuMu_fit_phi"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BuToKMuMu_fit_mass"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
    self._accumulator["BToKMuMu_chi2"]         = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BuToKMuMu_fit_cos2D"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BuToKMuMu_fit_theta2D"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, 0., math.pi))
    self._accumulator["BuToKMuMu_l_xy"]        = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BuToKMuMu_l_xy_sig"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))
    self._accumulator["BuToKMuMu_sv_prob"]     = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("sv_prob", r"SV prob", 50, 0.0, 1.0))
    self._accumulator["BuToKMuMu_jpsi_mass"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))

    #self._accumulator["BuToKMuMu_tag_fit_pt"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["BuToKMuMu_tag_fit_eta"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    #self._accumulator["BuToKMuMu_tag_fit_phi"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    #self._accumulator["BuToKMuMu_tag_fit_mass"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
    #self._accumulator["BuToKMuMu_tag_fit_chi2"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    #self._accumulator["BuToKMuMu_tag_fit_cos2D"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    #self._accumulator["BuToKMuMu_tag_l_xy"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    #self._accumulator["BuToKMuMu_tag_l_xy_sig"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

    #self._accumulator["BuToKMuMu_probe_fit_pt"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["BuToKMuMu_probe_fit_eta"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    #self._accumulator["BuToKMuMu_probe_fit_phi"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    #self._accumulator["BuToKMuMu_probe_fit_mass"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))
    #self._accumulator["BuToKMuMu_probe_fit_chi2"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    #self._accumulator["BuToKMuMu_probe_fit_cos2D"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    #self._accumulator["BuToKMuMu_probe_l_xy"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    #self._accumulator["BuToKMuMu_probe_l_xy_sig"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("BToKMuMu_l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))

    #self._accumulator["TruthBuToKMuMu_pt"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("TruthBToKMuMu_pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    #self._accumulator["TruthBuToKMuMu_eta"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("TruthBToKMuMu_eta", r"$\eta$", 50, -5.0, 5.0))
    #self._accumulator["TruthBuToKMuMu_phi"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("TruthBToKMuMu_phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
    #self._accumulator["TruthBuToKMuMu_mass"]  = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("TruthBToKMuMu_mass", r"$m$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))

    self._accumulator["nTruthMuon"]  = hist.Hist("Events", dataset_axis, hist.Bin("nTruthMuon", r"N(truth muons)", 11, -0.5, 10.5))

    self._accumulator["TruthProbeMuon_parent"]      = hist.Hist("Events", dataset_axis, hist.Bin("parentPdgId", "Parent pdgId", 1001, -0.5, 1000.5))
    self._accumulator["TruthProbeMuon_grandparent"] = hist.Hist("Events", dataset_axis, hist.Bin("grandparentPdgId", "Grandparent pdgId", 1001, -0.5, 1000.5))

    self._accumulator["TruthBuToKMuMu_pt"]               = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBuToKMuMu_eta"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("eta", r"$\eta$", 50, -5.0, 5.0))
    self._accumulator["TruthBuToKMuMu_absy"]             = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("absy", r"$|y|$", 20, 0.0, 5.0))
    self._accumulator["TruthBuToKMuMu_phi"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["TruthBuToKMuMu_mass"]             = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("mass", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBuToKMuMu_recopt_d_truthpt"] = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("recopt_d_truthpt", r"$m$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1))

    self._accumulator["TruthBuToKMuMu_k_p4"]   = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                    hist.Bin("pt", r"$p_{T}(K^{\pm})$ [GeV]", 100, 0.0, 100.0),
                                                    hist.Bin("eta", r"$\eta(K^{\pm})$ [GeV]", 20, -5., 5.),
                                                    hist.Bin("phi", r"$\phi(K^{\pm})$ [GeV]", 20, -1*math.pi, math.pi),
                                                    hist.Bin("mass", r"$m(K^{\pm})$ [GeV]", 30, K_MASS*0.5, K_MASS*2.0)
                                                  )
    self._accumulator["TruthBuToKMuMu_mup_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                    hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
                                                    hist.Bin("eta", r"$\eta(\mu^{+})$ [GeV]", 20, -5., 5.),
                                                    hist.Bin("phi", r"$\phi(\mu^{+})$ [GeV]", 20, -1*math.pi, math.pi),
                                                    hist.Bin("mass", r"$m(\mu^{+})$ [GeV]", 30, MUON_MASS*0.5, MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBuToKMuMu_mum_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                    hist.Bin("pt", r"$p_{T}(\mu^{-})$ [GeV]", 100, 0.0, 100.0),
                                                    hist.Bin("eta", r"$\eta(\mu^{-})$ [GeV]", 20, -5., 5.),
                                                    hist.Bin("phi", r"$\phi(\mu^{-})$ [GeV]", 20, -1*math.pi, math.pi),
                                                    hist.Bin("mass", r"$m(\mu^{-})$ [GeV]", 30, MUON_MASS*0.5, MUON_MASS*2.0)
                                                  )
    self._accumulator["TruthBuToKMuMu_jpsi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                    hist.Bin("pt", r"$p_{T}(J/\psi)$ [GeV]", 100, 0.0, 100.0),
                                                    hist.Bin("eta", r"$\eta(J/\psi)$ [GeV]", 20, -5., 5.),
                                                    hist.Bin("phi", r"$\phi(J/\psi)$ [GeV]", 20, -1*math.pi, math.pi),
                                                    hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 30, JPSI_1S_MASS*0.5, JPSI_1S_MASS*2.0)
                                                  )

    # One entry per truth B
    # - If truth B is not matched to reco, or if reco fails selection, fill (-1, truthpt)
    self._accumulator["TruthBuToKMuMu_truthpt_recopt"] = hist.Hist("Events", dataset_axis, selection_axis_truth,
                                                          hist.Bin("reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 500, 0., 100.),
                                                          hist.Bin("truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 500, 0., 100.)
                                                          )

  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    output["nevents"][dataset_name] += df.size

    # Create jagged object arrays
    reco_bukmumu = dataframereader.reco_bukmumu(df, is_mc=True)
    reco_muons     = dataframereader.reco_muons(df, is_mc=True)
    trigger_muons  = dataframereader.trigger_muons(df, is_mc=True)
    probe_tracks   = dataframereader.probe_tracks(df, is_mc=True)
    genparts       = dataframereader.genparts(df, is_mc=True)

    reco_bukmumu.add_attributes(
      l1_softId    = reco_muons.softId[reco_bukmumu.l1_idx],
      l1_softMvaId = reco_muons.softMvaId[reco_bukmumu.l1_idx],
      l2_softId    = reco_muons.softId[reco_bukmumu.l2_idx],
      l2_softMvaId = reco_muons.softMvaId[reco_bukmumu.l2_idx],
      l1_MediumId    = reco_muons.MediumId[reco_bukmumu.l1_idx],
      l2_MediumId    = reco_muons.MediumId[reco_bukmumu.l2_idx],
    )

    # Truth matching
    reco_bukmumu.l1_genIdx = reco_muons.genPartIdx[reco_bukmumu.l1_idx] 
    reco_bukmumu.l2_genIdx = reco_muons.genPartIdx[reco_bukmumu.l2_idx] 
    reco_bukmumu.k_genIdx  = probe_tracks.genPartIdx[reco_bukmumu.kIdx]

    reco_bukmumu.l1_genMotherIdx = where(reco_bukmumu.l1_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bukmumu.l1_genIdx], 
                                                              -1)
    reco_bukmumu.l2_genMotherIdx = where(reco_bukmumu.l2_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bukmumu.l2_genIdx], 
                                                              -1)
    reco_bukmumu.k_genMotherIdx = where(reco_bukmumu.k_genIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bukmumu.k_genIdx], 
                                                              -1)

    reco_bukmumu.l1_genGrandmotherIdx = where(reco_bukmumu.l1_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bukmumu.l1_genMotherIdx], 
                                                              -1)
    reco_bukmumu.l2_genGrandmotherIdx = where(reco_bukmumu.l2_genMotherIdx >= 0, 
                                                              genparts.genPartIdxMother[reco_bukmumu.l2_genMotherIdx], 
                                                              -1)

    reco_bukmumu.l1_genMotherPdgId = where(reco_bukmumu.l1_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bukmumu.l1_genMotherIdx],
                                                              -1)
    reco_bukmumu.l2_genMotherPdgId = where(reco_bukmumu.l2_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bukmumu.l2_genMotherIdx],
                                                              -1)
    reco_bukmumu.k_genMotherPdgId = where(reco_bukmumu.k_genMotherIdx >= 0, 
                                                              genparts.pdgId[reco_bukmumu.k_genMotherIdx],
                                                              -1)

    reco_bukmumu.l1_genGrandmotherPdgId = where(reco_bukmumu.l1_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bukmumu.l1_genGrandmotherIdx],
                                                              -1)
    reco_bukmumu.l2_genGrandmotherPdgId = where(reco_bukmumu.l2_genGrandmotherIdx >= 0, 
                                                              genparts.pdgId[reco_bukmumu.l2_genGrandmotherIdx],
                                                              -1)

    reco_bukmumu.mcmatch = (abs(reco_bukmumu.l1_genMotherPdgId) == 443) \
                              & (abs(reco_bukmumu.l2_genMotherPdgId) == 443) \
                              & (abs(reco_bukmumu.l2_genGrandmotherPdgId) == 521) \
                              & (abs(reco_bukmumu.l2_genGrandmotherPdgId) == 521) \
                              & (abs(reco_bukmumu.k_genMotherPdgId) == 521) \
                              & (reco_bukmumu.l1_genGrandmotherIdx == reco_bukmumu.l2_genGrandmotherIdx) \
                              & (reco_bukmumu.l1_genGrandmotherIdx == reco_bukmumu.k_genMotherIdx) 

    reco_bukmumu.genPartIdx = where(reco_bukmumu.mcmatch, reco_bukmumu.l1_genGrandmotherIdx, -1)


    # Tag/probe selection
    """
    reco_bukmumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggering[reco_bukmumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggering[reco_bukmumu.l2_idx])
    reco_bukmumu.add_attributes(MuonIsTrigCount = reco_bukmumu.Muon1IsTrig.astype(int) + reco_bukmumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bukmumu.add_attributes(TagCount = reco_bukmumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bukmumu.MuonIsTrigCount)
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
      reco_bukmumu.add_attributes(**{
        f"Muon1IsTrig_{trigger}": getattr(reco_muons, f"isTriggering_{trigger}")[reco_bukmumu.l1_idx],
        f"Muon2IsTrig_{trigger}": getattr(reco_muons, f"isTriggering_{trigger}")[reco_bukmumu.l2_idx],
        f"Muon1IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggering_{trigger}")[reco_bukmumu.l1_idx] \
                                        & (reco_muons.pt[reco_bukmumu.l1_idx] > tagmuon_ptcuts[trigger]) \
                                        & (reco_muons.dxySig[reco_bukmumu.l1_idx] > tagmuon_ipcuts[trigger]),
        f"Muon2IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggering_{trigger}")[reco_bukmumu.l2_idx] \
                                        & (reco_muons.pt[reco_bukmumu.l2_idx] > tagmuon_ptcuts[trigger]) \
                                        & (reco_muons.dxySig[reco_bukmumu.l2_idx] > tagmuon_ipcuts[trigger]),
      })
      reco_bukmumu.add_attributes(**{
        f"MuonIsTrigCount_{trigger}": getattr(reco_bukmumu, f"Muon1IsTrig_{trigger}").astype(int) + getattr(reco_bukmumu, f"Muon2IsTrig_{trigger}").astype(int)
      })
      event_ntriggingmuons = getattr(reco_muons, f"isTriggering_{trigger}").astype(int).sum()
      reco_bukmumu.add_attributes(**{
        f"TagCount_{trigger}": getattr(reco_bukmumu, f"MuonIsTrigCount_{trigger}").ones_like() * event_ntriggingmuons - getattr(reco_bukmumu, f"MuonIsTrigCount_{trigger}")
      })
    reco_bukmumu.add_attributes(l_xy_sig = where(reco_bukmumu.l_xy_unc > 0, reco_bukmumu.l_xy / reco_bukmumu.l_xy_unc, -1.e20))

    # General selection
    trigger_masks = {}
    trigger_masks["inclusive"] = np.full_like(df["HLT_Mu7_IP4"], True)
    trigger_masks["HLT_Mu7_IP4"] = (df["HLT_Mu7_IP4"] == 1)
    trigger_masks["HLT_Mu9_IP5"] = (df["HLT_Mu9_IP5"] == 1)
    trigger_masks["HLT_Mu9_IP6"] = (df["HLT_Mu9_IP6"] == 1)
    trigger_masks["HLT_Mu12_IP6"] = (df["HLT_Mu12_IP6"] == 1)
    trigger_masks["HLT_Mu9_IP5|HLT_Mu9_IP6"] = trigger_masks["HLT_Mu9_IP5"] | trigger_masks["HLT_Mu9_IP6"]
    trigger_masks["HLT_Mu9_IP5&~HLT_Mu7_IP4"] = trigger_masks["HLT_Mu9_IP5"] & ~trigger_masks["HLT_Mu7_IP4"]
    nominal_trigmask = trigger_masks["HLT_Mu9_IP5|HLT_Mu9_IP6"]

    reco_bukmumu_mask_template = reco_bukmumu.pt.ones_like().astype(bool)
    selections = {}
    selections["sv_pt"]    = (reco_bukmumu.fit_pt > final_cuts["Bu"]["sv_pt"])
    selections["l_xy_sig"] = (abs(reco_bukmumu.l_xy_sig) > final_cuts["Bu"]["l_xy_sig"])
    selections["sv_prob"]  = (reco_bukmumu.sv_prob > final_cuts["Bu"]["sv_prob"])
    selections["cos2D"]    = (reco_bukmumu.fit_cos2D > final_cuts["Bu"]["cos2D"])
    selections["l1"]       =  (reco_bukmumu.fit_l1_pt > final_cuts["Bu"]["l1_pt"]) & (abs(reco_bukmumu.fit_l1_eta) < 2.4) & (reco_bukmumu.l1_softId)
    selections["l2"]       =  (reco_bukmumu.fit_l2_pt > final_cuts["Bu"]["l2_pt"]) & (abs(reco_bukmumu.fit_l2_eta) < 2.4) & (reco_bukmumu.l2_softId)
    #selections["l2"]       = (abs(reco_bukmumu.fit_l2_eta) < 2.4) & (reco_bukmumu.l2_softId)
    #selections["l2"]       = (selections["l2"] & where(abs(reco_bukmumu.fit_l2_eta) < 1.4, 
    #                                                  (reco_bukmumu.fit_l2_pt > 1.5), 
    #                                                  (reco_bukmumu.fit_l2_pt > 1.0))).astype(bool)
    selections["k"]        = (reco_bukmumu.fit_k_pt > final_cuts["Bu"]["k_pt"]) & (abs(reco_bukmumu.fit_k_eta) < 2.5)
    #selections["dR"]       = (delta_r(reco_bukmumu.fit_k_eta, reco_bukmumu.fit_l1_eta, reco_bukmumu.fit_k_phi, reco_bukmumu.fit_l1_phi) > 0.03) \
    #                            & (delta_r(reco_bukmumu.fit_k_eta, reco_bukmumu.fit_l2_eta, reco_bukmumu.fit_k_phi, reco_bukmumu.fit_l2_phi) > 0.03) \
    #                            & (delta_r(reco_bukmumu.fit_l1_eta, reco_bukmumu.fit_l2_eta, reco_bukmumu.fit_l1_phi, reco_bukmumu.fit_l2_phi) > 0.03)

    selections["jpsi"]    = abs(reco_bukmumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    #(JPSI_1S_MASS - JPSI_WINDOW < reco_bukmumu.mll_fullfit) & (reco_bukmumu.mll_fullfit < JPSI_1S_MASS + JPSI_WINDOW)

    # Final selections
    selections["inclusive"] = reco_bukmumu.fit_pt.ones_like().astype(bool)
    selections["reco"]      = selections["sv_pt"] \
                              & selections["l_xy_sig"] \
                              & selections["sv_prob"] \
                              & selections["cos2D"] \
                              & selections["l1"] \
                              & selections["l2"] \
                              & selections["k"] \
                              & selections["jpsi"] \
                              #& selections["dR"] \
    selections["truthmatched"] = (reco_bukmumu.genPartIdx >= 0)
    selections["recomatch"]    = selections["reco"] & selections["truthmatched"]

    for trigger in self._triggers:
      trigger_mask = (df[trigger] == 1) * reco_bukmumu_mask_template

      selections[f"tag_{trigger}"]            = selections[f"reco"] & trigger_mask & (getattr(reco_bukmumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bukmumu, f"Muon2IsTrigTight_{trigger}"))
      selections[f"tagmatch_{trigger}"]       = selections[f"tag_{trigger}"] & selections["truthmatched"]
      selections[f"tagunmatched_{trigger}"]   = selections[f"tag_{trigger}"] & (~selections["truthmatched"])
      
      selections[f"probe_{trigger}"]          = selections[f"reco"] & trigger_mask & (getattr(reco_bukmumu, f"TagCount_{trigger}") >= 1)
      selections[f"probematch_{trigger}"]     = selections[f"probe_{trigger}"] & selections["truthmatched"]
      selections[f"probeunmatched_{trigger}"] = selections[f"probe_{trigger}"] & (~selections["truthmatched"])

    # If more than one B is selected, choose best chi2
    selections["recomatch"] = selections["recomatch"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections["recomatch"]].min())
    for trigger in self._triggers:
      selections[f"tag_{trigger}"]   = selections[f"tag_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"tag_{trigger}"]].min())
      selections[f"tagmatch_{trigger}"]   = selections[f"tagmatch_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"tagmatch_{trigger}"]].min())
      selections[f"tagunmatched_{trigger}"]   = selections[f"tagunmatched_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"tagunmatched_{trigger}"]].min())

      selections[f"probe_{trigger}"] = selections[f"probe_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"probe_{trigger}"]].min())
      selections[f"probematch_{trigger}"] = selections[f"probematch_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"probematch_{trigger}"]].min())
      selections[f"probeunmatched_{trigger}"] = selections[f"probeunmatched_{trigger}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections[f"probeunmatched_{trigger}"]].min())

    # Fill cutflow
    cumulative_selection = copy.deepcopy(reco_bukmumu_mask_template)
    output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    #cumulative_selection = cumulative_selection & (nominal_trigmask * reco_bukmumu_mask_template).astype(bool)
    #output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k", "jpsi"]: # dR
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow"][dataset_name]["tag"] += selections["tag_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probe"] += selections["probe_HLT_Mu7_IP4"].sum().sum()

    # Fill reco histograms
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    #sels_to_fill = ["inclusive", "reco"]
    #for x in ["tag", "tagmatch", "tagunmatched", "probe", "probematch", "probeunmatched"]:
    #  sels_to_fill.extend([f"{x}_{y}" for y in trigger_masks.keys()])
    #for selection_name in sels_to_fill:
    selection_names = ["inclusive", "reco", "recomatch", "truthmatched"]
    for trigger in self._triggers:
      selection_names.extend([f"tag_{trigger}", f"tagmatch_{trigger}", f"tagunmatched_{trigger}", f"probe_{trigger}", f"probematch_{trigger}", f"probeunmatched_{trigger}"])
      selection_names.extend([f"tagHiPtmatch_{trigger}", f"probeHiPtmatch_{trigger}"])
    for selection_name in selection_names:
      output["BuToKMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bukmumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_y=reco_bukmumu.fit_y[selections[selection_name]].flatten(),
                                            fit_mass=reco_bukmumu.fit_mass[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bukmumu.fit_pt[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bukmumu.fit_eta[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bukmumu.fit_phi[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bukmumu.fit_mass[selections[selection_name]].flatten())
      output["BToKMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bukmumu.chi2[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bukmumu.fit_cos2D[selections[selection_name]].flatten())
      output["BuToKMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bukmumu.fit_cos2D[selections[selection_name]]).flatten())
      output["BuToKMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bukmumu.l_xy[selections[selection_name]].flatten())
      output["BuToKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bukmumu.l_xy_sig[selections[selection_name]].flatten())
      output["BuToKMuMu_sv_prob"].fill(dataset=dataset_name, selection=selection_name, sv_prob=reco_bukmumu.sv_prob[selections[selection_name]].flatten())
      output["BuToKMuMu_jpsi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bukmumu.mll_fullfit[selections[selection_name]].flatten())

    # Build gen-to-reco map
    reco_genidx = reco_bukmumu.genPartIdx
    reco_hasmatch = reco_bukmumu.genPartIdx >= 0
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

    mask_k_frombu = (abs(genparts.pdgId) == 211) & (abs(genpart_mother_pdgId) == 521)
    k_frombu_mother_idx = genpart_mother_idx[mask_k_frombu]
    bu_k_idx = genparts.pt.ones_like().astype(int) * -1
    bu_k_idx[k_frombu_mother_idx] = genparts.pdgId.localindex[mask_k_frombu]

    mask_mup_frombu = (genparts.pdgId == 13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 521)
    mup_frombu_grandmother_idx = genpart_grandmother_idx[mask_mup_frombu]
    bu_mup_idx = genparts.pt.ones_like().astype(int) * -1
    bu_mup_idx[mup_frombu_grandmother_idx] = genparts.pdgId.localindex[mask_mup_frombu]

    mask_mum_frombu = (genparts.pdgId == -13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 521)
    mum_frombu_grandmother_idx = genpart_grandmother_idx[mask_mum_frombu]
    bu_mum_idx = genparts.pt.ones_like().astype(int) * -1
    bu_mum_idx[mum_frombu_grandmother_idx] = genparts.pdgId.localindex[mask_mum_frombu]

    mask_jpsi_frombu = (genparts.pdgId == 443) & (abs(genpart_mother_pdgId) == 521)
    jpsi_frombu_mother_idx = genpart_mother_idx[mask_jpsi_frombu]
    bu_jpsi_idx = genparts.pt.ones_like().astype(int) * -1
    bu_jpsi_idx[jpsi_frombu_mother_idx] = genparts.pdgId.localindex[mask_jpsi_frombu]

    # Count number of soft photon daughters
    nChildrenNoPhoton = genparts.nChildren - count_photon_children(genparts.genPartIdxMother, genparts.pdgId, genparts.pt)

    # Jagged array of truth BToKMuMus
    mask521 = (abs(genparts.pdgId)==521) & (bu_jpsi_idx >= 0) & (bu_k_idx >= 0) & (bu_mup_idx >= 0) & (bu_mum_idx >= 0) & (nChildrenNoPhoton == 2)
    #mask521 = mask521 & (mask521.sum() <= 1)
    truth_bukmumu = JaggedCandidateArray.candidatesfromcounts(
      genparts.pt[mask521].count(),
      pt                = genparts.pt[mask521].flatten(),
      eta               = genparts.eta[mask521].flatten(),
      phi               = genparts.phi[mask521].flatten(),
      mass              = genparts.mass[mask521].flatten(),
      reco_idx          = gen_recoidx[mask521].flatten(),
      gen_idx           = genparts.localindex[mask521].flatten(),
      k_idx             = bu_k_idx[mask521].flatten(),
      mup_idx           = bu_mup_idx[mask521].flatten(),
      mum_idx           = bu_mum_idx[mask521].flatten(),
      jpsi_idx          = bu_jpsi_idx[mask521].flatten(),
      recomatch_pt      = genparts.pt[mask521].ones_like().flatten() * -1,
      status            = genparts.status[mask521].flatten(),
      #nChildrenNoPhoton = nChildrenNoPhoton[mask521].flatten()
    )
    truth_bukmumu.add_attributes(
      y=np.log((np.sqrt(truth_bukmumu.mass**2 
        + truth_bukmumu.pt**2*np.cosh(truth_bukmumu.eta)**2) 
        + truth_bukmumu.pt*np.sinh(truth_bukmumu.eta)) / np.sqrt(truth_bukmumu.mass**2 
        + truth_bukmumu.pt**2))
      )

    #restrict_children = truth_bukmumu.nChildrenNoPhoton == 2 #(truth_bukmumu.nChildren == 2) |  ((truth_bukmumu.nChildren == 3) & (truth_bukmumu.photon_idx >= 0))
    #print("Fraction passing child cut:")
    #print(truth_bukmumu.pt[restrict_children].flatten().size / truth_bukmumu.pt.flatten().size)


    # Compute invariant mass of K and J/psi
    #jpsi_p4 = genparts[truth_bukmumu.jpsi_idx].p4
    #K_p4 = genparts[truth_bukmumu.k_idx].p4
    #inv_mass = (genparts[truth_bukmumu.mup_idx].p4 + genparts[truth_bukmumu.mum_idx].p4 + genparts[truth_bukmumu.k_idx].p4).mass
    #print("DEBUG : Invariant mass cut efficiency")
    #print( truth_bukmumu.pt[abs(inv_mass - BU_MASS) < 0.1].count().sum() / truth_bukmumu.pt.count().sum())
    #print(inv_mass[abs(inv_mass - BU_MASS) > 0.1].flatten())
    #truth_bukmumu = truth_bukmumu[abs(inv_mass - BU_MASS) < 0.1]

    # Truth selections
    truth_selections = {}
    truth_selections["inclusive"] = truth_bukmumu.pt.ones_like().astype(bool)

    # Fiducial selection: match reco cuts
    truth_selections["fiducial"] = (genparts.pt[truth_bukmumu.gen_idx] > 3.0) \
                                    & (genparts.pt[truth_bukmumu.k_idx] > 0.5) & (abs(genparts.eta[truth_bukmumu.k_idx]) < 2.5) \
                                    & (genparts.pt[truth_bukmumu.mup_idx] > 1.0) & (abs(genparts.eta[truth_bukmumu.mup_idx]) < 2.4) \
                                    & (genparts.pt[truth_bukmumu.mum_idx] > 1.0) & (abs(genparts.eta[truth_bukmumu.mum_idx]) < 2.4) \

    truth_selections["matched"] = (truth_bukmumu.reco_idx >= 0) #(awkward.fromiter(truth_bukmumu.reco_idx) >= 0)
    truth_selections["unmatched"] = ~truth_selections["matched"]
    truth_selections["matched_sel"] = truth_bukmumu.reco_idx.zeros_like().astype(bool)
    truth_selections["matched_sel"][truth_selections["matched"]] = selections["reco"][truth_bukmumu.reco_idx[truth_selections["matched"]]]

    for trigger in self._triggers:
      truth_selections[f"matched_tag_{trigger}"] = truth_bukmumu.reco_idx.zeros_like().astype(bool)
      truth_selections[f"matched_tag_{trigger}"][truth_selections["matched"]] = selections[f"tag_{trigger}"][truth_bukmumu.reco_idx[truth_selections["matched"]]]
      truth_selections[f"matched_probe_{trigger}"] = truth_bukmumu.reco_idx.zeros_like().astype(bool)
      truth_selections[f"matched_probe_{trigger}"][truth_selections["matched"]] = selections[f"probe_{trigger}"][truth_bukmumu.reco_idx[truth_selections["matched"]]]

    # Truth "cutflow"
    truth_selection_names = ["inclusive", "fiducial", "matched", "unmatched", "matched_sel"]
    for trigger in self._triggers:
      truth_selection_names.extend([f"matched_tag_{trigger}", f"matched_probe_{trigger}"])
    for selection_name in truth_selection_names:
      output["truth_cutflow"][dataset_name][selection_name] = truth_selections[selection_name].sum().sum()

    truth_bukmumu.recomatch_pt[truth_selections["matched"]] = reco_bukmumu.fit_pt[truth_bukmumu.reco_idx[truth_selections["matched"]]]

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
    #   AuntIDs=cms.untracked.vint32(211),)
    truth_muons = genparts[abs(genparts.pdgId) == 13]
    truth_muons_probefilter = (abs(truth_muons.eta) <= 2.5) \
                              & (truth_muons.pt >= 5.0) \
                              & ~(
                                  (abs(genparts.pdgId[truth_muons.genPartIdxMother]) == 443) \
                                  & (abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons.genPartIdxMother]]) == 521)
                                  & (bu_k_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (nChildrenNoPhoton[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] == 2) \
                                  )
    #event_probefilter = (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3)
    #for x in zip(truth_muons[~event_probefilter].pt.count(), truth_muons[~event_probefilter].pdgId, truth_muons[~event_probefilter].pt, truth_muons[~event_probefilter].eta, genparts[~event_probefilter].pdgId[truth_muons[~event_probefilter].genPartIdxMother]):
    #  print("Muon info in event failing probefilter:")
    #  print(x)
    truth_selections["probefilter"] = (truth_bukmumu.pt.ones_like() * ( \
                                        (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3) \
                                      )).astype(bool)

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
      output["TruthBuToKMuMu_pt"].fill(dataset=dataset_name, selection=selection_name, pt=truth_bukmumu.pt[truth_selections[selection_name]].flatten())
      output["TruthBuToKMuMu_eta"].fill(dataset=dataset_name, selection=selection_name, eta=truth_bukmumu.eta[truth_selections[selection_name]].flatten())
      output["TruthBuToKMuMu_absy"].fill(dataset=dataset_name, selection=selection_name, absy=abs(truth_bukmumu.y[truth_selections[selection_name]].flatten()))
      output["TruthBuToKMuMu_phi"].fill(dataset=dataset_name, selection=selection_name, phi=truth_bukmumu.phi[truth_selections[selection_name]].flatten())
      output["TruthBuToKMuMu_mass"].fill(dataset=dataset_name, selection=selection_name, mass=truth_bukmumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBuToKMuMu_recopt_d_truthpt"].fill(dataset=dataset_name, selection=selection_name, 
          recopt_d_truthpt=where(truth_selections["matched"].flatten(), 
                                  ((truth_bukmumu.recomatch_pt / truth_bukmumu.pt)).flatten(),
                                  -1.0)[truth_selections[selection_name].flatten()])
      output["TruthBuToKMuMu_truthpt_recopt"].fill(dataset=dataset_name, selection=selection_name,
          reco_pt=truth_bukmumu.recomatch_pt[truth_selections[selection_name]].flatten(), 
          truth_pt=truth_bukmumu.pt[truth_selections[selection_name]].flatten())

      output["TruthBuToKMuMu_k_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bukmumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bukmumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bukmumu.k_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bukmumu.k_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBuToKMuMu_mup_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bukmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bukmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bukmumu.mup_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bukmumu.mup_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBuToKMuMu_mum_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bukmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bukmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bukmumu.mum_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bukmumu.mum_idx[truth_selections[selection_name]]].flatten()
                                          )
      output["TruthBuToKMuMu_jpsi_p4"].fill(dataset=dataset_name, selection=selection_name, 
                                            pt=genparts.pt[truth_bukmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            eta=genparts.eta[truth_bukmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            phi=genparts.phi[truth_bukmumu.jpsi_idx[truth_selections[selection_name]]].flatten(), 
                                            mass=genparts.mass[truth_bukmumu.jpsi_idx[truth_selections[selection_name]]].flatten()
                                          )

    output["nTruthMuon"].fill(dataset=dataset_name, nTruthMuon=genparts[abs(genparts.pdgId)==13].pt.count())

    # Tree outputs
    for selection_name in self._Bcand_selections:
      output[f"Bcands_{selection_name}"][dataset_name].extend({
        "pt": reco_bukmumu.fit_pt[selections[selection_name]].flatten(),
        "eta": reco_bukmumu.fit_eta[selections[selection_name]].flatten(),
        "y": reco_bukmumu.fit_y[selections[selection_name]].flatten(),
        "phi": reco_bukmumu.fit_phi[selections[selection_name]].flatten(),
        "mass": reco_bukmumu.fit_mass[selections[selection_name]].flatten(),
        #"l_xy": reco_bukmumu.l_xy[selections[selection_name]].flatten(),
        #"l_xy_unc": reco_bukmumu.l_xy_unc[selections[selection_name]].flatten(),
        #"sv_prob": reco_bukmumu.sv_prob[selections[selection_name]].flatten(),
        #"cos2D": reco_bukmumu.fit_cos2D[selections[selection_name]].flatten(),
      })

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  # Inputs
  in_txt = {
      "Bu2PiJpsi2KMuMu_inclusive": "/home/dyu7/BFrag/boffea/barista/filelists/v2_7/files_Bu2PiJPsi2PiMuMu.txt",
  }
  dataset_files = {}
  for dataset_name, filelistpath in in_txt.items():
    with open(filelistpath, 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

  ts_start = time.time()
  import psutil
  print("psutil.cpu_count() = ".format(psutil.cpu_count()))
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

  print("Cutflow:")
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
