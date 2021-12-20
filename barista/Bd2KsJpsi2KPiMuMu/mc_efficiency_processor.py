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
from brazil.reweighting import reweight_trkpt, reweight_y
from brazil.fonll_sfs import fonll_sfs

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
    
    self._Bcand_selections = ["recomatch", "recomatchswap"]
    for trigger in self._triggers:
      self._Bcand_selections.append(f"tagmatch_{trigger}")
      self._Bcand_selections.append(f"probematch_{trigger}")
      self._Bcand_selections.append(f"tagmatchswap_{trigger}")
      self._Bcand_selections.append(f"probematchswap_{trigger}")

      self._Bcand_selections.append(f"tagxmatch_{trigger}")
      self._Bcand_selections.append(f"tagxmatchswap_{trigger}")

      #self._Bcand_selections.append(f"tagHiTrkPtmatch_{trigger}")
      #self._Bcand_selections.append(f"probeHiTrkPtmatch_{trigger}")
      #self._Bcand_selections.append(f"tagHiTrkPtmatchswap_{trigger}")
      #self._Bcand_selections.append(f"probeHiTrkPtmatchswap_{trigger}")

      self._Bcand_selections.append(f"tagHiMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"probeHiMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"tagHiMuonPtmatchswap_{trigger}")
      self._Bcand_selections.append(f"probeHiMuonPtmatchswap_{trigger}")

      self._Bcand_selections.append(f"tagMediumMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"probeMediumMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"tagMediumMuonPtmatchswap_{trigger}")
      self._Bcand_selections.append(f"probeMediumMuonPtmatchswap_{trigger}")

      self._Bcand_selections.append(f"tagHugeMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"probeHugeMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"tagHugeMuonPtmatchswap_{trigger}")
      self._Bcand_selections.append(f"probeHugeMuonPtmatchswap_{trigger}")

      self._Bcand_selections.append(f"tagVarMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"probeVarMuonPtmatch_{trigger}")
      self._Bcand_selections.append(f"tagVarMuonPtmatchswap_{trigger}")
      self._Bcand_selections.append(f"probeVarMuonPtmatchswap_{trigger}")

      self._Bcand_selections.append(f"tagMediumMuonmatch_{trigger}")
      self._Bcand_selections.append(f"probeMediumMuonmatch_{trigger}")
      self._Bcand_selections.append(f"tagMediumMuonmatchswap_{trigger}")
      self._Bcand_selections.append(f"probeMediumMuonmatchswap_{trigger}")

      #self._Bcand_selections.append(f"tagMaxPtmatch_{trigger}")
      #self._Bcand_selections.append(f"probeMaxPtmatch_{trigger}")
      #self._Bcand_selections.append(f"tagMaxPtmatchswap_{trigger}")
      #self._Bcand_selections.append(f"probeMaxPtmatchswap_{trigger}")


    for selection_name in self._Bcand_selections:
      self._accumulator[f"Bcands_{selection_name}"] = processor.defaultdict_accumulator(partial(Bcand_accumulator, cols=["pt", "y", "phi", "mass"])) # eta

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    #self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BdToKPiMuMu_fit_pt_absy_mass"] = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                            hist.Bin("fit_absy", r"$|y^{(fit)}|$", np.array(np.arange(0., 2.25+0.25, 0.125))),
                                                            #hist.Bin("fit_absy", r"$|y^{(fit)}|$", 50, 0.0, 2.5),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 50, BD_MASS*0.9, BD_MASS*1.1)
                                                          )
    self._accumulator["BdToKPiMuMu_fit_pt_absy_mass_rwgt"] = hist.Hist("Events", dataset_axis, selection_axis_reco, 
                                                            hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                            hist.Bin("fit_absy", r"$|y^{(fit)}|$", np.array(np.arange(0., 2.25+0.25, 0.125))),
                                                            #hist.Bin("fit_absy", r"$|y^{(fit)}|$", 50, 0.0, 2.5),
                                                            hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 50, BD_MASS*0.9, BD_MASS*1.1)
                                                          )
    self._accumulator["BdToKPiMuMu_fit_pt"]             = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_eta"]            = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 100, -2.5, 2.5))
    self._accumulator["BdToKPiMuMu_fit_y"]              = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_y", r"$y^{(fit)}$", 100, -2.5, 2.5))
    self._accumulator["BdToKPiMuMu_fit_phi"]            = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BdToKPiMuMu_fit_best_mass"]      = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_fit_best_barmass"]   = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_barmass", r"Swap $m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["BdToKPiMuMu_chi2"]               = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_cos2D"]          = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BdToKPiMuMu_fit_theta2D"]        = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, 0., math.pi))
    self._accumulator["BdToKPiMuMu_l_xy"]               = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BdToKPiMuMu_l_xy_sig"]           = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 4.0))
    self._accumulator["BdToKPiMuMu_sv_prob"]            = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("sv_prob", r"SV prob", 50, 0.0, 1.0))
    self._accumulator["BdToKPiMuMu_fit_best_mkstar"]    = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_mkstar", r"$m_{K^{*}}^{(fit)}$ [GeV]", 100, KSTAR_892_MASS*0.7, KSTAR_892_MASS*1.3))
    self._accumulator["BdToKPiMuMu_fit_best_barmkstar"] = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("fit_barmkstar", r"Swap $m_{K^{*}}^{(fit)}$ [GeV]", 100, KSTAR_892_MASS*0.7, KSTAR_892_MASS*1.3))
    self._accumulator["BdToKPiMuMu_jpsi_mass"]          = hist.Hist("Events", dataset_axis, selection_axis_reco, hist.Bin("mass", r"$m(J/\psi)$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))


    self._accumulator["nTruthMuon"]  = hist.Hist("Events", dataset_axis, hist.Bin("nTruthMuon", r"N(truth muons)", 11, -0.5, 10.5))

    self._accumulator["TruthProbeMuon_parent"]      = hist.Hist("Events", dataset_axis, hist.Bin("parentPdgId", "Parent pdgId", 1001, -0.5, 1000.5))
    self._accumulator["TruthProbeMuon_grandparent"] = hist.Hist("Events", dataset_axis, hist.Bin("grandparentPdgId", "Grandparent pdgId", 1001, -0.5, 1000.5))

    self._accumulator["TruthBdToKPiMuMu_pt_absy_mass"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                            hist.Bin("pt", r"$p_{T}$ [GeV]", 100, 0.0, 100.0),
                                                            hist.Bin("absy", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.125))),
                                                            hist.Bin("mass", r"$m$ [GeV]", 50, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["TruthBdToKPiMuMu_pt_absy_mass_rwgt"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
                                                            hist.Bin("pt", r"$p_{T}$ [GeV]", 100, 0.0, 100.0),
                                                            hist.Bin("absy", r"$|y|$", np.array(np.arange(0., 2.25+0.25, 0.125))),
                                                            hist.Bin("mass", r"$m$ [GeV]", 50, BD_MASS*0.9, BD_MASS*1.1))
    self._accumulator["TruthBdToKPiMuMu_pt"]               = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("pt", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBdToKPiMuMu_eta"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("eta", r"$\eta$", 100, -2.5, 2.5))
    self._accumulator["TruthBdToKPiMuMu_y"]                = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("y", "y", 100, -2.5, 2.5))
    self._accumulator["TruthBdToKPiMuMu_phi"]              = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("phi", r"$\phi$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["TruthBdToKPiMuMu_mass"]             = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("mass", r"$p_{T}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["TruthBdToKPiMuMu_recopt_d_truthpt"] = hist.Hist("Events", dataset_axis, selection_axis_truth, hist.Bin("recopt_d_truthpt", r"$m$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1))
    '''
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
    '''

    # One entry per truth B
    # - If truth B is not matched to reco, or if reco fails selection, fill (-1, truthpt)
    self._accumulator["TruthBdToKPiMuMu_truthpt_recopt"] = hist.Hist("Events", dataset_axis, selection_axis_truth,
                                              hist.Bin("reco_pt", r"$p_{T}^{(reco)}$ [GeV]", 250, 0., 50.),
                                              hist.Bin("truth_pt", r"$p_{T}^{(truth)}$ [GeV]", 250, 0., 50.))

    #self._accumulator["TruthBdToKPiMuMu_k_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(K^{\pm})$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(K^{\pm})$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(K^{\pm})$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(K^{\pm})$", 30, K_MASS*0.5, K_MASS*2.0)
    #                                              )
    #self._accumulator["TruthBdToKPiMuMu_pi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(\pi^{\pm})$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(\pi^{\pm})$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(\pi^{\pm})$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(\pi^{\pm})$", 30, PI_MASS*0.5, PI_MASS*2.0)
    #                                              )
    #self._accumulator["TruthBdToKPiMuMu_mup_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(\mu^{+})$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(\mu^{+})$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(\mu^{+})$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(\mu^{+})$", 30, MUON_MASS*0.5, MUON_MASS*2.0)
    #                                              )
    #self._accumulator["TruthBdToKPiMuMu_mum_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(\mu^{-})$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(\mu^{-})$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(\mu^{-})$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(\mu^{-})$", 30, MUON_MASS*0.5, MUON_MASS*2.0)
    #                                              )
    #self._accumulator["TruthBdToKPiMuMu_kstar_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(K^{*}(892))$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(K^{*}(892))$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(K^{*}(892))$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(K^{*}(892))$", 30, KSTAR_892_MASS*0.5, KSTAR_892_MASS*2.0)
    #                                              )
    #self._accumulator["TruthBdToKPiMuMu_jpsi_p4"] = hist.Hist("Events", dataset_axis, selection_axis_truth, 
    #                                              hist.Bin("pt", r"$p_{T}(J/\psi)$ [GeV]", 100, 0.0, 100.0),
    #                                              hist.Bin("eta", r"$\eta(J/\psi)$", 20, -5., 5.),
    #                                              hist.Bin("phi", r"$\phi(J/\psi)$", 20, -1*math.pi, math.pi),
    #                                              hist.Bin("mass", r"$m(J/\psi)$", 30, JPSI_1S_MASS*0.5, JPSI_1S_MASS*2.0)
    #                                              )


  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    #ts1 = time.time()
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    output["nevents"][dataset_name] += df.size
    #ts2 = time.time()
    #print("Time 1 = {}".format(ts2-ts1))

    # Create jagged object arrays
    #ts1 = time.time()
    reco_bdkpimumu = dataframereader.reco_bdkpimumu(df, is_mc=True)
    reco_muons     = dataframereader.reco_muons(df, is_mc=True)
    trigger_muons  = dataframereader.trigger_muons(df, is_mc=True)
    probe_tracks   = dataframereader.probe_tracks(df, is_mc=True)
    genparts       = dataframereader.genparts(df, is_mc=True)
    #ts2 = time.time()
    #print("Time 2 = {}".format(ts2-ts1))

    #ts1 = time.time()
    reco_bdkpimumu.add_attributes(
      l1_softId    = reco_muons.softId[reco_bdkpimumu.l1_idx],
      l1_softMvaId = reco_muons.softMvaId[reco_bdkpimumu.l1_idx],
      l2_softId    = reco_muons.softId[reco_bdkpimumu.l2_idx],
      l2_softMvaId = reco_muons.softMvaId[reco_bdkpimumu.l2_idx],
      l1_mediumId    = reco_muons.mediumId[reco_bdkpimumu.l1_idx],
      l2_mediumId    = reco_muons.mediumId[reco_bdkpimumu.l2_idx],
      trk1_charge  = probe_tracks.charge[reco_bdkpimumu.trk1_idx],
      trk2_charge  = probe_tracks.charge[reco_bdkpimumu.trk2_idx],
      k_pt = where(reco_bdkpimumu.nominal_kpi, 
                    probe_tracks.pt[reco_bdkpimumu.trk1_idx],
                    probe_tracks.pt[reco_bdkpimumu.trk2_idx]),
      pi_pt = where(reco_bdkpimumu.nominal_kpi, 
                    probe_tracks.pt[reco_bdkpimumu.trk2_idx],
                    probe_tracks.pt[reco_bdkpimumu.trk1_idx]),
    )
    #ts2 = time.time()
    #print("Time 3 = {}".format(ts2-ts1))

    # Truth matching
    #ts1 = time.time()
    reco_bdkpimumu.l1_genIdx   = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.l1_genIdx[reco_bdkpimumu.l1_idx >= 0] = reco_muons.genPartIdx[reco_bdkpimumu.l1_idx] 
    reco_bdkpimumu.l2_genIdx   = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.l2_genIdx[reco_bdkpimumu.l2_idx >= 0] = reco_muons.genPartIdx[reco_bdkpimumu.l2_idx]
    reco_bdkpimumu.trk1_genIdx = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.trk1_genIdx[reco_bdkpimumu.trk1_idx >= 0] = probe_tracks.genPartIdx[reco_bdkpimumu.trk1_idx]
    #print(reco_bdkpimumu[reco_bdkpimumu.trk2_idx.count() > 0].trk2_idx)
    reco_bdkpimumu.trk2_genIdx = reco_bdkpimumu.fit_pt.ones_like().astype(int) * -1
    reco_bdkpimumu.trk2_genIdx[reco_bdkpimumu.trk2_idx >= 0] = probe_tracks.genPartIdx[reco_bdkpimumu.trk2_idx]
    #ts2 = time.time()
    #print("Time 4 = {}".format(ts2-ts1))

    #ts1 = time.time()
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
    #ts2 = time.time()
    #print("Time 5 = {}".format(ts2-ts1))

    # MC match to trk1 = K, trk2 = pi
    #ts1 = time.time()
    mcmatch_hypo1 = (abs(reco_bdkpimumu.l1_genMotherPdgId) == 443) \
                        & (abs(reco_bdkpimumu.l2_genMotherPdgId) == 443) \
                        & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                        & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                        & (abs(reco_bdkpimumu.trk1_pdgId) == 321) & (abs(reco_bdkpimumu.trk2_pdgId) == 211) \
                        & (abs(reco_bdkpimumu.trk1_genMotherPdgId) == 313) \
                        & (abs(reco_bdkpimumu.trk2_genMotherPdgId) == 313) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.l2_genGrandmotherIdx) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk1_genGrandmotherIdx) \
                        & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk2_genGrandmotherIdx) 
    # MC match to trk1=pi, trk2=K
    mcmatch_hypo2 = (abs(reco_bdkpimumu.l1_genMotherPdgId) == 443) \
                       & (abs(reco_bdkpimumu.l2_genMotherPdgId) == 443) \
                       & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                       & (abs(reco_bdkpimumu.l2_genGrandmotherPdgId) == 511) \
                       & (abs(reco_bdkpimumu.trk1_pdgId) == 211) & (abs(reco_bdkpimumu.trk2_pdgId) == 321) \
                       & (abs(reco_bdkpimumu.trk1_genMotherPdgId) == 313) \
                       & (abs(reco_bdkpimumu.trk2_genMotherPdgId) == 313) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.l2_genGrandmotherIdx) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk1_genGrandmotherIdx) \
                       & (reco_bdkpimumu.l1_genGrandmotherIdx == reco_bdkpimumu.trk2_genGrandmotherIdx) 
    #ts2 = time.time()
    #print("Time 6 = {}".format(ts2-ts1))

    #ts1 = time.time()
    reco_bdkpimumu.mcmatch = where(reco_bdkpimumu.nominal_kpi, mcmatch_hypo1, mcmatch_hypo2).astype(np.bool)
    reco_bdkpimumu.mcmatch_swap = where(reco_bdkpimumu.nominal_kpi, mcmatch_hypo2, mcmatch_hypo1).astype(np.bool)

    reco_bdkpimumu.genPartIdx = where((reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap), reco_bdkpimumu.l1_genGrandmotherIdx, -1)

    reco_bdkpimumu.add_attributes(
      wgt_trkeff = reweight_trkpt(reco_bdkpimumu.trk1pt_fullfit) * reweight_trkpt(reco_bdkpimumu.trk2pt_fullfit),
      wgt_fonll  = fonll_sfs["pt"](reco_bdkpimumu.fit_pt),
    )

    #ts2 = time.time()
    #print("Time 7 = {}".format(ts2-ts1))

    # Tag/probe selection
    """
    reco_bdkpimumu.add_attributes(Muon1IsTrig = reco_muons.isTriggeringFull[reco_bdkpimumu.l1_idx], 
                                  Muon2IsTrig = reco_muons.isTriggeringFull[reco_bdkpimumu.l2_idx])
    reco_bdkpimumu.add_attributes(MuonIsTrigCount = reco_bdkpimumu.Muon1IsTrig.astype(int) + reco_bdkpimumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggeringFull.astype(int).sum()
    reco_bdkpimumu.add_attributes(TagCount = reco_bdkpimumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bdkpimumu.MuonIsTrigCount)
    """
    #ts1 = time.time()
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
    tagmuon_etacuts = {
     "HLT_Mu7_IP4": 1.5,
     "HLT_Mu9_IP5": 1.5,
     "HLT_Mu9_IP6": 1.5,
     "HLT_Mu12_IP6": 1.5,
    }
    for trigger in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6"]:
      reco_bdkpimumu.add_attributes(**{
        f"Muon1IsTrig_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bdkpimumu.l1_idx],
        f"Muon2IsTrig_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bdkpimumu.l2_idx],
        f"Muon1IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bdkpimumu.l1_idx] \
                                        & (reco_muons.pt[reco_bdkpimumu.l1_idx] > tagmuon_ptcuts[trigger]) \
                                        & (abs(reco_muons.eta[reco_bdkpimumu.l1_idx]) < tagmuon_etacuts[trigger]) \
                                        & (abs(reco_muons.dxySig[reco_bdkpimumu.l1_idx]) > tagmuon_ipcuts[trigger]),
        f"Muon2IsTrigTight_{trigger}": getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_bdkpimumu.l2_idx] \
                                        & (reco_muons.pt[reco_bdkpimumu.l2_idx] > tagmuon_ptcuts[trigger]) \
                                        & (abs(reco_muons.eta[reco_bdkpimumu.l2_idx]) < tagmuon_etacuts[trigger]) \
                                        & (abs(reco_muons.dxySig[reco_bdkpimumu.l2_idx]) > tagmuon_ipcuts[trigger]),
      })
      #reco_bdkpimumu.add_attributes(**{
      #  f"Muon1IsTrigMaxPt_{trigger}": getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") & (reco_muons.pt[reco_bdkpimumu.l1_idx] == reco_muons.pt.max()),
      #  f"Muon2IsTrigMaxPt_{trigger}": getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}") & (reco_muons.pt[reco_bdkpimumu.l2_idx] == reco_muons.pt.max()),
      #})
      reco_bdkpimumu.add_attributes(**{
        f"MuonIsTrigCount_{trigger}": getattr(reco_bdkpimumu, f"Muon1IsTrig_{trigger}").astype(int) + getattr(reco_bdkpimumu, f"Muon2IsTrig_{trigger}").astype(int)
      })
      event_ntriggingmuons = getattr(reco_muons, f"isTriggeringFull_{trigger}").astype(int).sum()
      reco_bdkpimumu.add_attributes(**{
        f"TagCount_{trigger}": getattr(reco_bdkpimumu, f"MuonIsTrigCount_{trigger}").ones_like() * event_ntriggingmuons - getattr(reco_bdkpimumu, f"MuonIsTrigCount_{trigger}")
      })

      #reco_bdkpimumu.add_attributes(**{
      #  f"MuonIsTrigCountMaxPt_{trigger}": getattr(reco_bdkpimumu, f"Muon1IsTrigMaxPt_{trigger}").astype(int) + getattr(reco_bdkpimumu, f"Muon2IsTrigMaxPt_{trigger}").astype(int)
      #})
      #event_ntriggeringmuons_maxpt =  getattr(reco_muons, f"isTriggeringFull_{trigger}")[reco_muons.pt == reco_muons.pt.max()].astype(int).sum()
      #reco_bdkpimumu.add_attributes(**{
      #  f"TagCountMaxPt_{trigger}": getattr(reco_bdkpimumu, f"MuonIsTrigCountMaxPt_{trigger}").ones_like() * event_ntriggeringmuons_maxpt - getattr(reco_bdkpimumu, f"MuonIsTrigCountMaxPt_{trigger}"),
      #})

    reco_bdkpimumu.add_attributes(l_xy_sig = where(reco_bdkpimumu.l_xy_unc > 0, reco_bdkpimumu.l_xy / reco_bdkpimumu.l_xy_unc, -1.e20))
    #ts2 = time.time()
    #print("Time 8 = {}".format(ts2-ts1))

    # General selection
    #ts1 = time.time()
    reco_bdkpimumu_mask_template = reco_bdkpimumu.pt.ones_like().astype(bool)
    selections = {}
    selections["sv_pt"]    = (reco_bdkpimumu.fit_pt > final_cuts["Bd"]["sv_pt"])
    selections["absy"]     = (abs(reco_bdkpimumu.fit_best_y) < 2.25)
    selections["l_xy_sig"] = (abs(reco_bdkpimumu.l_xy_sig) > final_cuts["Bd"]["l_xy_sig"])
    selections["sv_prob"]  = (reco_bdkpimumu.sv_prob > final_cuts["Bd"]["sv_prob"])
    selections["cos2D"]    = (reco_bdkpimumu.fit_cos2D > final_cuts["Bd"]["cos2D"])
    selections["l1"]       = (reco_bdkpimumu.lep1pt_fullfit > final_cuts["Bd"]["l1_pt"]) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l1_softId)
    selections["l2"]       = (reco_bdkpimumu.lep2pt_fullfit > final_cuts["Bd"]["l2_pt"]) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_softId)
    #selections["l2"]       = (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_softId)
    #selections["l2"]       = (selections["l2"] & where(abs(reco_bdkpimumu.lep2eta_fullfit) < 1.4, 
    #                                                  (reco_bdkpimumu.lep2pt_fullfit > 1.5), 
    #                                                  (reco_bdkpimumu.lep2pt_fullfit > 1.0))).astype(bool)
    selections["trk1"]     = (reco_bdkpimumu.trk1pt_fullfit > final_cuts["Bd"]["k1_pt"]) & (abs(reco_bdkpimumu.trk1eta_fullfit) < 2.5)
    selections["trk2"]     = (reco_bdkpimumu.trk2pt_fullfit > final_cuts["Bd"]["k2_pt"]) & (abs(reco_bdkpimumu.trk2eta_fullfit) < 2.5)
    selections["trk1HiPt"]     = (reco_bdkpimumu.trk1pt_fullfit > 1.2) & (abs(reco_bdkpimumu.trk1eta_fullfit) < 2.5)
    selections["trk2HiPt"]     = (reco_bdkpimumu.trk2pt_fullfit > 1.2) & (abs(reco_bdkpimumu.trk2eta_fullfit) < 2.5)
    selections["l1HiPt"]       = (reco_bdkpimumu.lep1pt_fullfit > 3.0) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l1_softId)
    selections["l2HiPt"]       = (reco_bdkpimumu.lep2pt_fullfit > 3.0) & (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_softId)
    selections["l1MediumPt"]       = (reco_bdkpimumu.lep1pt_fullfit > 2.5) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l1_softId)
    selections["l2MediumPt"]       = (reco_bdkpimumu.lep2pt_fullfit > 2.5) & (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_softId)
    selections["l1HugePt"]       = (reco_bdkpimumu.lep1pt_fullfit > 3.25) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l1_softId)
    selections["l2HugePt"]       = (reco_bdkpimumu.lep2pt_fullfit > 3.25) & (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_softId)

    selections["l1VarPt"]       = (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) \
                                    & (reco_bdkpimumu.l1_softId) \
                                    & varmuonpt(reco_bdkpimumu.lep1pt_fullfit, abs(reco_bdkpimumu.lep1eta_fullfit))
    selections["l2VarPt"]       = (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) \
                                    & (reco_bdkpimumu.l2_softId) \
                                    & varmuonpt(reco_bdkpimumu.lep2pt_fullfit, abs(reco_bdkpimumu.lep2eta_fullfit))
    selections["l1Medium"]       = (reco_bdkpimumu.lep1pt_fullfit > final_cuts["Bd"]["l1_pt"]) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l1_mediumId)
    selections["l2Medium"]       = (reco_bdkpimumu.lep2pt_fullfit > final_cuts["Bd"]["l2_pt"]) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & (reco_bdkpimumu.l2_mediumId)

    #selections["dR"]       = (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.trk2phi_fullfit) > 0.03) \
    #                            & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
    #                            & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
    #                            & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
    #                            & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
    #                            & (delta_r(reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.lep1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03)
    selections["jpsi"]     = (abs(reco_bdkpimumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW)
    selections["kstar"]    = (abs(reco_bdkpimumu.mkstar_best_fullfit - KSTAR_892_MASS) < final_cuts["Bd"]["kstar_window"])  & (reco_bdkpimumu.trk1_charge + reco_bdkpimumu.trk2_charge == 0)
    selections["phi_veto"] = (abs(reco_bdkpimumu.phi_mass - PHI_1020_MASS) > final_cuts["Bd"]["phi_veto"])
    #ts2 = time.time()
    #print("Time 9 = {}".format(ts2-ts1))

    # Final selections
    #ts1 = time.time()
    selections["inclusive"]  = reco_bdkpimumu.fit_pt.ones_like().astype(bool)
    selections["reco"]       = selections["sv_pt"] \
                              & selections["absy"] \
                              & selections["l_xy_sig"] \
                              & selections["sv_prob"] \
                              & selections["cos2D"] \
                              & selections["l1"] \
                              & selections["l2"] \
                              & selections["trk1"] \
                              & selections["trk2"] \
                              & selections["jpsi"] \
                              & selections["kstar"] \
                              & selections["phi_veto"] \
                              #& selections["dR"] \
    selections["truthmatched"]  = (reco_bdkpimumu.genPartIdx >= 0)
    selections["recomatch"]     = selections["reco"] & reco_bdkpimumu.mcmatch
    selections["recomatchswap"] = selections["reco"] & reco_bdkpimumu.mcmatch_swap
    selections["unmatched"] = (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))
    #ts2 = time.time()
    #print("Time 10 = {}".format(ts2-ts1))

    #ts1 = time.time()
    for trigger in self._triggers:      
      trigger_mask = ((df[trigger] == 1) & (df[l1_seeds[trigger]] == 1)) * reco_bdkpimumu_mask_template #  
      selections[f"recomatch_{trigger}"]     = selections["recomatch"] & trigger_mask
      selections[f"recomatchswap_{trigger}"] = selections["recomatchswap"] & trigger_mask

      selections[f"tag_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}"))
      selections[f"tagmatch_{trigger}"]       = selections[f"tag_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagmatchswap_{trigger}"]   = selections[f"tag_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagunmatched_{trigger}"]   = selections[f"tag_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probe_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1)
      selections[f"probematch_{trigger}"]     = selections[f"probe_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probematchswap_{trigger}"] = selections[f"probe_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeunmatched_{trigger}"]   = selections[f"probe_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"tagx_{trigger}"]          = selections[f"tag_{trigger}"] & ~selections[f"probe_{trigger}"]
      selections[f"tagxmatch_{trigger}"]     = selections[f"tagx_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagxmatchswap_{trigger}"] = selections[f"tagx_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagxunmatched_{trigger}"] = selections[f"tagx_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"recomatch_{trigger}"]     = selections["recomatch"] & trigger_mask
      selections[f"recomatchswap_{trigger}"] = selections["recomatchswap"] & trigger_mask

      #selections[f"tagHiTrkPt_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
      #                                                  & selections["trk1HiPt"] & selections["trk2HiPt"]
      #selections[f"tagHiTrkPtmatch_{trigger}"]       = selections[f"tagHiTrkPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      #selections[f"tagHiTrkPtmatchswap_{trigger}"]   = selections[f"tagHiTrkPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      #selections[f"tagHiTrkPtunmatched_{trigger}"]   = selections[f"tagHiTrkPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      #selections[f"probeHiTrkPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
      #                                                  & selections["trk1HiPt"] & selections["trk2HiPt"]
      #selections[f"probeHiTrkPtmatch_{trigger}"]     = selections[f"probeHiTrkPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      #selections[f"probeHiTrkPtmatchswap_{trigger}"] = selections[f"probeHiTrkPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      #selections[f"probeHiTrkPtunmatched_{trigger}"]   = selections[f"probeHiTrkPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))


      selections[f"tagHiMuonPt_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
                                                        & selections["l1HiPt"] & selections["l2HiPt"]
      selections[f"tagHiMuonPtmatch_{trigger}"]       = selections[f"tagHiMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagHiMuonPtmatchswap_{trigger}"]   = selections[f"tagHiMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagHiMuonPtunmatched_{trigger}"]   = selections[f"tagHiMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probeHiMuonPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
                                                        & selections["l1HiPt"] & selections["l2HiPt"]
      selections[f"probeHiMuonPtmatch_{trigger}"]     = selections[f"probeHiMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probeHiMuonPtmatchswap_{trigger}"] = selections[f"probeHiMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeHiMuonPtunmatched_{trigger}"]   = selections[f"probeHiMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))


      selections[f"tagMediumMuonPt_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
                                                        & selections["l1MediumPt"] & selections["l2MediumPt"]
      selections[f"tagMediumMuonPtmatch_{trigger}"]       = selections[f"tagMediumMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagMediumMuonPtmatchswap_{trigger}"]   = selections[f"tagMediumMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagMediumMuonPtunmatched_{trigger}"]   = selections[f"tagMediumMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probeMediumMuonPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
                                                        & selections["l1MediumPt"] & selections["l2MediumPt"]
      selections[f"probeMediumMuonPtmatch_{trigger}"]     = selections[f"probeMediumMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probeMediumMuonPtmatchswap_{trigger}"] = selections[f"probeMediumMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeMediumMuonPtunmatched_{trigger}"]   = selections[f"probeMediumMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))


      selections[f"tagHugeMuonPt_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
                                                        & selections["l1HugePt"] & selections["l2HugePt"]
      selections[f"tagHugeMuonPtmatch_{trigger}"]       = selections[f"tagHugeMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagHugeMuonPtmatchswap_{trigger}"]   = selections[f"tagHugeMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagHugeMuonPtunmatched_{trigger}"]   = selections[f"tagHugeMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probeHugeMuonPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
                                                        & selections["l1HugePt"] & selections["l2HugePt"]
      selections[f"probeHugeMuonPtmatch_{trigger}"]     = selections[f"probeHugeMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probeHugeMuonPtmatchswap_{trigger}"] = selections[f"probeHugeMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeHugeMuonPtunmatched_{trigger}"]   = selections[f"probeHugeMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"tagVarMuonPt_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
                                                        & selections["l1VarPt"] & selections["l2VarPt"]
      selections[f"tagVarMuonPtmatch_{trigger}"]       = selections[f"tagVarMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagVarMuonPtmatchswap_{trigger}"]   = selections[f"tagVarMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagVarMuonPtunmatched_{trigger}"]   = selections[f"tagVarMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probeVarMuonPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
                                                        & selections["l1VarPt"] & selections["l2VarPt"]
      selections[f"probeVarMuonPtmatch_{trigger}"]     = selections[f"probeVarMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probeVarMuonPtmatchswap_{trigger}"] = selections[f"probeVarMuonPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeVarMuonPtunmatched_{trigger}"]   = selections[f"probeVarMuonPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))


      selections[f"tagMediumMuon_{trigger}"]            = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") | getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}")) \
                                                        & selections["l1Medium"] & selections["l2Medium"]
      selections[f"tagMediumMuonmatch_{trigger}"]       = selections[f"tagMediumMuon_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"tagMediumMuonmatchswap_{trigger}"]   = selections[f"tagMediumMuon_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"tagMediumMuonunmatched_{trigger}"]   = selections[f"tagMediumMuon_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      selections[f"probeMediumMuon_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCount_{trigger}") >= 1) \
                                                        & selections["l1Medium"] & selections["l2Medium"]
      selections[f"probeMediumMuonmatch_{trigger}"]     = selections[f"probeMediumMuon_{trigger}"] & (reco_bdkpimumu.mcmatch)
      selections[f"probeMediumMuonmatchswap_{trigger}"] = selections[f"probeMediumMuon_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      selections[f"probeMediumMuonunmatched_{trigger}"]   = selections[f"probeMediumMuon_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      #selections[f"tagMaxPt_{trigger}"]          = selections["reco"] & trigger_mask & \
      #                                                (
      #                                                  (getattr(reco_bdkpimumu, f"Muon1IsTrigTight_{trigger}") & (reco_muons.pt[reco_bdkpimumu.l1_idx] == reco_muons.pt.max())) \
      #                                                  | (getattr(reco_bdkpimumu, f"Muon2IsTrigTight_{trigger}") & (reco_muons.pt[reco_bdkpimumu.l2_idx] == reco_muons.pt.max()))
      #                                                )
      #selections[f"tagMaxPtmatch_{trigger}"]     = selections[f"tagMaxPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      #selections[f"tagMaxPtmatchswap_{trigger}"] = selections[f"tagMaxPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      #selections[f"tagMaxPtunmatched_{trigger}"] = selections[f"tagMaxPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))

      #selections[f"probeMaxPt_{trigger}"]          = selections["reco"] & trigger_mask & (getattr(reco_bdkpimumu, f"TagCountMaxPt_{trigger}") >= 1)
      #selections[f"probeMaxPtmatch_{trigger}"]     = selections[f"probeMaxPt_{trigger}"] & (reco_bdkpimumu.mcmatch)
      #selections[f"probeMaxPtmatchswap_{trigger}"] = selections[f"probeMaxPt_{trigger}"] & (reco_bdkpimumu.mcmatch_swap)
      #selections[f"probeMaxPtunmatched_{trigger}"] = selections[f"probeMaxPt_{trigger}"] & (~(reco_bdkpimumu.mcmatch | reco_bdkpimumu.mcmatch_swap))


      # If more than one B is selected, choose best chi2
      selections[f"recomatch_{trigger}"]     = selections[f"recomatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"recomatch_{trigger}"]].chi2.min())
      selections[f"recomatchswap_{trigger}"] = selections[f"recomatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"recomatchswap_{trigger}"]].chi2.min())

      selections[f"tag_{trigger}"]            = selections[f"tag_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tag_{trigger}"]].chi2.min())
      selections[f"tagmatch_{trigger}"]       = selections[f"tagmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagmatch_{trigger}"]].chi2.min())
      selections[f"tagmatchswap_{trigger}"]   = selections[f"tagmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagmatchswap_{trigger}"]].chi2.min())
      selections[f"tagunmatched_{trigger}"]   = selections[f"tagunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagunmatched_{trigger}"]].chi2.min())

      selections[f"probe_{trigger}"]          = selections[f"probe_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probe_{trigger}"]].chi2.min())
      selections[f"probematch_{trigger}"]     = selections[f"probematch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probematch_{trigger}"]].chi2.min())
      selections[f"probematchswap_{trigger}"] = selections[f"probematchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probematchswap_{trigger}"]].chi2.min())
      selections[f"probeunmatched_{trigger}"]   = selections[f"probeunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeunmatched_{trigger}"]].chi2.min())

      selections[f"tagx_{trigger}"]            = selections[f"tagx_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagx_{trigger}"]].chi2.min())
      selections[f"tagxmatch_{trigger}"]       = selections[f"tagxmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagxmatch_{trigger}"]].chi2.min())
      selections[f"tagxmatchswap_{trigger}"]   = selections[f"tagxmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagxmatchswap_{trigger}"]].chi2.min())
      selections[f"tagxunmatched_{trigger}"]   = selections[f"tagxunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagxunmatched_{trigger}"]].chi2.min())

      #selections[f"tagHiTrkPt_{trigger}"]            = selections[f"tagHiTrkPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiTrkPt_{trigger}"]].chi2.min())
      #selections[f"tagHiTrkPtmatch_{trigger}"]       = selections[f"tagHiTrkPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiTrkPtmatch_{trigger}"]].chi2.min())
      #selections[f"tagHiTrkPtmatchswap_{trigger}"]   = selections[f"tagHiTrkPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiTrkPtmatchswap_{trigger}"]].chi2.min())
      #selections[f"tagHiTrkPtunmatched_{trigger}"]   = selections[f"tagHiTrkPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiTrkPtunmatched_{trigger}"]].chi2.min())

      #selections[f"probeHiTrkPt_{trigger}"]          = selections[f"probeHiTrkPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiTrkPt_{trigger}"]].chi2.min())
      #selections[f"probeHiTrkPtmatch_{trigger}"]     = selections[f"probeHiTrkPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiTrkPtmatch_{trigger}"]].chi2.min())
      #selections[f"probeHiTrkPtmatchswap_{trigger}"] = selections[f"probeHiTrkPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiTrkPtmatchswap_{trigger}"]].chi2.min())
      #selections[f"probeHiTrkPtunmatched_{trigger}"]   = selections[f"probeHiTrkPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiTrkPtunmatched_{trigger}"]].chi2.min())

      selections[f"tagHiMuonPt_{trigger}"]            = selections[f"tagHiMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiMuonPt_{trigger}"]].chi2.min())
      selections[f"tagHiMuonPtmatch_{trigger}"]       = selections[f"tagHiMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"tagHiMuonPtmatchswap_{trigger}"]   = selections[f"tagHiMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"tagHiMuonPtunmatched_{trigger}"]   = selections[f"tagHiMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHiMuonPtunmatched_{trigger}"]].chi2.min())

      selections[f"probeHiMuonPt_{trigger}"]          = selections[f"probeHiMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiMuonPt_{trigger}"]].chi2.min())
      selections[f"probeHiMuonPtmatch_{trigger}"]     = selections[f"probeHiMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"probeHiMuonPtmatchswap_{trigger}"] = selections[f"probeHiMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"probeHiMuonPtunmatched_{trigger}"]   = selections[f"probeHiMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHiMuonPtunmatched_{trigger}"]].chi2.min())


      selections[f"tagMediumMuonPt_{trigger}"]            = selections[f"tagMediumMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonPt_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonPtmatch_{trigger}"]       = selections[f"tagMediumMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonPtmatchswap_{trigger}"]   = selections[f"tagMediumMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonPtunmatched_{trigger}"]   = selections[f"tagMediumMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonPtunmatched_{trigger}"]].chi2.min())

      selections[f"probeMediumMuonPt_{trigger}"]          = selections[f"probeMediumMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonPt_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonPtmatch_{trigger}"]     = selections[f"probeMediumMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonPtmatchswap_{trigger}"] = selections[f"probeMediumMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonPtunmatched_{trigger}"]   = selections[f"probeMediumMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonPtunmatched_{trigger}"]].chi2.min())


      selections[f"tagHugeMuonPt_{trigger}"]            = selections[f"tagHugeMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHugeMuonPt_{trigger}"]].chi2.min())
      selections[f"tagHugeMuonPtmatch_{trigger}"]       = selections[f"tagHugeMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHugeMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"tagHugeMuonPtmatchswap_{trigger}"]   = selections[f"tagHugeMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHugeMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"tagHugeMuonPtunmatched_{trigger}"]   = selections[f"tagHugeMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagHugeMuonPtunmatched_{trigger}"]].chi2.min())

      selections[f"probeHugeMuonPt_{trigger}"]          = selections[f"probeHugeMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHugeMuonPt_{trigger}"]].chi2.min())
      selections[f"probeHugeMuonPtmatch_{trigger}"]     = selections[f"probeHugeMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHugeMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"probeHugeMuonPtmatchswap_{trigger}"] = selections[f"probeHugeMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHugeMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"probeHugeMuonPtunmatched_{trigger}"]   = selections[f"probeHugeMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeHugeMuonPtunmatched_{trigger}"]].chi2.min())


      selections[f"tagVarMuonPt_{trigger}"]            = selections[f"tagVarMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagVarMuonPt_{trigger}"]].chi2.min())
      selections[f"tagVarMuonPtmatch_{trigger}"]       = selections[f"tagVarMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagVarMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"tagVarMuonPtmatchswap_{trigger}"]   = selections[f"tagVarMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagVarMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"tagVarMuonPtunmatched_{trigger}"]   = selections[f"tagVarMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagVarMuonPtunmatched_{trigger}"]].chi2.min())

      selections[f"probeVarMuonPt_{trigger}"]          = selections[f"probeVarMuonPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeVarMuonPt_{trigger}"]].chi2.min())
      selections[f"probeVarMuonPtmatch_{trigger}"]     = selections[f"probeVarMuonPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeVarMuonPtmatch_{trigger}"]].chi2.min())
      selections[f"probeVarMuonPtmatchswap_{trigger}"] = selections[f"probeVarMuonPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeVarMuonPtmatchswap_{trigger}"]].chi2.min())
      selections[f"probeVarMuonPtunmatched_{trigger}"]   = selections[f"probeVarMuonPtunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeVarMuonPtunmatched_{trigger}"]].chi2.min())


      selections[f"tagMediumMuon_{trigger}"]            = selections[f"tagMediumMuon_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuon_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonmatch_{trigger}"]       = selections[f"tagMediumMuonmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonmatch_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonmatchswap_{trigger}"]   = selections[f"tagMediumMuonmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonmatchswap_{trigger}"]].chi2.min())
      selections[f"tagMediumMuonunmatched_{trigger}"]   = selections[f"tagMediumMuonunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMediumMuonunmatched_{trigger}"]].chi2.min())

      selections[f"probeMediumMuon_{trigger}"]          = selections[f"probeMediumMuon_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuon_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonmatch_{trigger}"]     = selections[f"probeMediumMuonmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonmatch_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonmatchswap_{trigger}"] = selections[f"probeMediumMuonmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonmatchswap_{trigger}"]].chi2.min())
      selections[f"probeMediumMuonunmatched_{trigger}"]   = selections[f"probeMediumMuonunmatched_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMediumMuonunmatched_{trigger}"]].chi2.min())

      #selections[f"tagMaxPt_{trigger}"]            = selections[f"tagMaxPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMaxPt_{trigger}"]].chi2.min())
      #selections[f"tagMaxPtmatch_{trigger}"]       = selections[f"tagMaxPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMaxPtmatch_{trigger}"]].chi2.min())
      #selections[f"tagMaxPtmatchswap_{trigger}"]   = selections[f"tagMaxPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"tagMaxPtmatchswap_{trigger}"]].chi2.min())

      #selections[f"probeMaxPt_{trigger}"]          = selections[f"probeMaxPt_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMaxPt_{trigger}"]].chi2.min())
      #selections[f"probeMaxPtmatch_{trigger}"]     = selections[f"probeMaxPtmatch_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMaxPtmatch_{trigger}"]].chi2.min())
      #selections[f"probeMaxPtmatchswap_{trigger}"] = selections[f"probeMaxPtmatchswap_{trigger}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu[selections[f"probeMaxPtmatchswap_{trigger}"]].chi2.min())

    #ts2 = time.time()
    #print("Time 11 = {}".format(ts2-ts1))

    # Fill cutflow
    #ts1 = time.time()
    cumulative_selection = reco_bdkpimumu.pt.ones_like().astype(bool)
    output["reco_cutflow"][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
    for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "trk1", "trk2", "jpsi", "kstar", "phi_veto"]: # dR
      cumulative_selection = cumulative_selection & selections[cut_name]
      output["reco_cutflow"][dataset_name][cut_name] += cumulative_selection.sum().sum()
    output["reco_cutflow"][dataset_name]["tag"] += selections[f"tag_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["tagmatch"] += selections[f"tagmatch_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["tagmatchswap"] += selections[f"tagmatchswap_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["tagunmatched"] += selections[f"tagunmatched_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probe"] += selections["probe_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probematch"] += selections["probematch_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probematchswap"] += selections["probematchswap_HLT_Mu7_IP4"].sum().sum()
    output["reco_cutflow"][dataset_name]["probeunmatched"] += selections["probeunmatched_HLT_Mu7_IP4"].sum().sum()
    #ts2 = time.time()
    #print("Time 12 = {}".format(ts2-ts1))

    # Fill reco histograms
    #ts1 = time.time()
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    #output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    selection_names = ["inclusive", "reco", "recomatch", "recomatchswap", "truthmatched", "unmatched"]
    for trigger in self._triggers:
      selection_names.extend([f"recomatch_{trigger}", f"recomatchswap_{trigger}", 
                              f"tag_{trigger}", f"tagmatch_{trigger}", f"tagmatchswap_{trigger}", f"tagunmatched_{trigger}", 
                              f"tagx_{trigger}", f"tagxmatch_{trigger}", f"tagxmatchswap_{trigger}", f"tagxunmatched_{trigger}", 
                              f"probe_{trigger}", f"probematch_{trigger}", f"probematchswap_{trigger}", f"probeunmatched_{trigger}", 
                              #f"tagHiTrkPt_{trigger}", f"tagHiTrkPtmatch_{trigger}", f"tagHiTrkPtmatchswap_{trigger}", f"tagHiTrkPtunmatched_{trigger}", 
                              #f"probeHiTrkPt_{trigger}", f"probeHiTrkPtmatch_{trigger}", f"probeHiTrkPtmatchswap_{trigger}", f"probeHiTrkPtunmatched_{trigger}",
                              f"tagHiMuonPt_{trigger}", f"tagHiMuonPtmatch_{trigger}", f"tagHiMuonPtmatchswap_{trigger}", f"tagHiMuonPtunmatched_{trigger}", 
                              f"probeHiMuonPt_{trigger}", f"probeHiMuonPtmatch_{trigger}", f"probeHiMuonPtmatchswap_{trigger}", f"probeHiMuonPtunmatched_{trigger}", 

                              f"tagMediumMuonPt_{trigger}", f"tagMediumMuonPtmatch_{trigger}", f"tagMediumMuonPtmatchswap_{trigger}", f"tagMediumMuonPtunmatched_{trigger}", 
                              f"probeMediumMuonPt_{trigger}", f"probeMediumMuonPtmatch_{trigger}", f"probeMediumMuonPtmatchswap_{trigger}", f"probeMediumMuonPtunmatched_{trigger}", 

                              f"tagHugeMuonPt_{trigger}", f"tagHugeMuonPtmatch_{trigger}", f"tagHugeMuonPtmatchswap_{trigger}", f"tagHugeMuonPtunmatched_{trigger}", 
                              f"probeHugeMuonPt_{trigger}", f"probeHugeMuonPtmatch_{trigger}", f"probeHugeMuonPtmatchswap_{trigger}", f"probeHugeMuonPtunmatched_{trigger}", 

                              f"tagVarMuonPt_{trigger}", f"tagVarMuonPtmatch_{trigger}", f"tagVarMuonPtmatchswap_{trigger}", f"tagVarMuonPtunmatched_{trigger}", 
                              f"probeVarMuonPt_{trigger}", f"probeVarMuonPtmatch_{trigger}", f"probeVarMuonPtmatchswap_{trigger}", f"probeVarMuonPtunmatched_{trigger}", 

                              f"tagMediumMuon_{trigger}", f"tagMediumMuonmatch_{trigger}", f"tagMediumMuonmatchswap_{trigger}", f"tagMediumMuonunmatched_{trigger}", 
                              f"probeMediumMuon_{trigger}", f"probeMediumMuonmatch_{trigger}", f"probeMediumMuonmatchswap_{trigger}", f"probeMediumMuonunmatched_{trigger}", 
                              ])
    for selection_name in selection_names:
      output["BdToKPiMuMu_fit_pt_absy_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bdkpimumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_absy=np.abs(reco_bdkpimumu.fit_best_y[selections[selection_name]].flatten()),
                                            fit_mass=reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_pt_absy_mass_rwgt"].fill(dataset=dataset_name, selection=selection_name, 
                                            fit_pt=reco_bdkpimumu.fit_pt[selections[selection_name]].flatten(),
                                            fit_absy=np.abs(reco_bdkpimumu.fit_best_y[selections[selection_name]].flatten()),
                                            fit_mass=reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten(), 
                                            weight=reco_bdkpimumu.wgt_fonll[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bdkpimumu.fit_pt[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bdkpimumu.fit_eta[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_y"].fill(dataset=dataset_name, selection=selection_name, fit_y=reco_bdkpimumu.fit_best_y[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bdkpimumu.fit_phi[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_barmass"].fill(dataset=dataset_name, selection=selection_name, fit_barmass=reco_bdkpimumu.fit_best_barmass[selections[selection_name]].flatten())
      output["BdToKPiMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bdkpimumu.chi2[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten())
      output["BdToKPiMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bdkpimumu.l_xy[selections[selection_name]].flatten())
      output["BdToKPiMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bdkpimumu.l_xy_sig[selections[selection_name]].flatten())
      output["BdToKPiMuMu_sv_prob"].fill(dataset=dataset_name, selection=selection_name, sv_prob=reco_bdkpimumu.sv_prob[selections[selection_name]].flatten())
      output["BdToKPiMuMu_jpsi_mass"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.mll_fullfit[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_mkstar"].fill(dataset=dataset_name, selection=selection_name, fit_mkstar=reco_bdkpimumu.mkstar_best_fullfit[selections[selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_barmkstar"].fill(dataset=dataset_name, selection=selection_name, fit_barmkstar=reco_bdkpimumu.barmkstar_best_fullfit[selections[selection_name]].flatten())
    #ts2 = time.time()
    #print("Time 13 = {}".format(ts2-ts1))

    # Build gen-to-reco map
    #ts1 = time.time()
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

    mask_mup_frombd = (genparts.pdgId == 13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 511)
    mup_frombd_grandmother_idx = genpart_grandmother_idx[mask_mup_frombd]
    bd_mup_idx = genparts.pt.ones_like().astype(int) * -1
    bd_mup_idx[mup_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_mup_frombd]

    mask_mum_frombd = (genparts.pdgId == -13) & (abs(genpart_mother_pdgId) == 443) & (abs(genpart_grandmother_pdgId) == 511)
    mum_frombd_grandmother_idx = genpart_grandmother_idx[mask_mum_frombd]
    bd_mum_idx = genparts.pt.ones_like().astype(int) * -1
    bd_mum_idx[mum_frombd_grandmother_idx] = genparts.pdgId.localindex[mask_mum_frombd]
    
    mask_kstar_frombd = (abs(genparts.pdgId) == 313) & (abs(genpart_mother_pdgId) == 511)
    kstar_frombd_mother_idx = genpart_mother_idx[mask_kstar_frombd]
    bd_kstar_idx = genparts.pt.ones_like().astype(int) * -1
    bd_kstar_idx[kstar_frombd_mother_idx] = genparts.pdgId.localindex[mask_kstar_frombd]

    mask_jpsi_frombd = (abs(genparts.pdgId) == 443) & (abs(genpart_mother_pdgId) == 511)
    jpsi_frombd_mother_idx = genpart_mother_idx[mask_jpsi_frombd]
    bd_jpsi_idx = genparts.pt.ones_like().astype(int) * -1
    bd_jpsi_idx[jpsi_frombd_mother_idx] = genparts.pdgId.localindex[mask_jpsi_frombd]
    #ts2 = time.time()
    #print("Time 14 = {}".format(ts2-ts1))

    # Count number of soft photon daughters
    #ts1 = time.time()
    nChildrenNoPhoton = genparts.nChildren - count_photon_children(genparts.genPartIdxMother, genparts.pdgId, genparts.pt)

    # Jagged array of truth BdToKPiMuMus
    mask511 = (abs(genparts.pdgId)==511) & (bd_jpsi_idx >= 0) & (bd_kstar_idx >= 0) & (bd_mup_idx >= 0) & (bd_mum_idx >= 0) & (bd_k_idx >= 0) & (bd_pi_idx >= 0) & (nChildrenNoPhoton == 2)
    #mask511 = mask511 & (mask511.sum() <= 1)
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
      wgt_trkeff   = genparts.pt[mask511].ones_like().flatten(), 
      wgt_fonll    = genparts.pt[mask511].ones_like().flatten(), 
    )
    truth_bdkpimumu.add_attributes(
      y=np.log((np.sqrt(truth_bdkpimumu.mass**2 
        + truth_bdkpimumu.pt**2*np.cosh(truth_bdkpimumu.eta)**2) 
        + truth_bdkpimumu.pt*np.sinh(truth_bdkpimumu.eta)) / np.sqrt(truth_bdkpimumu.mass**2 
        + truth_bdkpimumu.pt**2))
      )
    #ts2 = time.time()
    #print("Time 15 = {}".format(ts2-ts1))

    # Compute invariant mass of K* and J/psi
    #inv_mass = (genparts[truth_bdkpimumu.mup_idx].p4 + genparts[truth_bdkpimumu.mum_idx].p4 + genparts[truth_bdkpimumu.k_idx].p4 + genparts[truth_bdkpimumu.pi_idx].p4).mass
    #print("DEBUG : Invariant mass cut efficiency")
    #print( truth_bdkpimumu.pt[abs(inv_mass - BD_MASS) < 0.15].count().sum() / truth_bdkpimumu.pt.count().sum())
    #print(inv_mass.flatten()[:200])
    #print(inv_mass[abs(inv_mass - BD_MASS) > 0.15].flatten()[:100])
    #truth_bdkpimumu = truth_bdkpimumu[abs(inv_mass - BD_MASS) < 0.15]

    # Truth selections
    #ts1 = time.time()
    truth_selections = {}
    truth_selections["inclusive"] = truth_bdkpimumu.pt.ones_like().astype(bool)

    # Fiducial selection: match reco cuts
    truth_selections["fiducial"] =  (truth_bdkpimumu.pt > 3.0) & (abs(truth_bdkpimumu.y) < 2.25)
    #truth_selections["fiducial"] =  (genparts.pt[truth_bdkpimumu.gen_idx] > 3.0) \
    #                                & (
    #                                    ((genparts.pt[truth_bdkpimumu.k_idx] > 0.85) & (genparts.pt[truth_bdkpimumu.pi_idx] > 0.6)) \
    #                                    | ((genparts.pt[truth_bdkpimumu.pi_idx] > 0.85) & (genparts.pt[truth_bdkpimumu.k_idx] > 0.6))
    #                                  ) \
    #                                & (abs(genparts.eta[truth_bdkpimumu.k_idx]) < 2.5) & (abs(genparts.eta[truth_bdkpimumu.pi_idx]) < 2.5) \
    #                                & (genparts.pt[truth_bdkpimumu.mup_idx] > 1.5) & (abs(genparts.eta[truth_bdkpimumu.mup_idx]) < 2.4) \
    #                                & (genparts.pt[truth_bdkpimumu.mum_idx] > 1.5) & (abs(genparts.eta[truth_bdkpimumu.mum_idx]) < 2.4) \

    # Matching: a bit more complicated for Bd
    truth_selections["matched"] = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched"][truth_bdkpimumu.recoIdx >= 0] = reco_bdkpimumu.mcmatch[truth_bdkpimumu.recoIdx[truth_bdkpimumu.recoIdx >= 0]]
    truth_selections["matched_swap"] = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap"][truth_bdkpimumu.recoIdx >= 0] = reco_bdkpimumu.mcmatch_swap[truth_bdkpimumu.recoIdx[truth_bdkpimumu.recoIdx >= 0]]
    truth_selections["unmatched"]       = ~(truth_selections["matched"] | truth_selections["matched_swap"])

    truth_selections["matched_sel"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_sel"][truth_selections["matched"]]   = selections["reco"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

    for trigger in self._triggers:
      truth_selections[f"matched_tag_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tag_{trigger}"][truth_selections["matched"]]   = selections[f"tag_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probe_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probe_{trigger}"][truth_selections["matched"]] = selections[f"probe_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_tagx_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagx_{trigger}"][truth_selections["matched"]] = selections[f"tagx_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tag_{trigger}"] = truth_selections[f"matched_tag_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probe_{trigger}"] = truth_selections[f"matched_probe_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_tagx_{trigger}"] = truth_selections[f"matched_tagx_{trigger}"] & truth_selections["fiducial"]

      # Hi track pT
      #truth_selections[f"matched_tagHiTrkPt_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      #truth_selections[f"matched_tagHiTrkPt_{trigger}"][truth_selections["matched"]]   = selections[f"tagHiTrkPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      #truth_selections[f"matched_probeHiTrkPt_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      #truth_selections[f"matched_probeHiTrkPt_{trigger}"][truth_selections["matched"]] = selections[f"probeHiTrkPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      #truth_selections[f"matched_fid_tagHiTrkPt_{trigger}"] = truth_selections[f"matched_tagHiTrkPt_{trigger}"] & truth_selections["fiducial"]
      #truth_selections[f"matched_fid_probeHiTrkPt_{trigger}"] = truth_selections[f"matched_probeHiTrkPt_{trigger}"] & truth_selections["fiducial"]

      # Hi muon pT
      truth_selections[f"matched_tagHiMuonPt_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagHiMuonPt_{trigger}"][truth_selections["matched"]]   = selections[f"tagHiMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probeHiMuonPt_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probeHiMuonPt_{trigger}"][truth_selections["matched"]] = selections[f"probeHiMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tagHiMuonPt_{trigger}"] = truth_selections[f"matched_tagHiMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probeHiMuonPt_{trigger}"] = truth_selections[f"matched_probeHiMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_tagMediumMuonPt_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagMediumMuonPt_{trigger}"][truth_selections["matched"]]   = selections[f"tagMediumMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probeMediumMuonPt_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probeMediumMuonPt_{trigger}"][truth_selections["matched"]] = selections[f"probeMediumMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tagMediumMuonPt_{trigger}"] = truth_selections[f"matched_tagMediumMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probeMediumMuonPt_{trigger}"] = truth_selections[f"matched_probeMediumMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_tagHugeMuonPt_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagHugeMuonPt_{trigger}"][truth_selections["matched"]]   = selections[f"tagHugeMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probeHugeMuonPt_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probeHugeMuonPt_{trigger}"][truth_selections["matched"]] = selections[f"probeHugeMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tagHugeMuonPt_{trigger}"] = truth_selections[f"matched_tagHugeMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probeHugeMuonPt_{trigger}"] = truth_selections[f"matched_probeHugeMuonPt_{trigger}"] & truth_selections["fiducial"]

      truth_selections[f"matched_tagVarMuonPt_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagVarMuonPt_{trigger}"][truth_selections["matched"]]   = selections[f"tagVarMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probeVarMuonPt_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probeVarMuonPt_{trigger}"][truth_selections["matched"]] = selections[f"probeVarMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tagVarMuonPt_{trigger}"] = truth_selections[f"matched_tagVarMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probeVarMuonPt_{trigger}"] = truth_selections[f"matched_probeVarMuonPt_{trigger}"] & truth_selections["fiducial"]

      truth_selections[f"matched_tagMediumMuon_{trigger}"]                                = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_tagMediumMuon_{trigger}"][truth_selections["matched"]]   = selections[f"tagMediumMuon_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]
      truth_selections[f"matched_probeMediumMuon_{trigger}"]                              = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_probeMediumMuon_{trigger}"][truth_selections["matched"]] = selections[f"probeMediumMuon_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched"]]]

      truth_selections[f"matched_fid_tagMediumMuon_{trigger}"] = truth_selections[f"matched_tagMediumMuon_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_fid_probeMediumMuon_{trigger}"] = truth_selections[f"matched_probeMediumMuon_{trigger}"] & truth_selections["fiducial"]

    truth_selections["matched_swap_sel"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
    truth_selections["matched_swap_sel"][truth_selections["matched_swap"]]   = selections["reco"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
    for trigger in self._triggers:
      truth_selections[f"matched_swap_tag_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tag_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tag_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probe_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probe_{trigger}"][truth_selections["matched_swap"]] = selections[f"probe_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_tagx_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagx_{trigger}"][truth_selections["matched_swap"]] = selections[f"tagx_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tag_{trigger}"] = truth_selections[f"matched_swap_tag_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probe_{trigger}"] = truth_selections[f"matched_swap_probe_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_tagx_{trigger}"] = truth_selections[f"matched_swap_tagx_{trigger}"] & truth_selections["fiducial"]

      # High track pT
      #truth_selections[f"matched_swap_tagHiTrkPt_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      #truth_selections[f"matched_swap_tagHiTrkPt_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagHiTrkPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      #truth_selections[f"matched_swap_probeHiTrkPt_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      #truth_selections[f"matched_swap_probeHiTrkPt_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeHiTrkPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      #truth_selections[f"matched_swap_fid_tagHiTrkPt_{trigger}"] = truth_selections[f"matched_swap_tagHiTrkPt_{trigger}"] & truth_selections["fiducial"]
      #truth_selections[f"matched_swap_fid_probeHiTrkPt_{trigger}"] = truth_selections[f"matched_swap_probeHiTrkPt_{trigger}"] & truth_selections["fiducial"]

      # High muon pT
      truth_selections[f"matched_swap_tagHiMuonPt_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagHiMuonPt_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagHiMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probeHiMuonPt_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probeHiMuonPt_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeHiMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tagHiMuonPt_{trigger}"] = truth_selections[f"matched_swap_tagHiMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probeHiMuonPt_{trigger}"] = truth_selections[f"matched_swap_probeHiMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_swap_tagMediumMuonPt_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagMediumMuonPt_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagMediumMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probeMediumMuonPt_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probeMediumMuonPt_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeMediumMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tagMediumMuonPt_{trigger}"] = truth_selections[f"matched_swap_tagMediumMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probeMediumMuonPt_{trigger}"] = truth_selections[f"matched_swap_probeMediumMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_swap_tagHugeMuonPt_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagHugeMuonPt_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagHugeMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probeHugeMuonPt_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probeHugeMuonPt_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeHugeMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tagHugeMuonPt_{trigger}"] = truth_selections[f"matched_swap_tagHugeMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probeHugeMuonPt_{trigger}"] = truth_selections[f"matched_swap_probeHugeMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_swap_tagVarMuonPt_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagVarMuonPt_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagVarMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probeVarMuonPt_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probeVarMuonPt_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeVarMuonPt_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tagVarMuonPt_{trigger}"] = truth_selections[f"matched_swap_tagVarMuonPt_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probeVarMuonPt_{trigger}"] = truth_selections[f"matched_swap_probeVarMuonPt_{trigger}"] & truth_selections["fiducial"]


      truth_selections[f"matched_swap_tagMediumMuon_{trigger}"]                                     = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_tagMediumMuon_{trigger}"][truth_selections["matched_swap"]]   = selections[f"tagMediumMuon_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]
      truth_selections[f"matched_swap_probeMediumMuon_{trigger}"]                                   = truth_bdkpimumu.recoIdx.zeros_like().astype(bool)
      truth_selections[f"matched_swap_probeMediumMuon_{trigger}"][truth_selections["matched_swap"]] = selections[f"probeMediumMuon_{trigger}"][truth_bdkpimumu.recoIdx[truth_selections["matched_swap"]]]

      truth_selections[f"matched_swap_fid_tagMediumMuon_{trigger}"] = truth_selections[f"matched_swap_tagMediumMuon_{trigger}"] & truth_selections["fiducial"]
      truth_selections[f"matched_swap_fid_probeMediumMuon_{trigger}"] = truth_selections[f"matched_swap_probeMediumMuon_{trigger}"] & truth_selections["fiducial"]

    truth_bdkpimumu.recomatch_pt[truth_selections["matched"] | truth_selections["matched_swap"]] = reco_bdkpimumu.fit_pt[truth_bdkpimumu.recoIdx[truth_selections["matched"] | truth_selections["matched_swap"]]]
    truth_bdkpimumu.wgt_trkeff[truth_selections["matched"] | truth_selections["matched_swap"]] = reweight_trkpt(reco_bdkpimumu.trk1pt_fullfit[truth_bdkpimumu.recoIdx[truth_selections["matched"] | truth_selections["matched_swap"]]])  \
                                                                                                  * reweight_trkpt(reco_bdkpimumu.trk2pt_fullfit[truth_bdkpimumu.recoIdx[truth_selections["matched"] | truth_selections["matched_swap"]]])
    truth_bdkpimumu.wgt_fonll[truth_selections["matched"] | truth_selections["matched_swap"]] = fonll_sfs["pt"](truth_bdkpimumu.pt[truth_selections["matched"] | truth_selections["matched_swap"]])

    #ts2 = time.time()
    #print("Time 16 = {}".format(ts2-ts1))

    # Truth "cutflow"
    #ts1 = time.time()
    truth_selection_names = ["inclusive", "fiducial", "matched", "unmatched"] # matched_sel
    for trigger in self._triggers:
      #truth_selection_names.extend([f"matched_tag_{trigger}", f"matched_probe_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tag_{trigger}", f"matched_fid_probe_{trigger}", f"matched_fid_tagx_{trigger}"])
      #truth_selection_names.extend([f"matched_tagHiTrkPt_{trigger}", f"matched_probeHiTrkPt_{trigger}"])
      #truth_selection_names.extend([f"matched_fid_tagHiTrkPt_{trigger}", f"matched_fid_probeHiTrkPt_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tagHiMuonPt_{trigger}", f"matched_fid_probeHiMuonPt_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tagMediumMuonPt_{trigger}", f"matched_fid_probeMediumMuonPt_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tagHugeMuonPt_{trigger}", f"matched_fid_probeHugeMuonPt_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tagVarMuonPt_{trigger}", f"matched_fid_probeVarMuonPt_{trigger}"])
      truth_selection_names.extend([f"matched_fid_tagMediumMuon_{trigger}", f"matched_fid_probeMediumMuon_{trigger}"])
    for truth_selection_name in truth_selection_names:
      output["truth_cutflow"][dataset_name][truth_selection_name] = truth_selections[truth_selection_name].sum().sum()
    #ts2 = time.time()
    #print("Time 17 = {}".format(ts2-ts1))

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
    #ts1 = time.time()
    truth_muons = genparts[abs(genparts.pdgId) == 13]
    truth_muons_probefilter = (abs(truth_muons.eta) <= 2.5) \
                              & (truth_muons.pt >= 5.0) \
                              & ~(
                                  (abs(genparts.pdgId[truth_muons.genPartIdxMother]) == 443) \
                                  & (abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons.genPartIdxMother]]) == 511)
                                  & (bd_kstar_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (bd_k_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (bd_pi_idx[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] >= 0) \
                                  & (nChildrenNoPhoton[genparts.genPartIdxMother[truth_muons.genPartIdxMother]] == 2) \
                                  )
    truth_muons_mufilter = (abs(truth_muons.eta) <= 2.5) \
                              & (truth_muons.pt >= 5.0)

    event_probefilter = (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3)
    #for x in zip(truth_muons[~event_probefilter].pt.count(), truth_muons[~event_probefilter].pdgId, truth_muons[~event_probefilter].pt, truth_muons[~event_probefilter].eta, genparts[~event_probefilter].pdgId[truth_muons[~event_probefilter].genPartIdxMother]):
    #  print("Muon info in event failing probefilter:")
    #  print(x)
    truth_selections["probefilter"] = (truth_bdkpimumu.pt.ones_like() * ( \
                                        (truth_muons_probefilter.sum() >= 1) & (truth_muons.pt.count() >= 3) \
                                      )).astype(bool)
    truth_selections["mufilter"] = (truth_bdkpimumu.pt.ones_like() * ( \
                                        (truth_muons_mufilter.sum() >= 1) & (truth_muons.pt.count() >= 3) \
                                      )).astype(bool)

    truth_selections["probefilter_fiducial"] = truth_selections["probefilter"] & truth_selections["fiducial"]
    truth_selections["mufilter_fiducial"] = truth_selections["mufilter"] & truth_selections["fiducial"]

    #ts2 = time.time()
    #print("Time 18 = {}".format(ts2-ts1))

    #ts1 = time.time()
    self._accumulator["TruthProbeMuon_parent"].fill(
      dataset=dataset_name, 
      parentPdgId=abs(genparts.pdgId[truth_muons[truth_muons_probefilter].genPartIdxMother].flatten())
    )
    self._accumulator["TruthProbeMuon_grandparent"].fill(
      dataset=dataset_name, 
      grandparentPdgId=abs(genparts.pdgId[genparts.genPartIdxMother[truth_muons[truth_muons_probefilter].genPartIdxMother]].flatten())
    )
    #ts2 = time.time()
    #print("Time 19 = {}".format(ts2-ts1))

    # Fill truth histograms
    #ts1 = time.time()
    for selection_name in truth_selections.keys(): # truth_selection_names:
      output["TruthBdToKPiMuMu_pt_absy_mass"].fill(
        dataset=dataset_name, 
        selection=selection_name, 
        pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten(),
        absy=np.abs(truth_bdkpimumu.y[truth_selections[selection_name]].flatten()),
        mass=truth_bdkpimumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_pt_absy_mass_rwgt"].fill(
        dataset=dataset_name, 
        selection=selection_name, 
        pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten(),
        absy=np.abs(truth_bdkpimumu.y[truth_selections[selection_name]].flatten()),
        mass=truth_bdkpimumu.mass[truth_selections[selection_name]].flatten(), 
        weight=truth_bdkpimumu.wgt_fonll[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_pt"].fill(dataset=dataset_name, selection=selection_name, pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_eta"].fill(dataset=dataset_name, selection=selection_name, eta=truth_bdkpimumu.eta[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_y"].fill(dataset=dataset_name, selection=selection_name, y=truth_bdkpimumu.y[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_phi"].fill(dataset=dataset_name, selection=selection_name, phi=truth_bdkpimumu.phi[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_mass"].fill(dataset=dataset_name, selection=selection_name, mass=truth_bdkpimumu.mass[truth_selections[selection_name]].flatten())
      output["TruthBdToKPiMuMu_recopt_d_truthpt"].fill(dataset=dataset_name, selection=selection_name, 
          recopt_d_truthpt=where(truth_selections["matched"].flatten(), 
                                  ((truth_bdkpimumu.recomatch_pt / truth_bdkpimumu.pt)).flatten(),
                                  -1.0)[truth_selections[selection_name].flatten()])
      output["TruthBdToKPiMuMu_truthpt_recopt"].fill(dataset=dataset_name, selection=selection_name,
          reco_pt=truth_bdkpimumu.recomatch_pt[truth_selections[selection_name]].flatten(), 
          truth_pt=truth_bdkpimumu.pt[truth_selections[selection_name]].flatten())

      '''
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
      '''
    output["nTruthMuon"].fill(dataset=dataset_name, nTruthMuon=genparts[abs(genparts.pdgId)==13].pt.count())
    #ts2 = time.time()
    #print("Time 20 = {}".format(ts2-ts1))

    # Tree outputs
    #ts1 = time.time()
    for selection_name in self._Bcand_selections:
      output[f"Bcands_{selection_name}"][dataset_name].extend({
        "pt": reco_bdkpimumu.fit_pt[selections[selection_name]].flatten(),
        #"eta": reco_bdkpimumu.fit_eta[selections[selection_name]].flatten(),
        "y": reco_bdkpimumu.fit_best_y[selections[selection_name]].flatten(),
        "phi": reco_bdkpimumu.fit_phi[selections[selection_name]].flatten(),
        "mass": reco_bdkpimumu.fit_best_mass[selections[selection_name]].flatten(),
        #"l_xy": reco_bdkpimumu.l_xy[selections[selection_name]].flatten(),
        #"l_xy_unc": reco_bdkpimumu.l_xy_unc[selections[selection_name]].flatten(),
        #"sv_prob": reco_bdkpimumu.sv_prob[selections[selection_name]].flatten(),
        #"cos2D": reco_bdkpimumu.fit_cos2D[selections[selection_name]].flatten(),
      })
    #ts2 = time.time()
    #print("Time 21 = {}".format(ts2-ts1))

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Make histograms for B FFR data")
  parser.add_argument("-d", "--datasets", type=str, help="List of datasets, see in_txt. Comma-separated.")
  parser.add_argument("-j", "--subjob", type=int, nargs=2, help="Run subjob i/j")
  parser.add_argument("-w", "--workers", type=int, default=8, help="Workers")
  parser.add_argument("--condor", action="store_true", help="Flag for running on condor")
  parser.add_argument("--retar_venv", action="store_true", help="Retar venv (takes a while)")
  args = parser.parse_args()

  # Inputs
  in_txt = {
    #"Bd2KstarJpsi2KPiMuMu_probefilter_noconstr":"/home/dyu7/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_6/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    #"Bd2KstarJpsi2KPiMuMu_inclusive_noconstr":"/home/dyu7/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_6/files_BdToJpsiKstar_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    #"Bd2KstarJpsi2KPiMuMu_probefilter":"/home/dyu7/dryu/BFrag/boffea/barista/filelists/v2_5_2/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.dat",
    #"Bd2KsJpsi2KPiMuMu_probefilter":"/home/dyu7/dryu/BFrag/boffea/barista/filelists/v2_6/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    #"Bd2KsJpsi2KPiMuMu_inclusive":"/home/dyu7/dryu/BFrag/boffea/barista/filelists/v2_6/files_BdToJpsiKstar_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KsJpsi2KPiMuMu_probefilter":"/home/dyu7/dryu/BFrag/boffea/barista/filelists/frozen/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KsJpsi2KPiMuMu_inclusive": "/home/dyu7/dryu/BFrag/boffea/barista/filelists/frozen/files_BdToJpsiKstar_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KsJpsi2KPiMuMu_mufilter":"/home/dyu7/dryu/BFrag/boffea/barista/filelists/frozen/files_BdToKstarJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
  }
  if args.datasets:
    datasets_to_run = args.datasets.split(",")
  else:
    datasets_to_run = in_txt.keys()

  dataset_files = {}
  for dataset_name, filelistpath in in_txt.items():
    with open(filelistpath, 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

      if args.subjob:
        i_subjob = args.subjob[0]
        n_subjobs = args.subjob[1]
        total_files = len(dataset_files[dataset_name])
        files_per_job = int(math.ceil(total_files / n_subjobs))

        ijob_min = files_per_job * i_subjob
        ijob_max = min(files_per_job * (i_subjob+1) - 1, total_files)
        print("DEBUG : For subjob {}/{}, running files {}-{} out of {}".format(i_subjob, n_subjobs, ijob_min, ijob_max, total_files))
        dataset_files[dataset_name] = dataset_files[dataset_name][ijob_min:ijob_max+1]

        subjob_tag = "subjob{}d{}".format(i_subjob, n_subjobs)

  if args.condor:
    # Copy input files to worker node... seems to fail sporadically when reading remote :(
    local_filelist = []
    for remote_file in dataset_files[dataset_name]:
      retry_counter = 0
      expected_path = f"{os.path.expandvars('$_CONDOR_SCRATCH_DIR')}/{os.path.basename(remote_file)}"
      while retry_counter < 5 and not (os.path.isfile(expected_path) and os.path.getsize(expected_path) > 1.e6):
        if retry_counter >= 1:
          time.sleep(10)
        os.system(f"cp {remote_file} $_CONDOR_SCRATCH_DIR")
        retry_counter += 1
      if not (os.path.isfile(expected_path) and os.path.getsize(expected_path) > 1.e6):
        raise RuntimeError("FATAL : Failed to copy file {}".format(remote_file))
      os.system("ls -lrth $_CONDOR_SCRATCH_DIR")
      local_filelist.append(f"{os.path.expandvars('$_CONDOR_SCRATCH_DIR')}/{os.path.basename(remote_file)}")
    dataset_files[dataset_name] = local_filelist

  ts_start = time.time()
  import psutil
  print("psutil.cpu_count() = {}".format(psutil.cpu_count()))
  output = processor.run_uproot_job(dataset_files,
                                treename='Events',
                                processor_instance=MCEfficencyProcessor(),
                                executor=processor.futures_executor,
                                executor_args={'workers': args.workers, 'flatten': False, 'status':not args.condor},
                                chunksize=50000,
                                #maxchunks=args.workers*2,
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

  if args.subjob:
    output_file = "MCEfficiencyHistograms_Bd_{}.coffea".format(subjob_tag)
  else:
    output_file = "MCEfficiencyHistograms_Bd.coffea"
  util.save(output, output_file)

  print("Total time: {} seconds".format(ts_end - ts_start))
  print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))
  print("Total nevents: {}".format(total_events))
