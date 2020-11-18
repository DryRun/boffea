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
import re

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

class DataProcessor(processor.ProcessorABC):
  def __init__(self):
    # Histograms
    dataset_axis = hist.Cat("dataset", "Primary dataset")
    selection_axis = hist.Cat("selection", "Selection name")

    self._accumulator = processor.dict_accumulator()
    self._accumulator["nevents"] = processor.defaultdict_accumulator(int)

    self._accumulator["run_ls_nevents"] = processor.defaultdict_accumulator(partial(processor))

    for side in ["tag", "probe"]:
      for trigger_name in ["inclusive", "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu9_IP5&~HLT_Mu7_IP4"]:
        selection_name = f"{side}_{trigger_name}"
        for bflavor in ["Bu", "Bd", "Bs"]:
          self._accumulator[f"Bcands_{bflavor}_{selection_name}"] = processor.defaultdict_accumulator(Bcand_accumulator) #, outputfile=f"tree_{selection_name}.root"))

    for trigger_name in ["inclusive", "HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu9_IP5&~HLT_Mu7_IP4"]:
      for bflavor in ["Bu", "Bd", "Bs"]:
        self._accumulator[f"reco_cutflow_{bflavor}_{trigger_name}"] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, int))

    self._accumulator["nMuon"]          = hist.Hist("Events", dataset_axis, hist.Bin("nMuon", r"Number of muons", 11,-0.5, 10.5))
    self._accumulator["nMuon_isTrig"]   = hist.Hist("Events", dataset_axis, hist.Bin("nMuon_isTrig", r"Number of triggering muons", 11,-0.5, 10.5))
    self._accumulator["Muon_pt"]        = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt", r"Muon $p_{T}$ [GeV]", 100, 0.0, 100.0))
    self._accumulator["Muon_pt_isTrig"] = hist.Hist("Events", dataset_axis, hist.Bin("Muon_pt_isTrig", r"Triggering muon $p_{T}$ [GeV]", 100, 0.0, 100.0))

    self._accumulator["BuToKMuMu_fit_pt_y_mass"] = hist.Hist("Events", dataset_axis, selection_axis, 
                                                  hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                  hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                  hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.9, BU_MASS*1.1)
                                                )
    self._accumulator["BuToKMuMu_fit_pt"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BuToKMuMu_fit_eta"]     = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BuToKMuMu_fit_phi"]     = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BuToKMuMu_fit_mass"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.8, BU_MASS*1.2))
    self._accumulator["BuToKMuMu_chi2"]        = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BuToKMuMu_fit_cos2D"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BuToKMuMu_fit_theta2D"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, -1.*math.pi, 1.*math.pi))
    self._accumulator["BuToKMuMu_l_xy"]        = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BuToKMuMu_l_xy_sig"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 9.0))
    self._accumulator["BuToKMuMu_jpsi_m"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{J/\psi}$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["BuToKMuMu_fit_eta_vs_pt_vs_mass"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BU_MASS*0.8, BU_MASS*1.2), hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0), hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))

    self._accumulator["BsToKKMuMu_fit_pt_y_mass"] = hist.Hist("Events", dataset_axis, selection_axis, 
                                                       hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                       hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                       hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS*0.9, BS_MASS*1.1)
                                                     )
    self._accumulator["BsToKKMuMu_fit_pt"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_eta"]     = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BsToKKMuMu_fit_phi"]     = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BsToKKMuMu_fit_mass"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS * 0.8, BS_MASS * 1.2))
    self._accumulator["BsToKKMuMu_chi2"]        = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_fit_cos2D"]   = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BsToKKMuMu_fit_theta2D"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, 0., 1.*math.pi))
    self._accumulator["BsToKKMuMu_l_xy"]        = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BsToKKMuMu_l_xy_sig"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 9.0))
    self._accumulator["BsToKKMuMu_phi_m"]       = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{\phi}$ [GeV]", 100, PHI_1020_MASS * 0.8, PHI_1020_MASS * 1.2))
    self._accumulator["BsToKKMuMu_jpsi_m"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{J/\psi}$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["BsToKKMuMu_fit_eta_vs_pt_vs_mass"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BS_MASS * 0.8, BS_MASS * 1.2), hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0), hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BsToKKMuMu_kp_pt"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("pt", r"$p_{T}(K^{+})$ [GeV]", 200, 0.0, 100.0))
    self._accumulator["BsToKKMuMu_km_pt"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("pt", r"$p_{T}(K^{-})$ [GeV]", 200, 0.0, 100.0))

    self._accumulator["NM1_BsToKKMuMu_phi_m"]    = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("mass", r"$m_{\phi}$ [GeV]", 100, PHI_1020_MASS * 0.8, PHI_1020_MASS * 1.2))
    self._accumulator["NM1_BsToKKMuMu_jpsi_m"]   = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("mass", r"$m_{J/\psi}$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["NM1_BsToKKMuMu_l_xy_sig"] = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 9.0))
    self._accumulator["NM1_BsToKKMuMu_sv_prob"]  = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("sv_prob", r"SV prob.", 50, 0., 1.))
    self._accumulator["NM1_BsToKKMuMu_cos2D"]    = hist.Hist("Events", dataset_axis, hist.Cat("selection", "Selection name"), hist.Bin("cos2D", r"SV cos2D.", 50, -1., 1.))

    self._accumulator["BdToKPiMuMu_fit_pt_y_mass"] = hist.Hist("Events", dataset_axis, selection_axis, 
                                                          hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0),
                                                          hist.Bin("fit_y", r"$y^{(fit)}$", 50, -5.0, 5.0),
                                                          hist.Bin("fit_mass", r"$m^{(fit)}$ [GeV]", 100, BD_MASS*0.9, BD_MASS*1.1)
                                                        )
    self._accumulator["BdToKPiMuMu_fit_pt"]           = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 500, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_eta"]          = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BdToKPiMuMu_fit_phi"]          = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_phi", r"$\phi^{(fit)}$", 50, -2.0*math.pi, 2.0*math.pi))
    self._accumulator["BdToKPiMuMu_fit_mass"]         = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ (nom.) [GeV]", 100, BD_MASS*0.8, BD_MASS*1.2))
    self._accumulator["BdToKPiMuMu_fit_barmass"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_barmass", r"$m^{(fit)}$ (alt.) [GeV]", 100, BD_MASS*0.8, BD_MASS*1.2))
    self._accumulator["BdToKPiMuMu_fit_best_mass"]    = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ (best) [GeV]", 100, BD_MASS*0.8, BD_MASS*1.2))
    self._accumulator["BdToKPiMuMu_fit_best_barmass"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_barmass", r"$m^{(fit)}$ (swap) [GeV]", 100, BD_MASS*0.8, BD_MASS*1.2))
    self._accumulator["BdToKPiMuMu_chi2"]             = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("chi2", r"Fit $\chi^{2}$", 100, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_fit_cos2D"]        = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_cos2D", r"Fit cos2D", 100, -1., 1.))
    self._accumulator["BdToKPiMuMu_fit_theta2D"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_theta2D", r"Fit $\theta_{2D}$", 100, -1.*math.pi, 1.*math.pi))
    self._accumulator["BdToKPiMuMu_l_xy"]             = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy", r"$L_{xy}$",50, -1.0, 4.0))
    self._accumulator["BdToKPiMuMu_l_xy_sig"]         = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("l_xy_sig", r"$L_{xy}/\sigma(L_{xy})$",50, -1.0, 9.0))
    self._accumulator["BdToKPiMuMu_kstar_m"]          = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{K^{*}}$ (nom.) [GeV]", 100, KSTAR_892_MASS * 0.8, KSTAR_892_MASS * 1.2))
    self._accumulator["BdToKPiMuMu_kstar_m_bar"]      = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{K^{*}}$ (alt.) [GeV]", 100, KSTAR_892_MASS * 0.8, KSTAR_892_MASS * 1.2))
    self._accumulator["BdToKPiMuMu_kstar_m_best"]     = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{K^{*}}$ (best) [GeV]", 100, KSTAR_892_MASS * 0.8, KSTAR_892_MASS * 1.2))
    self._accumulator["BdToKPiMuMu_jpsi_m"]           = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("mass", r"$m_{J/\psi}$ [GeV]", 100, JPSI_1S_MASS * 0.8, JPSI_1S_MASS * 1.2))
    self._accumulator["BdToKPiMuMu_fit_eta_vs_pt_vs_mass"] = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("fit_mass", r"$m^{(fit)}$ (nom.) [GeV]", 100, BD_MASS*0.8, BD_MASS*1.2), hist.Bin("fit_pt", r"$p_{T}^{(fit)}$ [GeV]", 100, 0.0, 100.0), hist.Bin("fit_eta", r"$\eta^{(fit)}$", 50, -5.0, 5.0))
    self._accumulator["BdToKPiMuMu_k_pt"]             = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("pt", r"$p_{T}(K^{\pm})$ [GeV]", 200, 0.0, 100.0))
    self._accumulator["BdToKPiMuMu_pi_pt"]            = hist.Hist("Events", dataset_axis, selection_axis, hist.Bin("pt", r"$p_{T}(\pi^{\pm})$ [GeV]", 200, 0.0, 100.0))


    # One entry per truth B
    branches = ["BToKMuMu_*", "BToPhiMuMu_*", "BToKsMuMu_*", "Muon_*", "ProbeTracks_*", "TrigObj_*", "TriggerMuon_*", "GenPart_*", 
                "nBToKMuMu",  "nMuon",  "nProbeTracks",  "nTrigObj",  "nTriggerMuon",  "nGenPart",
                "HLT_Mu7_IP4", "HLT_Mu8_IP6", "HLT_Mu8_IP5", "HLT_Mu8_IP3", "HLT_Mu8p5_IP3p5", "HLT_Mu9_IP6", "HLT_Mu9_IP5", "HLT_Mu9_IP4", "HLT_Mu10p5_IP3p5", "HLT_Mu12_IP6", 
                "L1_SingleMu7er1p5", "L1_SingleMu8er1p5", "L1_SingleMu9er1p5", "L1_SingleMu10er1p5", "L1_SingleMu12er1p5", "L1_SingleMu22",             
                "event"]

    self._re_subjob = re.compile("(?P<subjob_tag>_subjob\d+)")

  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    match_subjob = self._re_subjob.search(dataset_name)
    if match_subjob:
      dataset_name.replace(match_subjob.group("subjob_tag"), "")
    output["nevents"][dataset_name] += df.size

    

    # Create jagged object arrays
    reco_bskkmumu  = dataframereader.reco_bskkmumu(df)
    reco_bukmumu   = dataframereader.reco_bukmumu(df)
    reco_bdkpimumu = dataframereader.reco_bdkpimumu(df)
    reco_muons     = dataframereader.reco_muons(df)
    probe_tracks   = dataframereader.probe_tracks(df)

    # Add muon quality to B objects
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
    reco_bukmumu.add_attributes(
      l1_softId    = reco_muons.softId[reco_bukmumu.l1_idx],
      l1_softMvaId = reco_muons.softMvaId[reco_bukmumu.l1_idx],
      l2_softId    = reco_muons.softId[reco_bukmumu.l2_idx],
      l2_softMvaId = reco_muons.softMvaId[reco_bukmumu.l2_idx],
    )
    reco_bdkpimumu.add_attributes(
      l1_softId    = reco_muons.softId[reco_bdkpimumu.l1_idx],
      l1_softMvaId = reco_muons.softMvaId[reco_bdkpimumu.l1_idx],
      l2_softId    = reco_muons.softId[reco_bdkpimumu.l2_idx],
      l2_softMvaId = reco_muons.softMvaId[reco_bdkpimumu.l2_idx],
      trk1_charge  = probe_tracks.charge[reco_bdkpimumu.trk1_idx],
      trk2_charge  = probe_tracks.charge[reco_bdkpimumu.trk2_idx],
      k_pt = where(reco_bdkpimumu.nominal_kpi, 
                    probe_tracks.pt[reco_bdkpimumu.trk1_idx],
                    probe_tracks.pt[reco_bdkpimumu.trk2_idx]),
      pi_pt = where(reco_bdkpimumu.nominal_kpi, 
                    probe_tracks.pt[reco_bdkpimumu.trk2_idx],
                    probe_tracks.pt[reco_bdkpimumu.trk1_idx]),
    )

    # Tag/probe selection
    reco_bskkmumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggering[reco_bskkmumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggering[reco_bskkmumu.l2_idx])
    reco_bskkmumu.add_attributes(MuonIsTrigCount = reco_bskkmumu.Muon1IsTrig.astype(int) + reco_bskkmumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bskkmumu.add_attributes(TagCount = reco_bskkmumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bskkmumu.MuonIsTrigCount)

    reco_bskkmumu.add_attributes(l_xy_sig = where(reco_bskkmumu.l_xy_unc > 0, reco_bskkmumu.l_xy / reco_bskkmumu.l_xy_unc, -1.e20))

    reco_bukmumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggering[reco_bukmumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggering[reco_bukmumu.l2_idx])
    reco_bukmumu.add_attributes(MuonIsTrigCount = reco_bukmumu.Muon1IsTrig.astype(int) + reco_bukmumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bukmumu.add_attributes(TagCount = reco_bukmumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bukmumu.MuonIsTrigCount)

    reco_bukmumu.add_attributes(l_xy_sig = where(reco_bukmumu.l_xy_unc > 0, reco_bukmumu.l_xy / reco_bukmumu.l_xy_unc, -1.e20))

    reco_bdkpimumu.add_attributes(
                  Muon1IsTrig = reco_muons.isTriggering[reco_bdkpimumu.l1_idx], 
                  Muon2IsTrig = reco_muons.isTriggering[reco_bdkpimumu.l2_idx])
    reco_bdkpimumu.add_attributes(MuonIsTrigCount = reco_bdkpimumu.Muon1IsTrig.astype(int) + reco_bdkpimumu.Muon2IsTrig.astype(int))
    event_ntriggingmuons = reco_muons.isTriggering.astype(int).sum()
    reco_bdkpimumu.add_attributes(TagCount = reco_bdkpimumu.MuonIsTrigCount.ones_like() * event_ntriggingmuons - reco_bdkpimumu.MuonIsTrigCount)

    reco_bdkpimumu.add_attributes(l_xy_sig = where(reco_bdkpimumu.l_xy_unc > 0, reco_bdkpimumu.l_xy / reco_bdkpimumu.l_xy_unc, -1.e20))

    # General selection
    trigger_masks = {}
    trigger_masks["inclusive"] = np.full_like(df["HLT_Mu7_IP4"], True)
    trigger_masks["HLT_Mu7_IP4"] = (df["HLT_Mu7_IP4"] == 1)
    trigger_masks["HLT_Mu9_IP5"] = (df["HLT_Mu9_IP5"] == 1)
    trigger_masks["HLT_Mu9_IP6"] = (df["HLT_Mu9_IP6"] == 1)
    trigger_masks["HLT_Mu12_IP6"] = (df["HLT_Mu12_IP6"] == 1)
    trigger_masks["HLT_Mu9_IP5|HLT_Mu9_IP6"] = trigger_masks["HLT_Mu9_IP5"] | trigger_masks["HLT_Mu9_IP6"]
    trigger_masks["HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6"] = trigger_masks["HLT_Mu7_IP4"] | trigger_masks["HLT_Mu9_IP5"] | trigger_masks["HLT_Mu9_IP6"]
    trigger_masks["HLT_Mu9_IP5&~HLT_Mu7_IP4"] = trigger_masks["HLT_Mu9_IP5"] & ~trigger_masks["HLT_Mu7_IP4"]

    # - Bs to Jpsi phi to mu mu K K
    selections = {"Bu":{}, "Bs":{}, "Bd":{}}
    nm1_selections = {"Bu":{}, "Bs":{}, "Bd":{}}

    reco_bskkmumu_mask_template = reco_bskkmumu.pt.ones_like().astype(bool)
    selections["Bs"]["sv_pt"]       = (reco_bskkmumu.pt > 3.0)
    selections["Bs"]["l_xy_sig"] = (abs(reco_bskkmumu.l_xy_sig) > 3.5)
    selections["Bs"]["sv_prob"]     = (reco_bskkmumu.sv_prob > 0.1)
    selections["Bs"]["cos2D"]    = (reco_bskkmumu.fit_cos2D > 0.999)
    selections["Bs"]["l1"]         =  (reco_bskkmumu.l1_pt > 1.5) & (abs(reco_bskkmumu.l1_eta) < 2.4) & reco_bskkmumu.l1_softId
    selections["Bs"]["l2"]         =  (reco_bskkmumu.l2_pt > 1.5) & (abs(reco_bskkmumu.l2_eta) < 2.4) & reco_bskkmumu.l2_softId
    selections["Bs"]["k1"]         = (reco_bskkmumu.trk1_pt > 0.5) & (abs(reco_bskkmumu.trk1_eta) < 2.5)
    selections["Bs"]["k2"]         = (reco_bskkmumu.trk2_pt > 0.5) & (abs(reco_bskkmumu.trk2_eta) < 2.5)
    selections["Bs"]["dR"]         = (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.trk2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.trk2_phi) > 0.03) \
                                    & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l1_phi) > 0.03) \
                                    & (delta_r(reco_bskkmumu.trk1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk1_phi, reco_bskkmumu.l2_phi) > 0.03) \
                                    & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l1_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l1_phi) > 0.03) \
                                    & (delta_r(reco_bskkmumu.trk2_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.trk2_phi, reco_bskkmumu.l2_phi) > 0.03) \
                                    & (delta_r(reco_bskkmumu.l1_eta, reco_bskkmumu.l2_eta, reco_bskkmumu.l1_phi, reco_bskkmumu.l2_phi) > 0.03)
    selections["Bs"]["jpsi"]       = abs(reco_bskkmumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    selections["Bs"]["phi"]        = (abs(reco_bskkmumu.phi_m - PHI_1020_MASS) < PHI_WINDOW) & (reco_bskkmumu.trk1_charge + reco_bskkmumu.trk2_charge == 0)
    selections["Bs"]["kstar_veto"] = (abs(reco_bskkmumu.Kstar1_mass - KSTAR_892_MASS) > BS_KSTAR_VETO_WINDOW) \
                                & (abs(reco_bskkmumu.Kstar2_mass - KSTAR_892_MASS) > BS_KSTAR_VETO_WINDOW)
    #selections["Bs"]["trigger"]    = trigger_mask * reco_bskkmumu_mask_template

    # Final selections
    selections["Bs"]["inclusive"] = reco_bskkmumu.fit_pt.ones_like().astype(bool)
    selections["Bs"]["reco"] = selections["Bs"]["sv_pt"] \
                              & selections["Bs"]["l_xy_sig"] \
                              & selections["Bs"]["sv_prob"] \
                              & selections["Bs"]["cos2D"] \
                              & selections["Bs"]["l1"] \
                              & selections["Bs"]["l2"] \
                              & selections["Bs"]["k1"] \
                              & selections["Bs"]["k2"] \
                              & selections["Bs"]["dR"] \
                              & selections["Bs"]["jpsi"] \
                              & selections["Bs"]["phi"] \
                              & selections["Bs"]["kstar_veto"]
    for trigger_name, trigger_mask in trigger_masks.items():
      selections["Bs"][f"recotrig_{trigger_name}"]   = selections["Bs"]["reco"] & (trigger_mask * reco_bskkmumu_mask_template).astype(bool)
      selections["Bs"][f"tag_{trigger_name}"]   = selections["Bs"][f"recotrig_{trigger_name}"] & (reco_bskkmumu.Muon1IsTrig | reco_bskkmumu.Muon2IsTrig)
      selections["Bs"][f"tag_{trigger_name}"]   = selections["Bs"][f"tag_{trigger_name}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["Bs"][f"tag_{trigger_name}"]].min())
      selections["Bs"][f"probe_{trigger_name}"] = selections["Bs"][f"recotrig_{trigger_name}"] & (reco_bskkmumu.TagCount >= 1)
      selections["Bs"][f"probe_{trigger_name}"] = selections["Bs"][f"probe_{trigger_name}"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[selections["Bs"][f"probe_{trigger_name}"]].min())

    # Fill cutflows
    for trigger_strat in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu9_IP5&~HLT_Mu7_IP4"]:
      cutflow_name = f"reco_cutflow_Bs_{trigger_strat}"
      cumulative_selection = reco_bskkmumu.pt.ones_like().astype(bool)
      output[cutflow_name][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
      cumulative_selection = cumulative_selection & (trigger_masks[trigger_strat] * reco_bskkmumu_mask_template).astype(bool)
      output[cutflow_name][dataset_name]["trigger"] = cumulative_selection.sum().sum()
      for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "dR", "jpsi", "phi", "kstar_veto"]:
        cumulative_selection = cumulative_selection & selections["Bs"][cut_name]
        output[cutflow_name][dataset_name][cut_name] += cumulative_selection.sum().sum()
      output[cutflow_name][dataset_name]["tag"] += selections["Bs"][f"tag_{trigger_strat}"].sum().sum()
      output[cutflow_name][dataset_name]["probe"] += selections["Bs"][f"probe_{trigger_strat}"].sum().sum()

    # Bu to Jpsi K to mu mu K
    reco_bukmumu_mask_template = reco_bukmumu.pt.ones_like().astype(bool)
    selections["Bu"]["sv_pt"]       = (reco_bukmumu.pt > 3.0)
    selections["Bu"]["l_xy_sig"] = (abs(reco_bukmumu.l_xy_sig) > 3.5)
    selections["Bu"]["sv_prob"]     = (reco_bukmumu.sv_prob > 0.1)
    selections["Bu"]["cos2D"]    = (reco_bukmumu.fit_cos2D > 0.999)
    selections["Bu"]["l1"]      =  (reco_bukmumu.fit_l1_pt > 1.5) & (abs(reco_bukmumu.fit_l1_eta) < 2.4) & reco_bukmumu.l1_softId
    selections["Bu"]["l2"]      =  (reco_bukmumu.fit_l2_pt > 1.5) & (abs(reco_bukmumu.fit_l2_eta) < 2.4) & reco_bukmumu.l2_softId
    selections["Bu"]["k"]       = (reco_bukmumu.fit_k_pt > 0.5) & (abs(reco_bukmumu.fit_k_eta) < 2.5)
    selections["Bu"]["dR"]      = (delta_r(reco_bukmumu.fit_k_eta, reco_bukmumu.fit_l1_eta, reco_bukmumu.fit_k_phi, reco_bukmumu.fit_l1_phi) > 0.03) \
                                 & (delta_r(reco_bukmumu.fit_k_eta, reco_bukmumu.fit_l2_eta, reco_bukmumu.fit_k_phi, reco_bukmumu.fit_l2_phi) > 0.03) \
                                 & (delta_r(reco_bukmumu.fit_l1_eta, reco_bukmumu.fit_l2_eta, reco_bukmumu.fit_l1_phi, reco_bukmumu.fit_l2_phi) > 0.03)
    selections["Bu"]["jpsi"]    = abs(reco_bukmumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW

    # Final selections["Bu"]
    selections["Bu"]["inclusive"]  = reco_bukmumu.fit_pt.ones_like().astype(bool)
    selections["Bu"]["reco"]       = selections["Bu"]["sv_pt"] \
                                      & selections["Bu"]["l_xy_sig"] \
                                      & selections["Bu"]["sv_prob"] \
                                      & selections["Bu"]["cos2D"] \
                                      & selections["Bu"]["l1"] \
                                      & selections["Bu"]["l2"] \
                                      & selections["Bu"]["k"] \
                                      & selections["Bu"]["dR"] \
                                      & selections["Bu"]["jpsi"]
    for trigger_name, trigger_mask in trigger_masks.items():
      selections["Bu"][f"recotrig_{trigger_name}"]   = selections["Bu"]["reco"] & (trigger_mask * reco_bukmumu_mask_template).astype(bool)
      selections["Bu"][f"tag_{trigger_name}"]   = selections["Bu"][f"recotrig_{trigger_name}"] & (reco_bukmumu.Muon1IsTrig | reco_bukmumu.Muon2IsTrig)
      selections["Bu"][f"tag_{trigger_name}"]   = selections["Bu"][f"tag_{trigger_name}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections["Bu"][f"tag_{trigger_name}"]].min())
      selections["Bu"][f"probe_{trigger_name}"] = selections["Bu"][f"recotrig_{trigger_name}"] & (reco_bukmumu.TagCount >= 1)
      selections["Bu"][f"probe_{trigger_name}"] = selections["Bu"][f"probe_{trigger_name}"] & (reco_bukmumu.chi2 == reco_bukmumu.chi2[selections["Bu"][f"probe_{trigger_name}"]].min())

    # Fill cutflow
    for trigger_strat in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu9_IP5&~HLT_Mu7_IP4"]:
      cutflow_name = f"reco_cutflow_Bu_{trigger_strat}"
      cumulative_selection = reco_bukmumu.pt.ones_like().astype(bool)
      output[cutflow_name][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
      cumulative_selection = cumulative_selection & (trigger_masks[trigger_strat] * reco_bukmumu_mask_template).astype(bool)
      output[cutflow_name][dataset_name]["trigger"] = cumulative_selection.sum().sum()
      for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k", "dR", "jpsi"]:
        cumulative_selection = cumulative_selection & selections["Bu"][cut_name]
        output[cutflow_name][dataset_name][cut_name] += cumulative_selection.sum().sum()
      output[cutflow_name][dataset_name]["tag"] += selections["Bu"][f"tag_{trigger_strat}"].sum().sum()
      output[cutflow_name][dataset_name]["probe"] += selections["Bu"][f"probe_{trigger_strat}"].sum().sum()


    # - Bd to Jpsi K* to mu mu K pi
    reco_bdkpimumu_mask_template = reco_bdkpimumu.pt.ones_like().astype(bool)
    selections["Bd"]["sv_pt"]       = (reco_bdkpimumu.pt > 3.0)
    selections["Bd"]["l_xy_sig"] = (abs(reco_bdkpimumu.l_xy_sig) > 3.5)
    selections["Bd"]["sv_prob"]     = (reco_bdkpimumu.sv_prob > 0.1)
    selections["Bd"]["cos2D"]    = (reco_bdkpimumu.fit_cos2D > 0.999)
    selections["Bd"]["l1"]       =  (reco_bdkpimumu.lep1pt_fullfit > 1.5) & (abs(reco_bdkpimumu.lep1eta_fullfit) < 2.4) & reco_bdkpimumu.l1_softId
    selections["Bd"]["l2"]       =  (reco_bdkpimumu.lep2pt_fullfit > 1.5) & (abs(reco_bdkpimumu.lep2eta_fullfit) < 2.4) & reco_bdkpimumu.l2_softId
    selections["Bd"]["trk1"]     = (reco_bdkpimumu.trk1pt_fullfit > 0.5) & (abs(reco_bdkpimumu.trk1eta_fullfit) < 2.5)
    selections["Bd"]["trk2"]     = (reco_bdkpimumu.trk2pt_fullfit > 0.5) & (abs(reco_bdkpimumu.trk2eta_fullfit) < 2.5)
    selections["Bd"]["dR"]       = (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.trk2phi_fullfit) > 0.03) \
                                    & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
                                    & (delta_r(reco_bdkpimumu.trk1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
                                    & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep1phi_fullfit) > 0.03) \
                                    & (delta_r(reco_bdkpimumu.trk2eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.trk2phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03) \
                                    & (delta_r(reco_bdkpimumu.lep1eta_fullfit, reco_bdkpimumu.lep2eta_fullfit, reco_bdkpimumu.lep1phi_fullfit, reco_bdkpimumu.lep2phi_fullfit) > 0.03)
    selections["Bd"]["jpsi"]           = abs(reco_bdkpimumu.mll_fullfit - JPSI_1S_MASS) < JPSI_WINDOW
    selections["Bd"]["kstar"]          = (abs(reco_bdkpimumu.mkstar_best_fullfit - KSTAR_892_MASS) < KSTAR_WINDOW) & (reco_bdkpimumu.trk1_charge + reco_bdkpimumu.trk2_charge == 0)
    selections["Bd"]["phi_veto"]       = (abs(reco_bdkpimumu.mkstar_best_fullfit - PHI_1020_MASS) > B0_PHI_VETO_WINDOW) \
                                  & (abs(reco_bdkpimumu.barmkstar_best_fullfit - PHI_1020_MASS) > B0_PHI_VETO_WINDOW)

    # Final selections
    selections["Bd"]["inclusive"] = reco_bdkpimumu.fit_pt.ones_like().astype(bool)
    selections["Bd"]["reco"]       = selections["Bd"]["sv_pt"] \
                                    & selections["Bd"]["l_xy_sig"] \
                                    & selections["Bd"]["sv_prob"] \
                                    & selections["Bd"]["cos2D"] \
                                    & selections["Bd"]["l1"] \
                                    & selections["Bd"]["l2"] \
                                    & selections["Bd"]["trk1"] \
                                    & selections["Bd"]["trk2"] \
                                    & selections["Bd"]["dR"] \
                                    & selections["Bd"]["jpsi"] \
                                    & selections["Bd"]["kstar"] \
                                    & selections["Bd"]["phi_veto"]
    for trigger_name, trigger_mask in trigger_masks.items():
      selections["Bd"][f"recotrig_{trigger_name}"]   = selections["Bd"]["reco"] & (trigger_mask * reco_bdkpimumu_mask_template).astype(bool)
      selections["Bd"][f"tag_{trigger_name}"]   = selections["Bd"][f"recotrig_{trigger_name}"] & (reco_bdkpimumu.Muon1IsTrig | reco_bdkpimumu.Muon2IsTrig)
      selections["Bd"][f"tag_{trigger_name}"]   = selections["Bd"][f"tag_{trigger_name}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu.chi2[selections["Bd"][f"tag_{trigger_name}"]].min())
      selections["Bd"][f"probe_{trigger_name}"] = selections["Bd"][f"recotrig_{trigger_name}"] & (reco_bdkpimumu.TagCount >= 1)
      selections["Bd"][f"probe_{trigger_name}"] = selections["Bd"][f"probe_{trigger_name}"] & (reco_bdkpimumu.chi2 == reco_bdkpimumu.chi2[selections["Bd"][f"probe_{trigger_name}"]].min())

    # Fill cutflow
    for trigger_strat in ["HLT_Mu7_IP4", "HLT_Mu9_IP5", "HLT_Mu9_IP6", "HLT_Mu12_IP6", "HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu7_IP4|HLT_Mu9_IP5|HLT_Mu9_IP6", "HLT_Mu9_IP5&~HLT_Mu7_IP4"]:
      cutflow_name = f"reco_cutflow_Bd_{trigger_strat}"
      cumulative_selection = reco_bdkpimumu.pt.ones_like().astype(bool)
      output[cutflow_name][dataset_name]["inclusive"] = cumulative_selection.sum().sum()
      cumulative_selection = cumulative_selection & (trigger_masks[trigger_strat] * reco_bdkpimumu_mask_template).astype(bool)
      output[cutflow_name][dataset_name]["trigger"] = cumulative_selection.sum().sum()
      for cut_name in ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "trk1", "trk2", "dR", "jpsi", "kstar", "phi_veto"]:
        cumulative_selection = cumulative_selection & selections["Bd"][cut_name]
        output[cutflow_name][dataset_name][cut_name] += cumulative_selection.sum().sum()
      output[cutflow_name][dataset_name]["tag"] += selections["Bd"][f"tag_{trigger_strat}"].sum().sum()
      output[cutflow_name][dataset_name]["probe"] += selections["Bd"][f"probe_{trigger_strat}"].sum().sum()


    # Fill reco histograms
    output["nMuon"].fill(dataset=dataset_name, nMuon=df["nMuon"])
    output["nMuon_isTrig"].fill(dataset=dataset_name, nMuon_isTrig=reco_muons.pt[reco_muons.isTriggering==1].count())
    output["Muon_pt"].fill(dataset=dataset_name, Muon_pt=reco_muons.pt.flatten())
    output["Muon_pt_isTrig"].fill(dataset=dataset_name, Muon_pt_isTrig=reco_muons.pt[reco_muons.isTriggering==1].flatten())

    for selection_name in ["inclusive"] + [f"tag_{x}" for x in trigger_masks.keys()] + [f"probe_{x}" for x in trigger_masks.keys()]:
      output["BuToKMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                              fit_pt=reco_bukmumu.fit_pt[selections["Bu"][selection_name]].flatten(),
                                              fit_y=reco_bukmumu.fit_y[selections["Bu"][selection_name]].flatten(),
                                              fit_mass=reco_bukmumu.fit_mass[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bukmumu.fit_pt[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bukmumu.fit_eta[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bukmumu.fit_phi[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bukmumu.fit_mass[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bukmumu.chi2[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bukmumu.fit_cos2D[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bukmumu.fit_cos2D[selections["Bu"][selection_name]]).flatten())
      output["BuToKMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bukmumu.l_xy[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bukmumu.l_xy_sig[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_jpsi_m"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bukmumu.mll_fullfit[selections["Bu"][selection_name]].flatten())
      output["BuToKMuMu_fit_eta_vs_pt_vs_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bukmumu.fit_mass[selections["Bu"][selection_name]].flatten(), fit_pt=reco_bukmumu.fit_pt[selections["Bu"][selection_name]].flatten(), fit_eta=reco_bukmumu.fit_eta[selections["Bu"][selection_name]].flatten())

      output["BsToKKMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                              fit_pt=reco_bskkmumu.fit_pt[selections["Bs"][selection_name]].flatten(),
                                              fit_y=reco_bskkmumu.fit_y[selections["Bs"][selection_name]].flatten(),
                                              fit_mass=reco_bskkmumu.fit_mass[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bskkmumu.fit_pt[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bskkmumu.fit_eta[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bskkmumu.fit_phi[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bskkmumu.fit_mass[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bskkmumu.chi2[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bskkmumu.fit_cos2D[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bskkmumu.fit_cos2D[selections["Bs"][selection_name]]).flatten())
      output["BsToKKMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bskkmumu.l_xy[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bskkmumu.l_xy_sig[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_jpsi_m"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.mll_fullfit[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_phi_m"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bskkmumu.phi_m[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_fit_eta_vs_pt_vs_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bskkmumu.fit_mass[selections["Bs"][selection_name]].flatten(), fit_pt=reco_bskkmumu.fit_pt[selections["Bs"][selection_name]].flatten(), fit_eta=reco_bskkmumu.fit_eta[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_kp_pt"].fill(dataset=dataset_name, selection=selection_name, pt=reco_bskkmumu.kp_pt[selections["Bs"][selection_name]].flatten())
      output["BsToKKMuMu_km_pt"].fill(dataset=dataset_name, selection=selection_name, pt=reco_bskkmumu.km_pt[selections["Bs"][selection_name]].flatten())

      output["BdToKPiMuMu_fit_pt_y_mass"].fill(dataset=dataset_name, selection=selection_name, 
                                              fit_pt=reco_bdkpimumu.fit_pt[selections["Bd"][selection_name]].flatten(),
                                              fit_y=reco_bdkpimumu.fit_y[selections["Bd"][selection_name]].flatten(),
                                              fit_mass=reco_bdkpimumu.fit_mass[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_pt"].fill(dataset=dataset_name, selection=selection_name, fit_pt=reco_bdkpimumu.fit_pt[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_eta"].fill(dataset=dataset_name, selection=selection_name, fit_eta=reco_bdkpimumu.fit_eta[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_phi"].fill(dataset=dataset_name, selection=selection_name, fit_phi=reco_bdkpimumu.fit_phi[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bdkpimumu.fit_mass[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_barmass"].fill(dataset=dataset_name, selection=selection_name, fit_barmass=reco_bdkpimumu.fit_barmass[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bdkpimumu.fit_best_mass[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_best_barmass"].fill(dataset=dataset_name, selection=selection_name, fit_barmass=reco_bdkpimumu.fit_best_barmass[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_chi2"].fill(dataset=dataset_name, selection=selection_name, chi2=reco_bdkpimumu.chi2[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_cos2D"].fill(dataset=dataset_name, selection=selection_name, fit_cos2D=reco_bdkpimumu.fit_cos2D[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_theta2D"].fill(dataset=dataset_name, selection=selection_name, fit_theta2D=np.arccos(reco_bdkpimumu.fit_cos2D[selections["Bd"][selection_name]]).flatten())
      output["BdToKPiMuMu_l_xy"].fill(dataset=dataset_name, selection=selection_name, l_xy=reco_bdkpimumu.l_xy[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_l_xy_sig"].fill(dataset=dataset_name, selection=selection_name, l_xy_sig=reco_bdkpimumu.l_xy_sig[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_jpsi_m"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.mll_fullfit[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_kstar_m"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.mkstar_fullfit[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_kstar_m_best"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.mkstar_best_fullfit[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_kstar_m_bar"].fill(dataset=dataset_name, selection=selection_name, mass=reco_bdkpimumu.barMkstar_fullfit[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_fit_eta_vs_pt_vs_mass"].fill(dataset=dataset_name, selection=selection_name, fit_mass=reco_bdkpimumu.fit_mass[selections["Bd"][selection_name]].flatten(), fit_pt=reco_bdkpimumu.fit_pt[selections["Bd"][selection_name]].flatten(), fit_eta=reco_bdkpimumu.fit_eta[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_k_pt"].fill(dataset=dataset_name, selection=selection_name, pt=reco_bdkpimumu.k_pt[selections["Bd"][selection_name]].flatten())
      output["BdToKPiMuMu_pi_pt"].fill(dataset=dataset_name, selection=selection_name, pt=reco_bdkpimumu.pi_pt[selections["Bd"][selection_name]].flatten())

    # N-1 histograms
    # Bs
    all_cuts = ["sv_pt", "l_xy_sig", "sv_prob", "cos2D", "l1", "l2", "k1", "k2", "dR", "jpsi", "phi", "kstar_veto"]
    for xcut in ["jpsi", "phi", "l_xy_sig", "sv_prob", "cos2D"]:
      nm1_selections["Bs"][f"{xcut}_tag"] = reco_bskkmumu_mask_template
      for cut in all_cuts:
        if not cut == xcut:
          nm1_selections["Bs"][f"{xcut}_tag"] = nm1_selections["Bs"][f"{xcut}_tag"] & selections["Bs"][cut]
      nm1_selections["Bs"][f"{xcut}_probe"] = copy.deepcopy(nm1_selections["Bs"][f"{xcut}_tag"])

      nm1_selections["Bs"][f"{xcut}_tag"] = nm1_selections["Bs"][f"{xcut}_tag"] & (reco_bskkmumu.Muon1IsTrig | reco_bskkmumu.Muon2IsTrig)
      nm1_selections["Bs"][f"{xcut}_tag"] = nm1_selections["Bs"][f"{xcut}_tag"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[nm1_selections["Bs"][f"{xcut}_tag"]].min())

      nm1_selections["Bs"][f"{xcut}_probe"] = nm1_selections["Bs"][f"{xcut}_probe"] & (reco_bskkmumu.TagCount >= 1)
      nm1_selections["Bs"][f"{xcut}_probe"] = nm1_selections["Bs"][f"{xcut}_probe"] & (reco_bskkmumu.chi2 == reco_bskkmumu.chi2[nm1_selections["Bs"][f"{xcut}_tag"]].min())
    for selection in ["tag", "probe"]:
      output["NM1_BsToKKMuMu_jpsi_m"].fill(dataset=dataset_name, selection="f{selection}", mass=reco_bskkmumu.mll_fullfit[nm1_selections["Bs"][f"jpsi_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_phi_m"].fill(dataset = dataset_name, selection="f{selection}", mass=reco_bskkmumu.phi_m[nm1_selections["Bs"][f"phi_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_l_xy_sig"].fill(dataset=dataset_name, selection="f{selection}", l_xy_sig=reco_bskkmumu.l_xy_sig[nm1_selections["Bs"][f"l_xy_sig_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_sv_prob"].fill(dataset=dataset_name, selection="f{selection}", sv_prob=reco_bskkmumu.sv_prob[nm1_selections["Bs"][f"sv_prob_{selection}"]].flatten())
      output["NM1_BsToKKMuMu_cos2D"].fill(dataset=dataset_name, selection="f{selection}", cos2D=reco_bskkmumu.fit_cos2D[nm1_selections["Bs"][f"cos2D_{selection}"]].flatten())

    # Tree outputs
    for selection_name in [f"tag_{x}" for x in trigger_masks.keys()] + [f"probe_{x}" for x in trigger_masks.keys()]:
      for key in output.keys():
        if "Bcands_Bu" in key:
          print(key)
      output[f"Bcands_Bu_{selection_name}"][dataset_name].extend({
        "pt": reco_bukmumu.fit_pt[selections["Bu"][selection_name]].flatten(),
        "eta": reco_bukmumu.fit_eta[selections["Bu"][selection_name]].flatten(),
        "y": reco_bukmumu.fit_y[selections["Bu"][selection_name]].flatten(),
        "phi": reco_bukmumu.fit_phi[selections["Bu"][selection_name]].flatten(),
        "mass": reco_bukmumu.fit_mass[selections["Bu"][selection_name]].flatten(),
        "l_xy": reco_bukmumu.l_xy[selections["Bu"][selection_name]].flatten(),
        "l_xy_unc": reco_bukmumu.l_xy_unc[selections["Bu"][selection_name]].flatten(),
        "sv_prob": reco_bukmumu.sv_prob[selections["Bu"][selection_name]].flatten(),
        "cos2D": reco_bukmumu.fit_cos2D[selections["Bu"][selection_name]].flatten(),
      })
      output[f"Bcands_Bd_{selection_name}"][dataset_name].extend({
        "pt": reco_bdkpimumu.fit_pt[selections["Bd"][selection_name]].flatten(),
        "eta": reco_bdkpimumu.fit_eta[selections["Bd"][selection_name]].flatten(),
        "y": reco_bdkpimumu.fit_y[selections["Bd"][selection_name]].flatten(),
        "phi": reco_bdkpimumu.fit_phi[selections["Bd"][selection_name]].flatten(),
        "mass": reco_bdkpimumu.fit_best_mass[selections["Bd"][selection_name]].flatten(),
        "l_xy": reco_bdkpimumu.l_xy[selections["Bd"][selection_name]].flatten(),
        "l_xy_unc": reco_bdkpimumu.l_xy_unc[selections["Bd"][selection_name]].flatten(),
        "sv_prob": reco_bdkpimumu.sv_prob[selections["Bd"][selection_name]].flatten(),
        "cos2D": reco_bdkpimumu.fit_cos2D[selections["Bd"][selection_name]].flatten(),
      })
      output[f"Bcands_Bs_{selection_name}"][dataset_name].extend({
        "pt": reco_bskkmumu.fit_pt[selections["Bs"][selection_name]].flatten(),
        "eta": reco_bskkmumu.fit_eta[selections["Bs"][selection_name]].flatten(),
        "y": reco_bskkmumu.fit_y[selections["Bs"][selection_name]].flatten(),
        "phi": reco_bskkmumu.fit_phi[selections["Bs"][selection_name]].flatten(),
        "mass": reco_bskkmumu.fit_mass[selections["Bs"][selection_name]].flatten(),
        "l_xy": reco_bskkmumu.l_xy[selections["Bs"][selection_name]].flatten(),
        "l_xy_unc": reco_bskkmumu.l_xy_unc[selections["Bs"][selection_name]].flatten(),
        "sv_prob": reco_bskkmumu.sv_prob[selections["Bs"][selection_name]].flatten(),
        "cos2D": reco_bskkmumu.fit_cos2D[selections["Bs"][selection_name]].flatten(),
      })

    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser(description="Make histograms for B FFR data")
  parser.add_argument("--datasets", "-d", type=str, help="List of datasets to run (comma-separated")
  parser.add_argument("--workers", "-w", type=int, default=16, help="Number of workers")
  parser.add_argument("--quicktest", "-q", action="store_true", help="Run a small test job")
  parser.add_argument("--save_tag", "-s", type=str, help="Save tag for output file")
  parser.add_argument("--nopbar", action="store_true", help="Disable progress bar (do this on condor)")
  args = parser.parse_args()

  datasets = args.datasets.split(",")

  # Inputs
  #in_txt = {
  #  "Run2018A_part1": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part1.txt",
  #  "Run2018A_part2": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part2.txt",
  #  "Run2018A_part3": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part3.txt",
  #  "Run2018A_part4": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part4.txt",
  #  "Run2018A_part5": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part5.txt",
  #  "Run2018A_part6": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018A_part6.txt",
  #  "Run2018B_part1": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part1.txt",
  #  "Run2018B_part2": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part2.txt",
  #  "Run2018B_part3": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part3.txt",
  #  "Run2018B_part4": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part4.txt",
  #  "Run2018B_part5": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part5.txt",
  #  "Run2018B_part6": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018B_part6.txt",
  #  "Run2018C_part1": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018C_part1.txt",
  #  "Run2018C_part2": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018C_part2.txt",
  #  "Run2018C_part3": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018C_part3.txt",
  #  "Run2018C_part4": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018C_part4.txt",
  #  "Run2018C_part5": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018C_part5.txt",
  #  "Run2018D_part1": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018D_part1.txt",
  #  "Run2018D_part2": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018D_part2.txt",
  #  "Run2018D_part3": "/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_3/files_Run2018D_part3.txt/home/dryu/BFrag/CMSSW_10_2_18/src/BParkingNANOAnalysis/BParkingNANOAnalyzer/data/skim_directory/v1_7/partialdatasplit/",
  #}
  #in_txt = {}
  from data_index import in_txt

  dataset_files = {}
  for dataset_name in datasets:
    if not dataset_name in in_txt:
      raise ValueError(f"Dataset {dataset_name} not in dictionary.")
    with open(in_txt[dataset_name], 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

    if args.quicktest:
      dataset_files[dataset_name] = dataset_files[dataset_name][:20]

  ts_start = time.time()
  print(dataset_files)
  output = processor.run_uproot_job(dataset_files,
                                treename='Events',
                                processor_instance=DataProcessor(),
                                executor=processor.futures_executor,
                                executor_args={'workers': 16, 'flatten': False, 'status':not args.nopbar},
                                chunksize=50000,
                                # maxchunks=1,
                            )
  util.save(output, f"DataHistograms_{args.save_tag}.coffea")

  # Performance benchmarking and cutflows
  ts_end = time.time()
  total_events = 0
  dataset_nevents = {}
  for k, v in output['nevents'].items():
    if k in dataset_nevents:
      dataset_nevents[k] += v
    else:
      dataset_nevents[k] = v
    total_events += v


  print("Reco cutflows:")
  for dataset, d1 in output["reco_cutflow_Bs_inclusive"].items():
    print(f"\tDataset={dataset}")
    print(f"\t\tnevents => {dataset_nevents[dataset]}")

    print("\n\t\tBs:\n")
    for cut_name, cut_npass in output["reco__cutflow_Bs_inclusive"][dataset].items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / dataset_nevents[dataset]}")
    print("\n\t\tBu:\n")
    for cut_name, cut_npass in output["reco__cutflow_Bu_inclusive"][dataset].items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / dataset_nevents[dataset]}")
    print("\n\t\tBd:\n")
    for cut_name, cut_npass in output["reco__cutflow_Bd_inclusive"][dataset].items():
      print(f"\t\t{cut_name} => {cut_npass} = {cut_npass / dataset_nevents[dataset]}")


  print("Total time: {} seconds".format(ts_end - ts_start))
  print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))

