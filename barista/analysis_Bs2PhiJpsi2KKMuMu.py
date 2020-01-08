#! /usr/bin/env python

#import ROOT
from math import ceil
import awkward
import uproot
import uproot_methods
import pandas as pd
import numpy as np
from analysis_base import *


class Bs2PhiJpsi2KKMuMuAnalyzer(AnalysisBase):
  def __init__(self, inputfiles, outputfile, hist=False, data_source=DataSource.kDATA):
    self._data_source = data_source
    inputbranches_BsToKKMuMu = [
      "nBToPhiMuMu",
      "BToPhiMuMu_chi2",
      "BToPhiMuMu_cos2D",
      "BToPhiMuMu_eta",
      "BToPhiMuMu_etaphi_fullfit",
      "BToPhiMuMu_fit_cos2D",
      "BToPhiMuMu_fit_eta",
      "BToPhiMuMu_fit_mass",
      "BToPhiMuMu_fit_massErr",
      "BToPhiMuMu_fit_phi",
      "BToPhiMuMu_fit_pt",
      "BToPhiMuMu_l_xy",
      "BToPhiMuMu_l_xy_unc",
      "BToPhiMuMu_lep1eta_fullfit",
      "BToPhiMuMu_lep1phi_fullfit",
      "BToPhiMuMu_lep1pt_fullfit",
      "BToPhiMuMu_lep2eta_fullfit",
      "BToPhiMuMu_lep2phi_fullfit",
      "BToPhiMuMu_lep2pt_fullfit",
      "BToPhiMuMu_mass",
      "BToPhiMuMu_max_dr",
      "BToPhiMuMu_min_dr",
      "BToPhiMuMu_mll_fullfit",
      "BToPhiMuMu_mll_llfit",
      "BToPhiMuMu_mll_raw",
      "BToPhiMuMu_mphi_fullfit",
      "BToPhiMuMu_phi",
      "BToPhiMuMu_phiphi_fullfit",
      "BToPhiMuMu_pt",
      "BToPhiMuMu_ptphi_fullfit",
      "BToPhiMuMu_svprob",
      "BToPhiMuMu_trk1eta_fullfit",
      "BToPhiMuMu_trk1phi_fullfit",
      "BToPhiMuMu_trk1pt_fullfit",
      "BToPhiMuMu_trk2eta_fullfit",
      "BToPhiMuMu_trk2phi_fullfit",
      "BToPhiMuMu_trk2pt_fullfit",
      "BToPhiMuMu_charge",
      "BToPhiMuMu_l1_idx",
      "BToPhiMuMu_l2_idx",
      "BToPhiMuMu_pdgId",
      "BToPhiMuMu_phi_idx",
      "BToPhiMuMu_trk1_idx",
      "BToPhiMuMu_trk2_idx",
      "nPhi",
      "Phi_eta",
      "Phi_fitted_eta",
      "Phi_fitted_mass",
      "Phi_fitted_phi",
      "Phi_fitted_pt",
      "Phi_mass",
      "Phi_phi",
      "Phi_pt",
      "Phi_svprob",
      "Phi_trk_deltaR",
      "Phi_charge",
      "Phi_pdgId",
      "Phi_trk1_idx",
      "Phi_trk2_idx",
      "HLT_Mu7_IP4",
      "HLT_Mu8_IP6",
      "HLT_Mu8_IP5",
      "HLT_Mu8_IP3",
      "HLT_Mu8p5_IP3p5",
      "HLT_Mu9_IP6",
      "HLT_Mu9_IP5",
      "HLT_Mu9_IP4",
      "HLT_Mu10p5_IP3p5",
      "HLT_Mu12_IP6",
      "L1_SingleMu7er1p5",
      "L1_SingleMu8er1p5",
      "L1_SingleMu9er1p5",
      "L1_SingleMu10er1p5",
      "L1_SingleMu12er1p5",
      "L1_SingleMu22",
      "nTrigObj",
      "TrigObj_pt",
      "TrigObj_eta",
      "TrigObj_phi",
      "TrigObj_l1pt",
      "TrigObj_l1pt_2",
      "TrigObj_l2pt",
      "TrigObj_id",
      "TrigObj_l1iso",
      "TrigObj_l1charge",
      "TrigObj_filterBits",
      "nTriggerMuon",
      "TriggerMuon_eta",
      "TriggerMuon_mass",
      "TriggerMuon_phi",
      "TriggerMuon_pt",
      "TriggerMuon_vx",
      "TriggerMuon_vy",
      "TriggerMuon_vz",
      "TriggerMuon_charge",
      "TriggerMuon_pdgId",
      "TriggerMuon_trgMuonIndex",
      "event",
    ]

    inputbranches_BsToKKMuMu_mc = ['GenPart_pdgId',
                                   'GenPart_genPartIdxMother'
                                  ]

    if (self._data_source == DataSource.kMC):
      inputbranches_BsToKKMuMu += inputbranches_BsToKKMuMu_mc

    # Histograms
    dataset = hist.Cat("dataset", "Primary dataset")


    self._histograms = {}
    self._histograms["BsToKKMuMu_mraw"] = ROOT.TH1D("BsToKKMuMu_mraw", "BsToKKMuMu_mraw", 50, 0.0, 5.0)

    self._histograms["BsToKKMuMu_mfit"] = ROOT.TH1D("BsToKKMuMu_mfit", "BsToKKMuMu_mfit", 50, 0.0, 5.0)

    self._histograms["BToPhiMuMu_chi2"]            = ROOT.TH1D("BToPhiMuMu_chi2", "BToPhiMuMu_chi2", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_eta"]             = ROOT.TH1D("BToPhiMuMu_eta", "BToPhiMuMu_eta", 50, -5.0, 5.0)
    self._histograms["BToPhiMuMu_fit_cos2D"]       = ROOT.TH1D("BToPhiMuMu_fit_cos2D", "BToPhiMuMu_fit_cos2D", 100, -1., 1.)
    self._histograms["BToPhiMuMu_fit_eta"]         = ROOT.TH1D("BToPhiMuMu_fit_eta", "BToPhiMuMu_fit_eta", 50, -5.0, 5.0)
    self._histograms["BToPhiMuMu_fit_mass"]        = ROOT.TH1D("BToPhiMuMu_fit_mass", "BToPhiMuMu_fit_mass", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_fit_phi"]         = ROOT.TH1D("BToPhiMuMu_fit_phi", "BToPhiMuMu_fit_phi", 50, -2.0*math.pi, 2.0*math.pi)
    self._histograms["BToPhiMuMu_fit_pt"]          = ROOT.TH1D("BToPhiMuMu_fit_pt", "BToPhiMuMu_fit_pt", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_l_xy"]            = ROOT.TH1D("BToPhiMuMu_l_xy", "BToPhiMuMu_l_xy", 100, -10.0, 10.0)
    self._histograms["BToPhiMuMu_lep1eta_fullfit"] = ROOT.TH1D("BToPhiMuMu_lep1eta_fullfit", "BToPhiMuMu_lep1eta_fullfit", 50, -5.0, 5.0)
    self._histograms["BToPhiMuMu_lep1phi_fullfit"] = ROOT.TH1D("BToPhiMuMu_lep1phi_fullfit", "BToPhiMuMu_lep1phi_fullfit", 50, -2.0*math.pi, 2.0*math.pi)
    self._histograms["BToPhiMuMu_lep1pt_fullfit"]  = ROOT.TH1D("BToPhiMuMu_lep1pt_fullfit", "BToPhiMuMu_lep1pt_fullfit", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_lep2eta_fullfit"] = ROOT.TH1D("BToPhiMuMu_lep2eta_fullfit", "BToPhiMuMu_lep2eta_fullfit", 50, -5.0, 5.0)
    self._histograms["BToPhiMuMu_lep2phi_fullfit"] = ROOT.TH1D("BToPhiMuMu_lep2phi_fullfit", "BToPhiMuMu_lep2phi_fullfit", 50, -2.0*math.pi, 2.0*math.pi)
    self._histograms["BToPhiMuMu_lep2pt_fullfit"]  = ROOT.TH1D("BToPhiMuMu_lep2pt_fullfit", "BToPhiMuMu_lep2pt_fullfit", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_mass"]            = ROOT.TH1D("BToPhiMuMu_mass", "BToPhiMuMu_mass", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_mll_fullfit"]     = ROOT.TH1D("BToPhiMuMu_mll_fullfit", "BToPhiMuMu_mll_fullfit", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_mll_llfit"]       = ROOT.TH1D("BToPhiMuMu_mll_llfit", "BToPhiMuMu_mll_llfit", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_mll_raw"]         = ROOT.TH1D("BToPhiMuMu_mll_raw", "BToPhiMuMu_mll_raw", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_mphi_fullfit"]    = ROOT.TH1D("BToPhiMuMu_mphi_fullfit", "BToPhiMuMu_mphi_fullfit", 100, 0.0, 5.0)
    self._histograms["BToPhiMuMu_phi"]             = ROOT.TH1D("BToPhiMuMu_phi", "BToPhiMuMu_phi", 50, -2.0*math.pi, 2.0*math.pi)
    self._histograms["BToPhiMuMu_etaphi_fullfit"]  = ROOT.TH1D("BToPhiMuMu_etaphi_fullfit", "BToPhiMuMu_etaphi_fullfit", 50, -5.0, 5.0)
    self._histograms["BToPhiMuMu_phiphi_fullfit"]  = ROOT.TH1D("BToPhiMuMu_phiphi_fullfit", "BToPhiMuMu_phiphi_fullfit", 50, -2.0*math.pi, 2.0*math.pi)
    self._histograms["BToPhiMuMu_pt"]              = ROOT.TH1D("BToPhiMuMu_pt", "BToPhiMuMu_pt", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_ptphi_fullfit"]   = ROOT.TH1D("BToPhiMuMu_ptphi_fullfit", "BToPhiMuMu_ptphi_fullfit", 100, 0.0, 100.0)
    self._histograms["BToPhiMuMu_svprob"]          = ROOT.TH1D("BToPhiMuMu_svprob", "BToPhiMuMu_svprob", 100, -1.0, 1.0)
    self._histograms["BToPhiMuMu_charge"]          = ROOT.TH1D("BToPhiMuMu_charge", "BToPhiMuMu_charge", 3, -1.5, 1.5)

    super(BsToKKMuMuAnalyzer, self).__init__(inputfiles, outputfile, inputbranches_BsToKKMuMu, outputbranches_BsToKKMuMu, hist)

  def run(self):

    print('[BsToKKMuMuAnalyzer::run] INFO: Running the analyzer...')
    self.print_timestamp()
    self.init_output()
    for (self._ifile, filename) in enumerate(self._file_in_name):
      print('[BsToKKMuMuAnalyzer::run] INFO: FILE: {}/{}. Loading file...'.format(self._ifile+1, self._num_files))
      tree = uproot.open(filename)['Events']
      self._branches = tree.arrays(self._inputbranches)
      self._branches = {key: awkward.fromiter(branch) for key, branch in self._branches.items()}

      print('[BsToKKMuMuAnalyzer::run] INFO: FILE: {}/{}. Analyzing...'.format(self._ifile+1, self._num_files))

      if (self._data_source = DataSource.kMC):
        # reconstruct full decay chain
        self._branches['BsToKKMuMu_l1_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BsToKKMuMu_l1Idx']]]
        self._branches['BsToKKMuMu_l2_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['Electron_genPartIdx'][self._branches['BsToKKMuMu_l2Idx']]]
        self._branches['BsToKKMuMu_k_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['ProbeTracks_genPartIdx'][self._branches['BsToKKMuMu_kIdx']]]

        self._branches['BsToKKMuMu_l1_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l1_genMotherIdx']]
        self._branches['BsToKKMuMu_l2_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l2_genMotherIdx']]
        self._branches['BsToKKMuMu_k_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_k_genMotherIdx']]

        self._branches['BsToKKMuMu_l1Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BsToKKMuMu_l1_genMotherIdx']]
        self._branches['BsToKKMuMu_l2Mother_genMotherIdx'] = self._branches['GenPart_genPartIdxMother'][self._branches['BsToKKMuMu_l2_genMotherIdx']]

        self._branches['BsToKKMuMu_l1Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l1Mother_genMotherIdx']]
        self._branches['BsToKKMuMu_l2Mother_genMotherPdgId'] = self._branches['GenPart_pdgId'][self._branches['BsToKKMuMu_l2Mother_genMotherIdx']]


      # remove cross referencing
      for branch in self._branches.keys():
        if 'Electron_' in branch:
          self._branches['BsToKKMuMu_l1_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BsToKKMuMu_l1Idx']] 
          self._branches['BsToKKMuMu_l2_'+branch.replace('Electron_','')] = self._branches[branch][self._branches['BsToKKMuMu_l2Idx']] 
          del self._branches[branch]

        if 'ProbeTracks_' in branch:
          self._branches['BsToKKMuMu_k_'+branch.replace('ProbeTracks_','')] = self._branches[branch][self._branches['BsToKKMuMu_kIdx']] 
          del self._branches[branch]
        
        if 'GenPart_' in branch:
          del self._branches[branch]

        if 'HLT_Mu9_IP6_' in branch:
          self._branches['BsToKKMuMu_'+branch] = np.repeat(self._branches[branch], self._branches['nBsToKKMuMu'])
          del self._branches[branch]

        if branch == 'event':
          self._branches['BsToKKMuMu_'+branch] = np.repeat(self._branches[branch], self._branches['nBsToKKMuMu'])
          del self._branches[branch]
        

      del self._branches['nBsToKKMuMu']



      #l1_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BsToKKMuMu_l1_pt'], self._branches['BsToKKMuMu_l1_eta'], self._branches['BsToKKMuMu_l1_phi'], ELECTRON_MASS)
      #l2_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BsToKKMuMu_l2_pt'], self._branches['BsToKKMuMu_l2_eta'], self._branches['BsToKKMuMu_l2_phi'], ELECTRON_MASS)
      #k_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(self._branches['BsToKKMuMu_k_pt'], self._branches['BsToKKMuMu_k_eta'], self._branches['BsToKKMuMu_k_phi'], K_MASS)
      #self._branches['BsToKKMuMu_mll_raw'] = (l1_p4 + l2_p4).mass
      #self._branches['BsToKKMuMu_mass'] = (l1_p4 + l2_p4 + k_p4).mass

      # flatten the jagged arrays to a normal numpy array, turn the whole dictionary to pandas dataframe
      self._branches = pd.DataFrame.from_dict({branch: array.flatten() for branch, array in self._branches.items()})
      #self._branches = awkward.topandas(self._branches, flatten=True)


      # add additional branches
      #self._branches['BsToKKMuMu_l_xy_sig'] = self._branches['BsToKKMuMu_l_xy'] / np.sqrt(self._branches['BsToKKMuMu_l_xy_unc'])
      self._branches['BsToKKMuMu_l_xy_sig'] = self._branches['BsToKKMuMu_l_xy'] / self._branches['BsToKKMuMu_l_xy_unc']
      self._branches['BsToKKMuMu_l1_dxy_sig'] = self._branches['BsToKKMuMu_l1_dxy'] / self._branches['BsToKKMuMu_l1_dxyErr']
      self._branches['BsToKKMuMu_l2_dxy_sig'] = self._branches['BsToKKMuMu_l2_dxy'] / self._branches['BsToKKMuMu_l2_dxyErr']
      self._branches['BsToKKMuMu_fit_l1_normpt'] = self._branches['BsToKKMuMu_fit_l1_pt'] / self._branches['BsToKKMuMu_fit_mass']
      self._branches['BsToKKMuMu_fit_l2_normpt'] = self._branches['BsToKKMuMu_fit_l2_pt'] / self._branches['BsToKKMuMu_fit_mass']
      self._branches['BsToKKMuMu_fit_k_normpt'] = self._branches['BsToKKMuMu_fit_k_pt'] / self._branches['BsToKKMuMu_fit_mass']
      self._branches['BsToKKMuMu_fit_normpt'] = self._branches['BsToKKMuMu_fit_pt'] / self._branches['BsToKKMuMu_fit_mass']

      # general selection
      
      sv_selection = (self._branches['BsToKKMuMu_fit_pt'] > 3.0) & (self._branches['BsToKKMuMu_l_xy_sig'] > 6.0 ) & (self._branches['BsToKKMuMu_svprob'] > 0.01) & (self._branches['BsToKKMuMu_fit_cos2D'] > 0.9)
      l1_selection = (self._branches['BsToKKMuMu_l1_convVeto']) & (self._branches['BsToKKMuMu_fit_l1_pt'] > 1.5) #& (self._branches['BsToKKMuMu_l1_mvaId'] > 3.94) #& (np.logical_not(self._branches['BsToKKMuMu_l1_isPFoverlap']))
      l2_selection = (self._branches['BsToKKMuMu_l2_convVeto']) & (self._branches['BsToKKMuMu_fit_l2_pt'] > 0.5) #& (self._branches['BsToKKMuMu_l2_mvaId'] > 3.94) #& (np.logical_not(self._branches['BsToKKMuMu_l2_isPFoverlap']))
      k_selection = (self._branches['BsToKKMuMu_fit_k_pt'] > 0.5) #& (self._branches['BsToKKMuMu_k_DCASig'] > 2.0)
      #additional_selection = (self._branches['BsToKKMuMu_fit_mass'] > B_LOW) & (self._branches['BsToKKMuMu_fit_mass'] < B_UP)

      b_lowsb_selection = (self._branches['BsToKKMuMu_fit_mass'] > B_LOWSB_LOW) & (self._branches['BsToKKMuMu_fit_mass'] < B_LOWSB_UP)
      b_upsb_selection = (self._branches['BsToKKMuMu_fit_mass'] > B_UPSB_LOW) & (self._branches['BsToKKMuMu_fit_mass'] < B_UPSB_UP)
      b_sb_selection = b_lowsb_selection | b_upsb_selection
      if (self._data_source == DataSource.kMC):
        mc_matched_selection = (self._branches['BsToKKMuMu_l1_genPartIdx'] > -0.5) & (self._branches['BsToKKMuMu_l2_genPartIdx'] > -0.5) & (self._branches['BsToKKMuMu_k_genPartIdx'] > -0.5)
        # B->K J/psi(ee)
        #mc_parent_selection = (abs(self._branches['BsToKKMuMu_l1_genMotherPdgId']) == 443) & (abs(self._branches['BsToKKMuMu_k_genMotherPdgId']) == 521)
        #mc_chain_selection = (self._branches['BsToKKMuMu_l1_genMotherPdgId'] == self._branches['BsToKKMuMu_l2_genMotherPdgId']) & (self._branches['BsToKKMuMu_k_genMotherPdgId'] == self._branches['BsToKKMuMu_l1Mother_genMotherPdgId']) & (self._branches['BsToKKMuMu_k_genMotherPdgId'] == self._branches['BsToKKMuMu_l2Mother_genMotherPdgId'])

        # B->K*(K pi) J/psi(ee)
        mc_parent_selection = (abs(self._branches['BsToKKMuMu_l1_genMotherPdgId']) == 443) & (abs(self._branches['BsToKKMuMu_k_genMotherPdgId']) == 313)
        mc_chain_selection = (self._branches['BsToKKMuMu_l1_genMotherPdgId'] == self._branches['BsToKKMuMu_l2_genMotherPdgId'])
        mc_selection = mc_matched_selection & mc_parent_selection & mc_chain_selection

      #additional_selection = b_sb_selection
      if (self._data_source == DataSource.kMC):
        selection = l1_selection & l2_selection & k_selection & mc_selection

      else:
        selection = l1_selection & l2_selection & k_selection

      self._branches = self._branches[selection]

      # fill output
      self.fill_output()

    self.finish()
    print('[BsToKKMuMuAnalyzer::run] INFO: Finished')
    self.print_timestamp()


