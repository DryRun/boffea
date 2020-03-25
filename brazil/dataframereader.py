from coffea.analysis_objects import JaggedCandidateArray
import awkward
import uproot_methods
from brazil.aguapreta import *

def genparts(df, is_mc=True):
    genparts_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nGenPart"].flatten(),
        eta              = df["GenPart_eta"].flatten(),
        mass             = df["GenPart_mass"].flatten(),
        phi              = df["GenPart_phi"].flatten(),
        pt               = df["GenPart_pt"].flatten(),
        vx               = df["GenPart_vx"].flatten(),
        vy               = df["GenPart_vy"].flatten(),
        vz               = df["GenPart_vz"].flatten(),
        status           = df["GenPart_status"].flatten(),
        genPartIdxMother = df["GenPart_genPartIdxMother"].flatten(),
        pdgId            = df["GenPart_pdgId"].flatten(),
    )
    return genparts_collection

def probe_tracks(df, is_mc=False):
    probe_tracks_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nProbeTracks"].flatten(),
        DCASig                = df["ProbeTracks_DCASig"].flatten(),
        dxy                   = df["ProbeTracks_dxy"].flatten(),
        dxyS                  = df["ProbeTracks_dxyS"].flatten(),
        dz                    = df["ProbeTracks_dz"].flatten(),
        dzS                   = df["ProbeTracks_dzS"].flatten(),
        eta                   = df["ProbeTracks_eta"].flatten(),
        mass                  = df["ProbeTracks_mass"].flatten(),
        phi                   = df["ProbeTracks_phi"].flatten(),
        pt                    = df["ProbeTracks_pt"].flatten(),
        vx                    = df["ProbeTracks_vx"].flatten(),
        vy                    = df["ProbeTracks_vy"].flatten(),
        vz                    = df["ProbeTracks_vz"].flatten(),
        charge                = df["ProbeTracks_charge"].flatten(),
        isLostTrk             = df["ProbeTracks_isLostTrk"].flatten(),
        isPacked              = df["ProbeTracks_isPacked"].flatten(),
        isMatchedToEle        = df["ProbeTracks_isMatchedToEle"].flatten(),
        isMatchedToLooseMuon  = df["ProbeTracks_isMatchedToLooseMuon"].flatten(),
        isMatchedToMediumMuon = df["ProbeTracks_isMatchedToMediumMuon"].flatten(),
        isMatchedToMuon       = df["ProbeTracks_isMatchedToMuon"].flatten(),
        isMatchedToSoftMuon   = df["ProbeTracks_isMatchedToSoftMuon"].flatten(),
    )
    if is_mc:
        probe_tracks_collection.add_attributes(
            pdgId                 = df["ProbeTracks_pdgId"].flatten(),
            genPartIdx            = df["ProbeTracks_genPartIdx"].flatten(),
            genPartFlav           = df["ProbeTracks_genPartFlav"].flatten(),
        )
    return probe_tracks_collection

# Unpack barmass into unique candidates
def reco_bdkpimumu(df, is_mc=False):
    '''
    for key in ["BToKsMuMu_barMass", "BToKsMuMu_barMkstar_fullfit", "BToKsMuMu_chi2", "BToKsMuMu_cos2D", "BToKsMuMu_eta", "BToKsMuMu_etakstar_fullfit", "BToKsMuMu_fit_cos2D", "BToKsMuMu_fit_eta", "BToKsMuMu_fit_mass", "BToKsMuMu_fit_massErr", "BToKsMuMu_fit_phi", "BToKsMuMu_fit_pt", "BToKsMuMu_fitted_barMass", "BToKsMuMu_l_xy", "BToKsMuMu_l_xy_unc", "BToKsMuMu_lep1eta_fullfit", "BToKsMuMu_lep1phi_fullfit", "BToKsMuMu_lep1pt_fullfit", "BToKsMuMu_lep2eta_fullfit", "BToKsMuMu_lep2phi_fullfit", "BToKsMuMu_lep2pt_fullfit", "BToKsMuMu_mass", "BToKsMuMu_max_dr", "BToKsMuMu_min_dr", "BToKsMuMu_mkstar_fullfit", "BToKsMuMu_mll_fullfit", "BToKsMuMu_mll_llfit", "BToKsMuMu_mll_raw", "BToKsMuMu_phi", "BToKsMuMu_phikstar_fullfit", "BToKsMuMu_pt", "BToKsMuMu_ptkstar_fullfit", "BToKsMuMu_svprob", "BToKsMuMu_trk1eta_fullfit_new", "BToKsMuMu_trk1phi_fullfit_new", "BToKsMuMu_trk1pt_fullfit_new", "BToKsMuMu_trk2eta_fullfit_new", "BToKsMuMu_trk2phi_fullfit_new", "BToKsMuMu_trk2pt_fullfit_new", "BToKsMuMu_charge", "BToKsMuMu_kstar_idx", "BToKsMuMu_l1_idx", "BToKsMuMu_l2_idx", "BToKsMuMu_pdgId", "BToKsMuMu_trk1_idx_new", "BToKsMuMu_trk2_idx_new"]:
        df.preload(key)
    #df.preload(df.columns())
    # Branches shared between nominal and alternative
    for key in ["BToKsMuMu_chi2", "BToKsMuMu_cos2D", "BToKsMuMu_eta", "BToKsMuMu_etakstar_fullfit", "BToKsMuMu_fit_cos2D", "BToKsMuMu_fit_eta", "BToKsMuMu_fit_massErr", "BToKsMuMu_fit_phi", "BToKsMuMu_fit_pt", "BToKsMuMu_l_xy", "BToKsMuMu_l_xy_unc", "BToKsMuMu_lep1eta_fullfit", "BToKsMuMu_lep1phi_fullfit", "BToKsMuMu_lep1pt_fullfit", "BToKsMuMu_lep2eta_fullfit", "BToKsMuMu_lep2phi_fullfit", "BToKsMuMu_lep2pt_fullfit", "BToKsMuMu_max_dr", "BToKsMuMu_min_dr", "BToKsMuMu_mll_fullfit", "BToKsMuMu_mll_llfit", "BToKsMuMu_mll_raw", "BToKsMuMu_phi", "BToKsMuMu_phikstar_fullfit", "BToKsMuMu_pt", "BToKsMuMu_ptkstar_fullfit", "BToKsMuMu_svprob", "BToKsMuMu_charge", "BToKsMuMu_kstar_idx", "BToKsMuMu_l1_idx", "BToKsMuMu_l2_idx", "BToKsMuMu_pdgId"]:
        df.preload(key)
        df[key] = awkward.concatenate((df[key], df[key]), axis=1)

    # Branches to swap alternate and nominal
    df["BToKsMuMu_fit_mass"]        = awkward.concatenate((df["BToKsMuMu_fit_mass"], df["BToKsMuMu_fitted_barMass"]), axis=1)
    df["BToKsMuMu_mass"]            = awkward.concatenate((df["BToKsMuMu_mass"], df["BToKsMuMu_barMass"]), axis=1)
    df["BToKsMuMu_mkstar_fullfit"]  = awkward.concatenate((df["BToKsMuMu_mkstar_fullfit"], df["BToKsMuMu_barMkstar_fullfit"]), axis=1)
    df["BToKsMuMu_trk1_idx_new"]        = awkward.concatenate((df["BToKsMuMu_trk1_idx"], df["BToKsMuMu_trk2_idx"]), axis=1)
    df["BToKsMuMu_trk2_idx_new"]        = awkward.concatenate((df["BToKsMuMu_trk2_idx"], df["BToKsMuMu_trk1_idx"]), axis=1)
    df["BToKsMuMu_trk1eta_fullfit_new"] = awkward.concatenate((df["BToKsMuMu_trk1eta_fullfit"], df["BToKsMuMu_trk2eta_fullfit"]), axis=1)
    df["BToKsMuMu_trk2eta_fullfit_new"] = awkward.concatenate((df["BToKsMuMu_trk2eta_fullfit"], df["BToKsMuMu_trk1eta_fullfit"]), axis=1)
    
    df["BToKsMuMu_trk1phi_fullfit_new"] = awkward.concatenate((df["BToKsMuMu_trk1phi_fullfit"], df["BToKsMuMu_trk2phi_fullfit"]), axis=1)
    df["BToKsMuMu_trk2phi_fullfit_new"] = awkward.concatenate((df["BToKsMuMu_trk2phi_fullfit"], df["BToKsMuMu_trk1phi_fullfit"]), axis=1)
    
    df["BToKsMuMu_trk1pt_fullfit_new"]  = awkward.concatenate((df["BToKsMuMu_trk1pt_fullfit"], df["BToKsMuMu_trk2pt_fullfit"]), axis=1)
    df["BToKsMuMu_trk2pt_fullfit_new"]  = awkward.concatenate((df["BToKsMuMu_trk2pt_fullfit"], df["BToKsMuMu_trk1pt_fullfit"]), axis=1)

    df["nBToKsMuMu"] = df["nBToKsMuMu"] * 2
    '''
    reco_bdkpimumu_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKsMuMu"].flatten(),
        barmass           = df["BToKsMuMu_barMass"].flatten(),
        barMkstar_fullfit = df["BToKsMuMu_barMkstar_fullfit"].flatten(),
        chi2              = df["BToKsMuMu_chi2"].flatten(),
        cos2D             = df["BToKsMuMu_cos2D"].flatten(),
        eta               = df["BToKsMuMu_eta"].flatten(),
        etakstar_fullfit  = df["BToKsMuMu_etakstar_fullfit"].flatten(),
        fit_cos2D         = df["BToKsMuMu_fit_cos2D"].flatten(),
        fit_eta           = df["BToKsMuMu_fit_eta"].flatten(),
        fit_mass          = df["BToKsMuMu_fit_mass"].flatten(),
        fit_massErr       = df["BToKsMuMu_fit_massErr"].flatten(),
        fit_phi           = df["BToKsMuMu_fit_phi"].flatten(),
        fit_pt            = df["BToKsMuMu_fit_pt"].flatten(),
        fit_barmass       = df["BToKsMuMu_fitted_barMass"].flatten(),
        l_xy              = df["BToKsMuMu_l_xy"].flatten(),
        l_xy_unc          = df["BToKsMuMu_l_xy_unc"].flatten(),
        lep1eta_fullfit   = df["BToKsMuMu_lep1eta_fullfit"].flatten(),
        lep1phi_fullfit   = df["BToKsMuMu_lep1phi_fullfit"].flatten(),
        lep1pt_fullfit    = df["BToKsMuMu_lep1pt_fullfit"].flatten(),
        lep2eta_fullfit   = df["BToKsMuMu_lep2eta_fullfit"].flatten(),
        lep2phi_fullfit   = df["BToKsMuMu_lep2phi_fullfit"].flatten(),
        lep2pt_fullfit    = df["BToKsMuMu_lep2pt_fullfit"].flatten(),
        mass              = df["BToKsMuMu_mass"].flatten(),
        max_dr            = df["BToKsMuMu_max_dr"].flatten(),
        min_dr            = df["BToKsMuMu_min_dr"].flatten(),
        mkstar_fullfit    = df["BToKsMuMu_mkstar_fullfit"].flatten(),
        mll_fullfit       = df["BToKsMuMu_mll_fullfit"].flatten(),
        mll_llfit         = df["BToKsMuMu_mll_llfit"].flatten(),
        mll_raw           = df["BToKsMuMu_mll_raw"].flatten(),
        phi               = df["BToKsMuMu_phi"].flatten(),
        phikstar_fullfit  = df["BToKsMuMu_phikstar_fullfit"].flatten(),
        pt                = df["BToKsMuMu_pt"].flatten(),
        ptkstar_fullfit   = df["BToKsMuMu_ptkstar_fullfit"].flatten(),
        sv_prob           = df["BToKsMuMu_svprob"].flatten(),
        trk1eta_fullfit   = df["BToKsMuMu_trk1eta_fullfit"].flatten(),
        trk1phi_fullfit   = df["BToKsMuMu_trk1phi_fullfit"].flatten(),
        trk1pt_fullfit    = df["BToKsMuMu_trk1pt_fullfit"].flatten(),
        trk2eta_fullfit   = df["BToKsMuMu_trk2eta_fullfit"].flatten(),
        trk2phi_fullfit   = df["BToKsMuMu_trk2phi_fullfit"].flatten(),
        trk2pt_fullfit    = df["BToKsMuMu_trk2pt_fullfit"].flatten(),
        charge            = df["BToKsMuMu_charge"].flatten(),
        kstar_idx         = df["BToKsMuMu_kstar_idx"].flatten(),
        l1_idx            = df["BToKsMuMu_l1_idx"].flatten(),
        l2_idx            = df["BToKsMuMu_l2_idx"].flatten(),
        pdgId             = df["BToKsMuMu_pdgId"].flatten(),
        trk1_idx          = df["BToKsMuMu_trk1_idx"].flatten(),
        trk2_idx          = df["BToKsMuMu_trk2_idx"].flatten(),
    )
    reco_bdkpimumu_collection.add_attributes(
        fit_y = np.log((np.sqrt(reco_bdkpimumu_collection.fit_mass**2 + reco_bdkpimumu_collection.fit_pt**2*np.cosh(reco_bdkpimumu_collection.fit_eta)**2) + reco_bdkpimumu_collection.fit_pt*np.sinh(reco_bdkpimumu_collection.fit_eta)) / np.sqrt(reco_bdkpimumu_collection.fit_mass**2 + reco_bdkpimumu_collection.fit_pt**2))
    )

    # Additional variable: choose nominal or alternative Bd according to best K* mass
    reco_bdkpimumu_collection.add_attributes(
        nominal_kpi = (abs(reco_bdkpimumu_collection.mkstar_fullfit - KSTAR_892_MASS) <= abs(reco_bdkpimumu_collection.barMkstar_fullfit - KSTAR_892_MASS))
    )
    reco_bdkpimumu_collection.add_attributes(
        fit_best_mass = where(reco_bdkpimumu_collection.nominal_kpi, 
                        reco_bdkpimumu_collection.fit_mass,
                        reco_bdkpimumu_collection.fit_barmass
                        ),
        fit_best_barmass = where(reco_bdkpimumu_collection.nominal_kpi, 
                        reco_bdkpimumu_collection.fit_barmass,
                        reco_bdkpimumu_collection.fit_mass
                        ),
        mkstar_best_fullfit = where(reco_bdkpimumu_collection.nominal_kpi, 
                        reco_bdkpimumu_collection.mkstar_fullfit,
                        reco_bdkpimumu_collection.barMkstar_fullfit
                        ),
        barmkstar_best_fullfit = where(reco_bdkpimumu_collection.nominal_kpi, 
                        reco_bdkpimumu_collection.barMkstar_fullfit,
                        reco_bdkpimumu_collection.mkstar_fullfit
                        )
    )

    # TLorentzVector arrays for tracks and muons
    """
    k_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKsMuMu"].flatten(),
        pt = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk1pt_fullfit,
                reco_bdkpimumu_collection.trk2pt_fullfit).flatten(),
        eta = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk1eta_fullfit,
                reco_bdkpimumu_collection.trk2eta_fullfit).flatten(),
        phi = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk1phi_fullfit,
                reco_bdkpimumu_collection.trk2phi_fullfit).flatten(),
        mass = reco_bdkpimumu_collection.trk1pt_fullfit.ones_like().flatten() * K_MASS,
    )
    pi_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKsMuMu"].flatten(),
        pt = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk2pt_fullfit,
                reco_bdkpimumu_collection.trk1pt_fullfit).flatten(),
        eta = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk2eta_fullfit,
                reco_bdkpimumu_collection.trk1eta_fullfit).flatten(),
        phi = where(reco_bdkpimumu_collection.nominal_kpi,
                reco_bdkpimumu_collection.trk2phi_fullfit,
                reco_bdkpimumu_collection.trk1phi_fullfit).flatten(),
        mass = reco_bdkpimumu_collection.trk1pt_fullfit.ones_like().flatten() * PI_MASS,
    )
    l1_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKsMuMu"].flatten(),
        pt = reco_bdkpimumu_collection.lep1pt_fullfit.flatten(),
        eta = reco_bdkpimumu_collection.lep1eta_fullfit.flatten(),
        phi = reco_bdkpimumu_collection.lep1phi_fullfit.flatten(),
        mass = reco_bdkpimumu_collection.lep1pt_fullfit.ones_like().flatten() * MU_MASS,
    )
    l2_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKsMuMu"].flatten(),
        pt = reco_bdkpimumu_collection.lep2pt_fullfit.flatten(),
        eta = reco_bdkpimumu_collection.lep2eta_fullfit.flatten(),
        phi = reco_bdkpimumu_collection.lep2phi_fullfit.flatten(),
        mass = reco_bdkpimumu_collection.lep2pt_fullfit.ones_like().flatten() * MU_MASS,
    )
    reco_bdkpimumu_collection.add_attributes(
        k_p4  = k_p4, 
        pi_p4 = pi_p4,
        l1_p4 = l1_p4,
        l2_p4 = l2_p4
    )
    """

    return reco_bdkpimumu_collection


def reco_bskkmumu(df, is_mc=False):
    reco_bskkmumu_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        chi2        = df["BToPhiMuMu_chi2"].flatten(),
        cos2D       = df["BToPhiMuMu_cos2D"].flatten(),
        eta         = df["BToPhiMuMu_eta"].flatten(),
        phi_eta     = df["BToPhiMuMu_etaphi_fullfit"].flatten(),
        phi_m       = df["BToPhiMuMu_mphi_fullfit"].flatten(),
        phi_phi     = df["BToPhiMuMu_phiphi_fullfit"].flatten(),
        phi_pt      = df["BToPhiMuMu_ptphi_fullfit"].flatten(),
        fit_cos2D   = df["BToPhiMuMu_fit_cos2D"].flatten(),
        fit_eta     = df["BToPhiMuMu_fit_eta"].flatten(),
        fit_mass    = df["BToPhiMuMu_fit_mass"].flatten(),
        fit_massErr = df["BToPhiMuMu_fit_massErr"].flatten(),
        fit_phi     = df["BToPhiMuMu_fit_phi"].flatten(),
        fit_pt      = df["BToPhiMuMu_fit_pt"].flatten(),
        l_xy        = df["BToPhiMuMu_l_xy"].flatten(),
        l_xy_unc    = df["BToPhiMuMu_l_xy_unc"].flatten(),
        l1_eta      = df["BToPhiMuMu_lep1eta_fullfit"].flatten(),
        l1_phi      = df["BToPhiMuMu_lep1phi_fullfit"].flatten(),
        l1_pt       = df["BToPhiMuMu_lep1pt_fullfit"].flatten(),
        l2_eta      = df["BToPhiMuMu_lep2eta_fullfit"].flatten(),
        l2_phi      = df["BToPhiMuMu_lep2phi_fullfit"].flatten(),
        l2_pt       = df["BToPhiMuMu_lep2pt_fullfit"].flatten(),
        mass        = df["BToPhiMuMu_mass"].flatten(),
        max_dr      = df["BToPhiMuMu_max_dr"].flatten(),
        min_dr      = df["BToPhiMuMu_min_dr"].flatten(),
        mll_fullfit = df["BToPhiMuMu_mll_fullfit"].flatten(),
        mll_llfit   = df["BToPhiMuMu_mll_llfit"].flatten(),
        mll_raw     = df["BToPhiMuMu_mll_raw"].flatten(),
        phi         = df["BToPhiMuMu_phi"].flatten(),
        pt          = df["BToPhiMuMu_pt"].flatten(),
        sv_prob     = df["BToPhiMuMu_svprob"].flatten(),
        trk1_eta    = df["BToPhiMuMu_trk1eta_fullfit"].flatten(),
        trk1_phi    = df["BToPhiMuMu_trk1phi_fullfit"].flatten(),
        trk1_pt     = df["BToPhiMuMu_trk1pt_fullfit"].flatten(),
        trk2_eta    = df["BToPhiMuMu_trk2eta_fullfit"].flatten(),
        trk2_phi    = df["BToPhiMuMu_trk2phi_fullfit"].flatten(),
        trk2_pt     = df["BToPhiMuMu_trk2pt_fullfit"].flatten(),
        charge      = df["BToPhiMuMu_charge"].flatten(),
        l1_idx      = df["BToPhiMuMu_l1_idx"].flatten(),
        l2_idx      = df["BToPhiMuMu_l2_idx"].flatten(),
        pdgId       = df["BToPhiMuMu_pdgId"].flatten(),
        phi_idx     = df["BToPhiMuMu_phi_idx"].flatten(),
        trk1_idx    = df["BToPhiMuMu_trk1_idx"].flatten(),
        trk2_idx    = df["BToPhiMuMu_trk2_idx"].flatten(),
    )
    #reco_bskkmumu_collection.add_attributes(
    #    fit_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(reco_bskkmumu_collection.fit_pt, reco_bskkmumu_collection.fit_eta, reco_bskkmumu_collection.fit_phi, reco_bskkmumu_collection.fit_mass)
    #)
    
    # TLorentzVector arrays for tracks and muons
    """
    trk1_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk1_pt.flatten(),
        eta = reco_bskkmumu_collection.trk1_eta.flatten(),
        phi = reco_bskkmumu_collection.trk1_phi.flatten(),
        mass = reco_bskkmumu_collection.trk1_pt.ones_like().flatten() * K_MASS,
    )
    trk2_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk2_pt.flatten(),
        eta = reco_bskkmumu_collection.trk2_eta.flatten(),
        phi = reco_bskkmumu_collection.trk2_phi.flatten(),
        mass = reco_bskkmumu_collection.trk2_pt.ones_like().flatten() * K_MASS,
    )
    l1_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.l1_pt.flatten(),
        eta = reco_bskkmumu_collection.l1_eta.flatten(),
        phi = reco_bskkmumu_collection.l1_phi.flatten(),
        mass = reco_bskkmumu_collection.l1_pt.ones_like().flatten() * MU_MASS,
    )
    l2_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.l2_pt.flatten(),
        eta = reco_bskkmumu_collection.l2_eta.flatten(),
        phi = reco_bskkmumu_collection.l2_phi.flatten(),
        mass = reco_bskkmumu_collection.l2_pt.ones_like().flatten() * MU_MASS,
    )
    reco_bskkmumu_collection.add_attributes(
        trk1_p4 = trk1_p4, 
        trk2_p4 = trk2_p4,
        l1_p4   = l1_p4,
        l2_p4   = l2_p4
    )
    """
    reco_bskkmumu_collection.add_attributes(
        fit_y = np.log((np.sqrt(reco_bskkmumu_collection.fit_mass**2 + reco_bskkmumu_collection.fit_pt**2*np.cosh(reco_bskkmumu_collection.fit_eta)**2) + reco_bskkmumu_collection.fit_pt*np.sinh(reco_bskkmumu_collection.fit_eta)) / np.sqrt(reco_bskkmumu_collection.fit_mass**2 + reco_bskkmumu_collection.fit_pt**2))
    )

    # Add di-track K/pi mass, for K* > K pi veto
    Kstar_cand1_K = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk1_pt.flatten(),
        eta = reco_bskkmumu_collection.trk1_eta.flatten(),
        phi = reco_bskkmumu_collection.trk1_phi.flatten(),
        mass = reco_bskkmumu_collection.trk1_pt.ones_like().flatten() * K_MASS,
    )
    Kstar_cand1_pi = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk2_pt.flatten(),
        eta = reco_bskkmumu_collection.trk2_eta.flatten(),
        phi = reco_bskkmumu_collection.trk2_phi.flatten(),
        mass = reco_bskkmumu_collection.trk2_pt.ones_like().flatten() * PI_MASS,
    )
    Kstar_cand2_K = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk2_pt.flatten(),
        eta = reco_bskkmumu_collection.trk2_eta.flatten(),
        phi = reco_bskkmumu_collection.trk2_phi.flatten(),
        mass = reco_bskkmumu_collection.trk2_pt.ones_like().flatten() * K_MASS,
    )
    Kstar_cand2_pi = JaggedCandidateArray.candidatesfromcounts(
        df["nBToPhiMuMu"].flatten(),
        pt = reco_bskkmumu_collection.trk1_pt.flatten(),
        eta = reco_bskkmumu_collection.trk1_eta.flatten(),
        phi = reco_bskkmumu_collection.trk1_phi.flatten(),
        mass = reco_bskkmumu_collection.trk1_pt.ones_like().flatten() * PI_MASS,
    )
    reco_bskkmumu_collection.add_attributes(
        Kstar1_mass = (Kstar_cand1_K.p4 + Kstar_cand1_pi.p4).mass,
        Kstar2_mass = (Kstar_cand2_K.p4 + Kstar_cand2_pi.p4).mass,
    )

    return reco_bskkmumu_collection

def reco_bukmumu(df, is_mc=False):
    reco_bukmumu_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKMuMu"].flatten(),
        chi2         = df["BToKMuMu_chi2"].flatten(),
        cos2D        = df["BToKMuMu_cos2D"].flatten(),
        eta          = df["BToKMuMu_eta"].flatten(),
        fit_cos2D    = df["BToKMuMu_fit_cos2D"].flatten(),
        fit_eta      = df["BToKMuMu_fit_eta"].flatten(),
        fit_k_eta    = df["BToKMuMu_fit_k_eta"].flatten(),
        fit_k_phi    = df["BToKMuMu_fit_k_phi"].flatten(),
        fit_k_pt     = df["BToKMuMu_fit_k_pt"].flatten(),
        fit_l1_eta   = df["BToKMuMu_fit_l1_eta"].flatten(),
        fit_l1_phi   = df["BToKMuMu_fit_l1_phi"].flatten(),
        fit_l1_pt    = df["BToKMuMu_fit_l1_pt"].flatten(),
        fit_l2_eta   = df["BToKMuMu_fit_l2_eta"].flatten(),
        fit_l2_phi   = df["BToKMuMu_fit_l2_phi"].flatten(),
        fit_l2_pt    = df["BToKMuMu_fit_l2_pt"].flatten(),
        fit_mass     = df["BToKMuMu_fit_mass"].flatten(),
        fit_massErr  = df["BToKMuMu_fit_massErr"].flatten(),
        fit_phi      = df["BToKMuMu_fit_phi"].flatten(),
        fit_pt       = df["BToKMuMu_fit_pt"].flatten(),
        l_xy         = df["BToKMuMu_l_xy"].flatten(),
        l_xy_unc     = df["BToKMuMu_l_xy_unc"].flatten(),
        mass         = df["BToKMuMu_mass"].flatten(),
        maxDR        = df["BToKMuMu_maxDR"].flatten(),
        minDR        = df["BToKMuMu_minDR"].flatten(),
        mllErr_llfit = df["BToKMuMu_mllErr_llfit"].flatten(),
        mll_fullfit  = df["BToKMuMu_mll_fullfit"].flatten(),
        mll_llfit    = df["BToKMuMu_mll_llfit"].flatten(),
        mll_raw      = df["BToKMuMu_mll_raw"].flatten(),
        phi          = df["BToKMuMu_phi"].flatten(),
        pt           = df["BToKMuMu_pt"].flatten(),
        sv_prob      = df["BToKMuMu_svprob"].flatten(),
        vtx_ex       = df["BToKMuMu_vtx_ex"].flatten(),
        vtx_ey       = df["BToKMuMu_vtx_ey"].flatten(),
        vtx_ez       = df["BToKMuMu_vtx_ez"].flatten(),
        vtx_x        = df["BToKMuMu_vtx_x"].flatten(),
        vtx_y        = df["BToKMuMu_vtx_y"].flatten(),
        vtx_z        = df["BToKMuMu_vtx_z"].flatten(),
        charge       = df["BToKMuMu_charge"].flatten(),
        kIdx         = df["BToKMuMu_kIdx"].flatten(),
        l1_idx       = df["BToKMuMu_l1Idx"].flatten(),
        l2_idx       = df["BToKMuMu_l2Idx"].flatten(),
        pdgId        = df["BToKMuMu_pdgId"].flatten(),
    )
    reco_bukmumu_collection.add_attributes(
        fit_y = np.log((np.sqrt(reco_bukmumu_collection.fit_mass**2 + reco_bukmumu_collection.fit_pt**2*np.cosh(reco_bukmumu_collection.fit_eta)**2) + reco_bukmumu_collection.fit_pt*np.sinh(reco_bukmumu_collection.fit_eta)) / np.sqrt(reco_bukmumu_collection.fit_mass**2 + reco_bukmumu_collection.fit_pt**2))
    )
    
    #reco_bukmumu_collection.add_attributes(
    #    fit_p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(reco_bukmumu_collection.fit_pt, reco_bukmumu_collection.fit_eta, reco_bukmumu_collection.fit_phi, reco_bukmumu_collection.fit_mass)
    #)
    
    """
    k_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKMuMu"].flatten(),
        pt = reco_bukmumu_collection.fit_k_pt,
        eta = reco_bukmumu_collection.fit_k_eta,
        phi = reco_bukmumu_collection.fit_k_phi,
        mass = reco_bukmumu_collection.fit_k_pt.ones_like() * K_MASS
    )
    l1_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKMuMu"].flatten(),
        pt = reco_bukmumu_collection.fit_l1_pt,
        eta = reco_bukmumu_collection.fit_l1_eta,
        phi = reco_bukmumu_collection.fit_l1_phi,
        mass = reco_bukmumu_collection.fit_k_pt.ones_like() * MU_MASS
    )
    l2_p4 = JaggedCandidateArray.candidatesfromcounts(
        df["nBToKMuMu"].flatten(),
        pt = reco_bukmumu_collection.fit_l2_pt,
        eta = reco_bukmumu_collection.fit_l2_eta,
        phi = reco_bukmumu_collection.fit_l2_phi,
        mass = reco_bukmumu_collection.fit_k_pt.ones_like() * MU_MASS
    )
    reco_bukmumu_collection.add_attributes(
        k_p4  = k_p4,
        l1_p4 = l1_p4,
        l2_p4 = l2_p4,
    )
    """
    return reco_bukmumu_collection


def reco_muons(df, is_mc=False):
    reco_muons_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nMuon"].flatten(),
        dxy            = df["Muon_dxy"].flatten(),
        dxyErr         = df["Muon_dxyErr"].flatten(),
        dz             = df["Muon_dz"].flatten(),
        dzErr          = df["Muon_dzErr"].flatten(),
        eta            = df["Muon_eta"].flatten(),
        ip3d           = df["Muon_ip3d"].flatten(),
        mass           = df["Muon_mass"].flatten(),
        pfRelIso03_all = df["Muon_pfRelIso03_all"].flatten(),
        pfRelIso03_chg = df["Muon_pfRelIso03_chg"].flatten(),
        pfRelIso04_all = df["Muon_pfRelIso04_all"].flatten(),
        phi            = df["Muon_phi"].flatten(),
        pt             = df["Muon_pt"].flatten(),
        ptErr          = df["Muon_ptErr"].flatten(),
        segmentComp    = df["Muon_segmentComp"].flatten(),
        sip3d          = df["Muon_sip3d"].flatten(),
        vx             = df["Muon_vx"].flatten(),
        vy             = df["Muon_vy"].flatten(),
        vz             = df["Muon_vz"].flatten(),
        charge         = df["Muon_charge"].flatten(),
        isTriggering   = (df["Muon_isTriggering"]==1).flatten(),
        nStations      = df["Muon_nStations"].flatten(),
        pdgId          = df["Muon_pdgId"].flatten(),
        tightCharge    = df["Muon_tightCharge"].flatten(),
        highPtId       = df["Muon_highPtId"].flatten(),
        inTimeMuon     = df["Muon_inTimeMuon"].flatten(),
        isGlobal       = df["Muon_isGlobal"].flatten(),
        isPFcand       = df["Muon_isPFcand"].flatten(),
        isTracker      = df["Muon_isTracker"].flatten(),
        mediumId       = df["Muon_mediumId"].flatten(),
        mediumPromptId = df["Muon_mediumPromptId"].flatten(),
        miniIsoId      = df["Muon_miniIsoId"].flatten(),
        multiIsoId     = df["Muon_multiIsoId"].flatten(),
        mvaId          = df["Muon_mvaId"].flatten(),
        pfIsoId        = df["Muon_pfIsoId"].flatten(),
        softId         = df["Muon_softId"].flatten(),
        softMvaId      = df["Muon_softMvaId"].flatten(),
        tightId        = df["Muon_tightId"].flatten(),
        tkIsoId        = df["Muon_tkIsoId"].flatten(),
        triggerIdLoose = df["Muon_triggerIdLoose"].flatten(),
    )
    if is_mc:
        reco_muons_collection.add_attributes(
            genPartIdx     = df["Muon_genPartIdx"].flatten(),
            genPartFlav    = df["Muon_genPartFlav"].flatten(),
        )
    return reco_muons_collection

def trigger_muons(df, is_mc=False):
    trigger_muons_collection = JaggedCandidateArray.candidatesfromcounts(
        df["nTriggerMuon"].flatten(),
        eta          = df["TriggerMuon_eta"].flatten(),
        mass         = df["TriggerMuon_mass"].flatten(),
        phi          = df["TriggerMuon_phi"].flatten(),
        pt           = df["TriggerMuon_pt"].flatten(),
        vx           = df["TriggerMuon_vx"].flatten(),
        vy           = df["TriggerMuon_vy"].flatten(),
        vz           = df["TriggerMuon_vz"].flatten(),
        charge       = df["TriggerMuon_charge"].flatten(),
        pdgId        = df["TriggerMuon_pdgId"].flatten(),
        trgMuonIndex = df["TriggerMuon_trgMuonIndex"].flatten(),
    )
    return trigger_muons_collection
