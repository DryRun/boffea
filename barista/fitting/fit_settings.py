import ROOT

BU_FIT_WINDOW = [5.05, 5.5]
BD_FIT_WINDOW = [5.05, 5.4]
BS_FIT_WINDOW = [5.2, 5.52]

BU_FIT_WINDOW_MC = [5.1, 5.45]
BD_FIT_WINDOW_MC = [5.05, 5.5]
BS_FIT_WINDOW_MC = [5.2 + 0.05, 5.52 - 0.05]

BU_FIT_NBINS = 100
BD_FIT_NBINS = 100
BS_FIT_NBINS = 100

# Fit bins
#ptbins_probe = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
ptbins_probe = [8., 13., 18., 23., 28., 33.]
ptbins_tag = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 34.0, 45.0, 60.0, 1000.0] # 10.0, 
ybins = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.0, 2.25]

fit_cuts = {"tag": [], "probe": [], "tagx": []}
cut_strings = {}
fit_text = {} # Text for plots

fit_cuts["tag"].append("inclusive")
fit_cuts["probe"].append("inclusive")
cut_strings["inclusive"] = "1"
fit_text["inclusive"] = "Inclusive"

cut_xvals = {}

# pT, probe
for ipt in range(len(ptbins_probe) - 1):
    cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins_probe[ipt], ptbins_probe[ipt+1])
    cut_name = "ptbin_{}_{}".format(ptbins_probe[ipt], ptbins_probe[ipt+1]).replace(".", "p")
    fit_cuts["probe"].append(cut_name)
    #fit_cuts["tag"].append(cut_name)
    cut_strings[cut_name] = cut_str
    fit_text[cut_name] = "{}<pT<{}".format(ptbins_probe[ipt], ptbins_probe[ipt+1])
    cut_xvals[cut_name] = (ptbins_probe[ipt], ptbins_probe[ipt+1])

# pT, tag
for ipt in range(len(ptbins_tag) - 1):
    cut_str = "(abs(y) < 1.5) && (pt > {}) && (pt < {})".format(ptbins_tag[ipt], ptbins_tag[ipt+1])
    cut_name = "ptbin_{}_{}".format(ptbins_tag[ipt], ptbins_tag[ipt+1]).replace(".", "p")
    fit_cuts["tag"].append(cut_name)
    cut_strings[cut_name] = cut_str
    fit_text[cut_name] = "{}<pT<{}".format(ptbins_tag[ipt], ptbins_tag[ipt+1])
    cut_xvals[cut_name] = (ptbins_tag[ipt], ptbins_tag[ipt+1])

# y, both sides
for iy in range(len(ybins) - 1):
    cut_str = "(pt > 12.0) && (pt < 45.0) && ({} < abs(y)) && (abs(y) < {})".format(ybins[iy], ybins[iy+1])
    cut_name = "ybin_{}_{}".format(ybins[iy], ybins[iy+1]).replace(".", "p")
    fit_cuts["probe"].append(cut_name)
    if ybins[iy+1] <= 1.75:
        fit_cuts["tag"].append(cut_name)
    cut_strings[cut_name] = cut_str
    fit_text[cut_name] = "{}<|y|<{}".format(ybins[iy], ybins[iy+1])
    cut_xvals[cut_name] = (ybins[iy], ybins[iy+1])

fit_cuts["tagx"] = fit_cuts["tag"]

def MakeSymHypatia(ws, mass_range, tag="", rcache=None):
    name = "signal_{}".format(tag)
    hyp_x      = ws.var("mass")
    hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
    hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
    hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
    hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
    hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
    hyp_a      = ws.factory("hyp_a{}[3., 1.0, 4.0]".format(tag))
    hyp_n      = ws.factory("hyp_n{}[5., 0.5,  20.]".format(tag))
    #hyp_a2     = ws.factory("hyp_a2[10., 0., 100.]")
    #hyp_n2     = ws.factory("hyp_n2[5., 0., 100.]")
    hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a, hyp_n)
    if rcache:
        rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n])
    return hyp_pdf

def MakeHypatia(ws, mass_range, tag="", rcache=None):
    name = "signal_{}".format(tag)
    hyp_x      = ws.var("mass")
    hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
    hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
    hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
    hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
    hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
    hyp_a      = ws.factory("hyp_a{}[3., 1.0, 4.0]".format(tag))
    hyp_n      = ws.factory("hyp_n{}[5., 0.75,  20.]".format(tag))
    hyp_a2     = ws.factory("hyp_a2{}[3.0, 1.0, 4.0]".format(tag))
    hyp_n2     = ws.factory("hyp_n2{}[5., 0.75,  20.]".format(tag))
    hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2)
    if rcache:
        rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2])
    return hyp_pdf

def MakeHypatia2(ws, mass_range, tag="", rcache=None):
    '''
    Hypatia function with independent left and right tails, but implemented as a2 = r_a * a ; n2 = r_n * n, so the r can be frozen to 1 to be symmetric.
    '''
    name = "signal{}".format(tag)

    hyp_x      = ws.var("mass")
    hyp_lambda = ws.factory("hyp_lambda{}[ -1., -10., -0.00001]".format(tag))
    hyp_zeta   = ws.factory("hyp_zeta{}[0.]".format(tag))
    hyp_beta   = ws.factory("hyp_beta{}[0.]".format(tag))
    hyp_sigma  = ws.factory("hyp_sigma{}[0.01, 0.005, 0.3]".format(tag))
    hyp_mu     = ws.factory(f"hyp_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
    hyp_a      = ws.factory("hyp_a{}[5., 2.0, 100.]".format(tag))
    hyp_n      = ws.factory("hyp_n{}[5., 0.5,  20.]".format(tag))
    hyp_lr_a   = ws.factory("hyp_lr_a{}[1., 0.01, 100.]".format(tag))
    hyp_lr_n   = ws.factory("hyp_lr_n{}[1., 0.01, 100.]".format(tag))
    hyp_a2     = ws.factory("prod::hyp_a2{}(hyp_a{}, hyp_lr_a{})".format(tag, tag, tag))
    hyp_n2     = ws.factory("prod::hyp_n2{}(hyp_n{}, hyp_lr_n{})".format(tag, tag, tag))
    hyp_pdf    = ROOT.RooHypatia2(name, name, hyp_x, hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2)

    if rcache:
        rcache.extend([hyp_lambda, hyp_zeta, hyp_beta, hyp_sigma, hyp_mu, hyp_a, hyp_n, hyp_a2, hyp_n2])
    return hyp_pdf

def MakeJohnson(ws, mass_range, tag="", rcache=None):
    '''
    Johnson function
    '''
    name = "signal{}".format(tag)
    j_x = ws.var("mass")
    j_mu = ws.factory(f"j_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
    j_gamma = ws.factory(f"j_gamma{tag}[0., -10., 10.]")
    j_delta = ws.factory(f"j_delta{tag}[1.0, 0.1, 10.0]")

    j_lambda = ws.factory(f"j_lambda{tag}[0.05, 0.001, 1.0]")
    #j_variance = ws.factory(f"j_var{tag}[0.02, 0., 0.1]")
    #j_lambda = ws.factory(f"")


    johnson_pdf = ROOT.RooJohnson(name, name, j_x, j_mu, j_lambda, j_gamma, j_delta)
    if rcache:
        rcache.extend([j_mu, j_lambda, j_gamma, j_delta])
    return johnson_pdf

def MakeDoubleGaussian(ws, mass_range, tag="", rcache=None):
    '''
    Double gaussian function with common mean
    '''
    name = "signal{}".format(tag)
    dg_x = ws.var("mass")
    dg_mu = ws.factory(f"dg_mu{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")

    dg_sigma1 = ws.factory(f"dg_sigma1{tag}[0.016, 0.008, 0.1]")
    dg_sigma2 = ws.factory(f"dg_sigma2{tag}[0.05, 0.02, 0.5]")

    gauss1 = ROOT.RooGaussian(f"gauss1{tag}", f"gauss1{tag}", dg_x, dg_mu, dg_sigma1)
    gauss2 = ROOT.RooGaussian(f"gauss2{tag}", f"gauss2{tag}", dg_x, dg_mu, dg_sigma2)
    getattr(ws, "import")(gauss1, ROOT.RooFit.RecycleConflictNodes())
    getattr(ws, "import")(gauss2, ROOT.RooFit.RecycleConflictNodes())
    alpha = ws.factory(f"alpha{tag}[0.95, 0., 1.0]")

    dg_pdf = ws.factory(f"SUM::{name}(alpha{tag}*gauss1{tag},gauss2{tag})")
    if rcache:
        rcache.extend([dg_x, dg_mu, dg_sigma1, dg_sigma1, alpha])
    return dg_pdf

def MakeTripleGaussian(ws, mass_range, tag="", rcache=None):
    ROOT.gSystem.Load("include/TripleGaussianPdf_cc.so")
    from ROOT import TripleGaussianPdf
    name = "signal{}".format(tag)

    tg_x = ws.var("mass")
    tg_mean = ws.factory(f"tg_mean{tag}[{0.5*(mass_range[0]+mass_range[1])}, {mass_range[0]}, {mass_range[1]}]")
    tg_s1 = ws.factory(f"tg_sigma1{tag}[0.012, 0.008, 0.15]")
    tg_s2 = ws.factory(f"tg_sigma2{tag}[0.02, 0.01, 0.2]")
    tg_s3 = ws.factory(f"tg_sigma3{tag}[0.08, 0.01, 0.3]")
    tg_aa = ws.factory(f"tg_aa{tag}[0.67, 0., 1.]")
    tg_bb = ws.factory(f"tg_bb{tag}[0.67, 0., 1.]")

    tg_pdf = TripleGaussianPdf(name, name, tg_x, tg_mean, tg_s1, tg_s2, tg_s3, tg_aa, tg_bb)

    if rcache:
        rcache.extend([tg_x, tg_mean, tg_s1, tg_s2, tg_s3, tg_aa, tg_bb])
    return tg_pdf

# Stuff for handling different selections:

def get_Bcands_name_data(btype, side, trigger, selection="nominal"):
    if selection == "HiTrkPt":
        Bcands_name = f"Bcands_{btype}_{side}HiTrkPt_{trigger}"
    elif "MuonPt" in selection:
        Bcands_name = f"Bcands_{btype}_{side}{selection}_{trigger}"
    else:
        Bcands_name = f"Bcands_{btype}_{side}_{trigger}"
    Bcands_name = Bcands_name.replace("tag_probebins", "tag")
    return Bcands_name

def get_Bcands_name_mc(btype, side, trigger, selection="nominal"):
    if "MuonPt" in selection or "HiTrkPt" in selection:
        selection_str = selection
    elif selection == "nominal":
        selection_str = ""
    else:
        selection_str = ""

    if btype == "Bd": 
        name = f"Bcands_{side}{selection_str}_{trigger}_Bd2KsJpsi2KPiMuMu_probefilter"

    elif btype == "Bs": 
        name = f"Bcands_{side}{selection_str}_{trigger}_Bs2PhiJpsi2KKMuMu_probefilter"

    elif btype == "Bu": 
        name = f"Bcands_{side}{selection_str}_{trigger}_Bu2KJpsi2KMuMu_probefilter"

    # Remove trigger from recomatch
    if "recomatch" in side:
        name = name.replace(f"_{trigger}", "")
    return name

def get_MC_fit_params(btype, selection="nominal", fitfunc="johnson", frozen="False"):
    if btype == "Bd":
        fit_params_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bd/prefitparams_MC_Bd_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_params_file.replace(".pkl", "_frozen.pkl")

    elif btype == "Bu":
        fit_params_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bu/fitparams_MC_Bu_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_params_file.replace(".pkl", "_frozen.pkl")

    elif btype == "Bs":
        fit_params_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bs/fitparams_MC_Bs_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_params_file.replace(".pkl", "_frozen.pkl")
    return fit_params_file

def get_MC_fit_errs(btype, selection="nominal", fitfunc="johnson", frozen="False"):
    if btype == "Bd":
        fit_errs_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bd/prefiterrs_MC_Bd_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_errs_file.replace(".pkl", "_frozen.pkl")

    elif btype == "Bu":
        fit_errs_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bu/fiterrs_MC_Bu_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_errs_file.replace(".pkl", "_frozen.pkl")

    elif btype == "Bs":
        fit_errs_file = f"/home/dryu/BFrag/boffitting/barista/fitting/Bs/fiterrs_MC_Bs_{selection}_{fitfunc}.pkl"
        if frozen:
            fit_errs_file.replace(".pkl", "_frozen.pkl")
    return fit_errs_file


def eff_truth_selection(btype, side, trigger, selection="nominal", rwgt=False):
    truth_selection_names = ["inclusive", "fiducial", "matched", "unmatched"] # matched_sel
    for trigger in self._triggers:
        #truth_selection_names.extend([f"matched_tag_{trigger}", f"matched_probe_{trigger}"])
        truth_selection_names.extend([f"matched_fid_tag_{trigger}", f"matched_fid_probe_{trigger}", f"matched_fid_tagx_{trigger}"])
        #truth_selection_names.extend([f"matched_tagHiTrkPt_{trigger}", f"matched_probeHiTrkPt_{trigger}"])
        truth_selection_names.extend([f"matched_fid_tagHiTrkPt_{trigger}", f"matched_fid_probeHiTrkPt_{trigger}"])
        truth_selection_names.extend([f"matched_fid_tagHiMuonPt_{trigger}", f"matched_fid_probeHiMuonPt_{trigger}"])

    for selection_name in truth_selection_names:
        output["truth_cutflow"][dataset_name][selection_name] = truth_selections[selection_name].sum().sum()

    if selection == "HiTrkPt":
        name = f"Bcands_{btype}_{side}HiTrkPt_{trigger}"
    elif selection == "HiMuonPt":
        name = f"Bcands_{btype}_{side}HiMuonPt_{trigger}"
    else:
        name = f"Bcands_{btype}_{side}_{trigger}"
    if rwgt:
        name += "_rwgt"
    return name