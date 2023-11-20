from glob import glob
import ROOT
from brazil.aguapreta import *
import math
import array

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
ROOT.gStyle.SetOptTitle(False)

# Load events
chain_data = ROOT.TChain("Bcands_Bs_opt")
for f in glob("/home/dryu/BFrag/data/histograms/Run2018*root"):
    chain_data.Add(f)
chain_mc = ROOT.TChain("Bcands_Bs_opt_Bs2PhiJpsi2KKMuMu_probefilter")
chain_mc.Add("/home/dryu/BFrag/boffea/barista/fitting/optimization_Bs_mc.root")
#chain_mc.Add("")
#chain_mc.Print()

preselection_cuts = {
    "probe": "(l1_pt > 1.5) && (l2_pt > 1.5) && (pt < 12.0) && (dm_phi < 0.01) && (k1_pt > 0.5) && (k2_pt > 0.5) && (l_xy_sig > 2.0) && (cos2D > 0.99) && (sv_prob > 0.07)", 
    "tag": "(l1_pt > 1.5) && (l2_pt > 1.5) && (((l1_pt > 7*1.05) && (TMath::Abs(l1_eta)<1.5)) || ((l2_pt > 7*1.05) && (TMath::Abs(l2_eta)<1.5))) (pt < 15.0) && (dm_phi < 0.01) && (k1_pt > 0.5) && (k2_pt > 0.5) && (l_xy_sig > 2.0) && (cos2D > 0.99) && (sv_prob > 0.07)", 

}
training_vars = ["cos2D", "l_xy_sig", "k1_pt", "k2_pt"] # , "l1_pt", "l2_pt" # , "dm_phi", "sv_prob"

varprops = {
    "l_xy": "FMax",
    "l_xy_sig": "FMax", 
    "sv_prob": "FMax", 
    "cos2D": "FMax", 
    "dm_phi": "FMin", 
    "l1_pt": "FMax", 
    "l2_pt": "FMax", 
    "k1_pt": "FMax", 
    "k2_pt": "FMax"
}
var_cutranges = {
    #"l_xy": [0.0, 0.5],
    "l_xy_sig": [1.5, 10.0], 
    #"sv_prob": [0.03, 0.1], 
    #"cos2D": [0.8, 1.0], 
    #"dm_phi": [0.007, 0.030], 
    #"l1_pt": [1.5, 5.0], 
    #"l2_pt": [1.5, 5.0], 
    #"k1_pt": [0.5, 1.5], 
    #"k2_pt": [0.5, 1.5]
}
bins = {
    "l_xy": (100, 0., 1.0),
    "l_xy_sig": (100, 0., 10.0),
    "sv_prob": (100, 0., 0.5),
    "cos2D": (100, 0.98, 1.0001),
    "dm_phi": (100, 0., 0.05),
    "dm_jpsi": (100, 0., 0.5),
    "l1_pt": (100, 0.0, 10.0),
    "l2_pt": (100, 0.0, 10.0),
    "k1_pt": (100, 0.0, 5.0),
    "k2_pt": (100, 0.0, 5.0),
    "mass": (100, 5.2, 5.52), 
    "pt": (100, 5., 15.),
    "y": (50, -2.5, 2.5), 
    "eta": (50, -2.5, 2.5),
    "phi": (50, -1*math.pi, math.pi)
}
xlabels = {
    "mass": r"M [GeV]",
    "pt": r"p_{T} [GeV]",
    "y": r"y",
    "eta": r"#eta",
    "phi": r"#phi",
    "l_xy": r"L_{xy}",
    "l_xy_sig": r"L_{xy}/#sigma_{L_{xy}}",
    "sv_prob": "SV prob.",
    "cos2D": r"cos#theta_{2D}",
    "dm_phi": r"|M_{K^{+}K^{-}} - M_{\phi}| [GeV]",
    "dm_jpsi": r"|M_{#mu^{+}#mu^{-}} - M_{J/\psi}| [GeV]",
    "l1_pt": r"p_{T}(#mu_{1}) [GeV]",
    "l2_pt": r"p_{T}(#mu_{2}) [GeV]",
    "k1_pt": r"p_{T}(K_{1}^{\pm}) [GeV]",
    "k2_pt": r"p_{T}(K_{2}^{\pm}) [GeV]",
    "mass": "M [GeV]",
}

def GetSBFractions(chain_data, side):
    # Determine initial s and b by fitting data TChain
    f_sb = ROOT.TFile(f"opt_sbfit_{side}.root", "RECREATE")
    ws = ROOT.RooWorkspace('ws')
    ws.Clear()
    mass = ws.factory(f"mass[5.35, 5.2, 5.52]")
    pt = ws.factory(f"pt[10., 0., 100.]")
    l1_pt = ws.factory(f"l1_pt[10., 0., 100.]")
    l2_pt = ws.factory(f"l2_pt[10., 0., 100.]")
    l1_eta = ws.factory(f"l1_eta[0., -3.0, 3.0]")
    l2_eta = ws.factory(f"l2_eta[0., -3.0, 3.0]")
    k1_pt = ws.factory(f"k1_pt[10., 0., 100.]")
    k2_pt = ws.factory(f"k2_pt[10., 0., 100.]")
    dm_phi = ws.factory(f"dm_phi[0.01., 0., 0.5]")
    l_xy_sig = ws.factory(f"l_xy_sig[5.0, 1.5, 100.0]")
    cos2D = ws.factory(f"cos2D[0.995, 0.0, 1.0]")
    sv_prob = ws.factory(f"sv_prob[0.1, 0.0, 1.0]")
    mass.setBins(100)
    rdatavars = ROOT.RooArgSet()
    rdatavars.add(mass)
    rdatavars.add(pt)
    rdatavars.add(l1_pt)
    rdatavars.add(l2_pt)
    rdatavars.add(l1_eta)
    rdatavars.add(l2_eta)
    rdatavars.add(dm_phi)
    rdatavars.add(k1_pt)
    rdatavars.add(k2_pt)
    rdatavars.add(l_xy_sig)
    rdatavars.add(cos2D)
    rdatavars.add(sv_prob)
    rdataset =  ROOT.RooDataSet("fitData", "fitData", rdatavars, 
                                ROOT.RooFit.Import(chain_data), 
                                ROOT.RooFit.Cut(preselection_cuts[side])) # 
    rdatahist = ROOT.RooDataHist("fitDataBinned", "fitDataBinned", ROOT.RooArgSet(mass), rdataset)
    ndata = rdataset.sumEntries()

    getattr(ws, "import")(rdatahist)

    # Signal shape
    #mean = ws.factory(f"mean[{BS_MASS}, {BS_MASS-0.05}, {BS_MASS+0.05}]")
    #sigma1 = ws.factory("sigma1[0.008, 0.0005, 0.05]")
    #sigma2 = ws.factory("sigma2[0.02, 0.0005, 0.05]")
    #sigma3 = ws.factory("sigma3[0.04, 0.0005, 0.05]")
    #aa = ws.factory(f"aa[0.5, 0.001, 1.0]")
    #bb = ws.factory(f"bb[0.8, 0.001, 1.0]")
    #signal_tg = ws.factory(f"GenericPdf::signal_tg('TripleGaussian(mass, mean, sigma1, sigma2, sigma3, aa, bb)', {{mass, mean, sigma1, sigma2, sigma3, aa, bb}})")
    #signal_tg = TripleGaussianPdf("signal_tg", "signal_tg", mass, mean, sigma1, sigma2, sigma3, aa, bb)
    signal_g = ws.factory(f"Gaussian::g(mass, mean[{BS_MASS}, {BS_MASS-0.01}, {BS_MASS+0.01}], sigma[0.01, 0.005, 0.02])")
    #getattr(ws, "import")(signal_tg, ROOT.RooFit.RecycleConflictNodes())
    nsignal = ws.factory(f"nsignal[500, 1.0, {ndata}]")
    signal_model = ROOT.RooExtendPdf("signal_model", "signal_model", signal_g, nsignal)
    getattr(ws, "import")(signal_model, ROOT.RooFit.RecycleConflictNodes())

    # Background model: exponential
    bkgd_exp = ws.factory(f"Exponential::bkgd_exp(mass, alpha[-3.66, -100., -0.01])")
    nbkgd_exp = ws.factory(f"nbkgd[{ndata*0.5}, 0.0, {ndata*2.0}]")
    bkgd_exp_model = ROOT.RooExtendPdf("bkgd_exp_model", "bkgd_exp_model", bkgd_exp, nbkgd_exp)
    getattr(ws, "import")(bkgd_exp_model, ROOT.RooFit.RecycleConflictNodes())

    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(signal_model, bkgd_exp_model))
    getattr(ws, "import")(model, ROOT.RooFit.RecycleConflictNodes())

    fit_args = [rdatahist, ROOT.RooFit.NumCPU(8), ROOT.RooFit.Save()]
    print("Starting fit...")
    fit_result = model.fitTo(*fit_args)
    fit_result.Print()
    print("Done with fit.")

    #print(BS_MASS)
    print("sigma = {}".format(ws.var("sigma").getVal()))
    mass.setRange("sigwin", ws.var("mean").getVal()-0.015, ws.var("mean").getVal()+0.015)
    sfrac = signal_g.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), "sigwin")
    sfrac.Print()
    s = sfrac.getVal() * nsignal.getVal()
    print(f"sfrac = {sfrac.getVal()}, nsignal={nsignal.getVal()}, s={s}")

    bfrac = bkgd_exp_model.createIntegral(ROOT.RooArgSet(mass), ROOT.RooArgSet(mass), "sigwin")
    b = bfrac.getVal() * nbkgd_exp.getVal()
    print(f"bfrac = {bfrac.getVal()}, nbkgd={nbkgd_exp.getVal()}, b={b}")

    svar = ws.factory(f"s[{s}, 0., {s*20}]")
    bvar = ws.factory(f"b[{b}, 0., {b*20}]")
    getattr(ws, "import")(rdatahist)

    f_sb.cd()
    ws.Write("ws")
    f_sb.Close()

    # Plot fit for sanity check
    c_fit = ROOT.TCanvas("c_sb_fit", "c_sb_fit", 600, 400)
    rplot = mass.frame()#ROOT.RooFit.Bins(100))
    rdatahist.plotOn(rplot, ROOT.RooFit.Name("data"))
    model.plotOn(rplot, ROOT.RooFit.Name("fit"))
    model.plotOn(rplot, ROOT.RooFit.Name("signal"), ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.FillColor(ROOT.kGreen+2), ROOT.RooFit.FillStyle(3002), ROOT.RooFit.DrawOption("LF"))
    model.plotOn(rplot, ROOT.RooFit.Name("bkgd_exp"), ROOT.RooFit.Components("bkgd_exp_model"), ROOT.RooFit.LineColor(ROOT.kRed+1))
    rplot.Draw()
    # Draw lines denoting signal and background windows
    tline1 = ROOT.TLine(ws.var("mean").getVal() - 0.015, rplot.GetMinimum(), ws.var("mean").getVal() - 0.015, rplot.GetMaximum())
    tline1.SetLineStyle(3)
    tline1.SetLineColor(ROOT.kGray)
    tline1.Draw()
    tline2 = ROOT.TLine(ws.var("mean").getVal() + 0.015, rplot.GetMinimum(), ws.var("mean").getVal() + 0.015, rplot.GetMaximum())
    tline2.SetLineStyle(3)
    tline2.SetLineColor(ROOT.kGray)
    tline2.Draw()

    tline3 = ROOT.TLine(BS_MASS - 0.05, rplot.GetMinimum(), BS_MASS - 0.05, rplot.GetMaximum())
    tline3.SetLineStyle(3)
    tline3.SetLineColor(ROOT.kGray+1)
    tline3.Draw()
    tline4 = ROOT.TLine(BS_MASS + 0.05, rplot.GetMinimum(), BS_MASS + 0.05, rplot.GetMaximum())
    tline4.SetLineStyle(3)
    tline4.SetLineColor(ROOT.kGray+1)
    tline4.Draw()
    c_fit.SaveAs(f"/home/dryu/BFrag/data/optimization/figures/{c_fit.GetName()}_{side}.pdf")

    return s, b

def RunMethodCuts(chain_data, chain_mc):
    tmva_file = ROOT.TFile(f"tmva_cuts_{side}.root", "RECREATE")
    factory = ROOT.TMVA.Factory("cuts", tmva_file, "V:!Silent")
    dataloader = ROOT.TMVA.DataLoader("dataset")

    #training_vars = ["l_xy", "l_xy_sig", "sv_prob", "cos2D", "dm_phi", "l1_pt", "l2_pt", "k1_pt", "k2_pt"]
    for var in training_vars:
        dataloader.AddVariable(var, "F", bins[var][1], bins[var][2])
    for var in ["mass", "pt", "y", "eta", "phi"]:
        dataloader.AddSpectator(var, xlabels[var], "", bins[var][1], bins[var][2])
    dataloader.AddSignalTree(chain_mc)
    dataloader.AddBackgroundTree(chain_data)
    base_cuts = preselection_cuts[side]
    #for var in training_vars:
    #    if varprops[var] == "FMax":
    #        base_cuts += f" && ({var} > {var_cutranges[var][0]})"
    #    base_cuts += f" && ({var} < {var_cutranges[var][1]})"

    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(f"{base_cuts}"), ROOT.TCut(f"{base_cuts} && (TMath::Abs(mass - {BS_MASS}) > 0.05)"), "nTrain_Signal=0:nTrain_Background=20000")

    # Specify direction and range of cuts
    opt_list = []
    opt_list = ["FitMethod=GA", "EffMethod=EffSel"]
    for ivar, var in enumerate(training_vars):
        opt_list.append(f"VarProp[{ivar}]={varprops[var]}")
        if var in var_cutranges:
            opt_list.append(f"CutRangeMin[{ivar}]={var_cutranges[var][1]}")
            opt_list.append(f"CutRangeMax[{ivar}]={var_cutranges[var][1]}")
    opt_str = ":".join(opt_list)
    print(opt_str)
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kCuts, "cuts", opt_str)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    tmva_file.Close()

def KinematicPlots(chain_data, chain_mc):
    print("KinematicPlots")
    canvases = {}
    hist_signal = {}
    hist_bkgd = {}
    for var in training_vars + ["l1_pt", "l2_pt"]:
        canvases[var] = ROOT.TCanvas(f"c_{var}", f"c_{var}", 800, 600)
        hist_signal[var] = ROOT.TH1D(f"h_signal_{var}", f"h_signal_{var}", bins[var][0], bins[var][1], bins[var][2])
        hist_bkgd[var] = ROOT.TH1D(f"h_bkgd_{var}", f"h_bkgd_{var}", bins[var][0], bins[var][1], bins[var][2])

        print("Preselection cuts:")
        print(preselection_cuts[side])
        chain_mc.Draw(f"{var} >> h_signal_{var}", preselection_cuts[side])
        chain_data.Draw(f"{var} >> h_bkgd_{var}", f"{preselection_cuts[side]} && (TMath::Abs(mass - {BS_MASS}) > 0.05)")

        if hist_signal[var].Integral() > 0:
            hist_signal[var].Scale(1. / hist_signal[var].Integral())
        if hist_bkgd[var].Integral() > 0:
            hist_bkgd[var].Scale(1. / hist_bkgd[var].Integral())

        frame = ROOT.TH1D("frame", "frame", bins[var][0], bins[var][1], bins[var][2])
        frame.SetMaximum(1.2 * max(hist_signal[var].GetMaximum(), hist_bkgd[var].GetMaximum()))
        frame.GetXaxis().SetTitle(xlabels[var])
        frame.Draw()

        hist_signal[var].SetLineStyle(1)
        hist_signal[var].SetLineColor(ROOT.kGreen+3)
        hist_signal[var].SetLineWidth(2)
        hist_signal[var].Draw("hist same")

        hist_signal[var].SetLineStyle(2)
        hist_signal[var].SetLineColor(ROOT.kRed+1)
        hist_signal[var].SetLineWidth(2)
        hist_bkgd[var].Draw("hist same")

        legend = ROOT.TLegend(0.7, 0.7, 0.88, 0.8)
        legend.SetFillStyle(0)
        legend.SetFillColor(0)
        legend.SetBorderSize(0)
        legend.AddEntry(hist_signal[var], "Signal", "l")
        legend.AddEntry(hist_bkgd[var], "Background", "l")
        legend.Draw()

        canvases[var].SaveAs(f"/home/dryu/BFrag/data/optimization/figures/{canvases[var].GetName()}_{side}.pdf")
        canvases[var].SaveAs(f"/home/dryu/BFrag/data/optimization/figures/{canvases[var].GetName()}_{side}.png")


def ComputeSignificance():
    print("ComputeSignificance")
    # Load s and b from GetSBFractions()
    f_sb = ROOT.TFile(f"opt_sbfit_{side}.root", "READ")
    ws = f_sb.Get("ws")
    #ws = ROOT.RooWorkspace('ws')
    ws.Print()
    print(ws.var("s"))
    s = ws.var("s").getVal()
    b = ws.var("b").getVal()

    print(f"Using N_s={s}, N_b={b}")

    # Load ROC curve from RunMethodCuts()
    tmva_file = ROOT.TFile(f"tmva_cuts_{side}.root", "READ")
    hist_roc = tmva_file.Get("dataset/Method_cuts/cuts/MVA_cuts_effBvsS")

    # Make significance graph
    tg_signif = ROOT.TGraph()
    for bin in range(1, hist_roc.GetNbinsX()+1):
        eff_s = hist_roc.GetXaxis().GetBinCenter(bin)
        eff_b = hist_roc.GetBinContent(bin)
        signif = s * eff_s / math.sqrt(s * eff_s + b * eff_b)
        #print(f"eff_s = {eff_s}, eff_b = {eff_b}, signif = {signif}")
        tg_signif.SetPoint(bin-1, eff_s, signif)        

    # Fit with a polynomial
    tmp = FindMaxSmoothed(tg_signif)
    fit_signif = tmp["fit"]
    best_eff_s = tmp["max"][0]
    fit_signif.Print()

    # Plot
    c_signif = ROOT.TCanvas("c_signif", "c_signif", 800, 600)
    tg_signif.GetHistogram().GetXaxis().SetTitle("#epsilon(signal)")
    tg_signif.GetHistogram().GetYaxis().SetTitle("s/#sqrt{s+b}")
    tg_signif.GetXaxis().SetTitle("#epsilon(signal)")
    tg_signif.GetYaxis().SetTitle("s/#sqrt{s+b}")
    tg_signif.Draw("apl")
    fit_signif.SetLineColor(2)
    fit_signif.SetLineWidth(1)
    fit_signif.Draw("same")
    c_signif.SaveAs(f"/home/dryu/BFrag/data/optimization/figures/{c_signif.GetName()}_{side}.pdf")
    c_signif.SaveAs(f"/home/dryu/BFrag/data/optimization/figures/{c_signif.GetName()}_{side}.png")

    # Print cuts
    reader = ROOT.TMVA.Reader("!Color:!Silent")
    containers = {}
    for var in training_vars:
        containers[var] = array.array("f", [0.])
        reader.AddVariable(var, containers[var])
    for var in ["mass", "pt", "y", "eta", "phi"]:
        containers[var] = array.array("f", [0.])
        reader.AddSpectator(var, containers[var])
    reader.BookMVA("cuts", "dataset/weights/cuts_cuts.weights.xml")
    mcuts = reader.FindCutsMVA("cuts")
    cutsMin = ROOT.std.vector("double")()
    cutsMax = ROOT.std.vector("double")()

    # Print a bunch of points
    for bin in range(1, hist_roc.GetNbinsX()+1):
        eff_s = hist_roc.GetXaxis().GetBinCenter(bin)
        eff_b = hist_roc.GetBinContent(bin)
        signif = s * eff_s / math.sqrt(s * eff_s + b * eff_b)
        print("\n\neff_s = {}, eff_b = {}, signif = {}".format(eff_s, eff_b, signif))
        cutsMin.clear()
        cutsMax.clear()
        mcuts.GetCuts(eff_s, cutsMin, cutsMax)
        for i in range(cutsMin.size()):
            print(f"{training_vars[i]}: {cutsMin[i]} <===> {cutsMax[i]}")

    print("\n\nBest eff_s = {}".format(best_eff_s))
    cutsMin.clear()
    cutsMax.clear()
    mcuts.GetCuts(best_eff_s, cutsMin, cutsMax)
    for i in range(cutsMin.size()):
        print(f"{training_vars[i]}: {cutsMin[i]} <===> {cutsMax[i]}")

# Find the maximum of a TGraph, first fitting with a polynomial to smooth bumps and then choosing the closest point
def FindMaxSmoothed(graph):
    #fit = ROOT.TF1("fit_poly3", "[0]*(1-x)*(1-x)*(1-x) + [1]*3*x*(1-x)*(1-x) + [2]*3*x*x*(1-x) + [3]*x**3", 0., 1.)
    fit = ROOT.TF1("fit_poly4", "[0]*(1-x)**4 + [1]*4*x*(1-x)**3 + [2]*6*x**2*(1-x)**2 + [3]*4*x**3*(1-x) + [4]*x**4", 0., 1.)
    fit.SetParameter(0, 0.2)
    fit.SetParameter(1, 0.2)
    fit.SetParameter(2, 0.2)
    fit.SetParameter(3, 0.2)
    fit.SetParameter(4, 0.2)
    graph.Fit(fit)

    xarr = graph.GetX()
    fitarr = []
    for i, x in enumerate(xarr):
        fitarr.append((x, fit.Eval(x)))
    return {"fit": fit, "max": max(fitarr, key=lambda x: x[1])}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run cut optimization")
    parser.add_argument("--step1", action="store_true", help="Step 1: extract s and b from data")
    parser.add_argument("--step2", action="store_true", help="Step 2: run TMVA MethodCuts")
    parser.add_argument("--step3", action="store_true", help="Step 3: make significant plot")
    parser.add_argument("--kinplots", action="store_true", help="Run KinematicPlots")
    parser.add_argument("--all", action="store_true", help="Run the whole thing")
    parser.add_argument("--side", type=str, choices=["tag", "probe"], help="tag or probe")
    args = parser.parse_args()

    # Load events
    chain_data = ROOT.TChain("Bcands_Bs_opt")
    for f in glob("/home/dryu/BFrag/data/histograms/Run*root"):
        chain_data.Add(f)
    chain_mc = ROOT.TChain("Bcands_Bs_opt_Bs2PhiJpsi2KKMuMu_probefilter")
    chain_mc.Add("/home/dryu/BFrag/boffea/barista/fitting/optimization_Bs_mc.root")
    #chain_mc.Print()
    print("MC nentries = {}".format(chain_mc.GetEntries()))
    print("Data nentries = {}".format(chain_data.GetEntries()))

    if args.step1 or args.all:
        GetSBFractions(chain_data, side=args.side)
    if args.step2 or args.all:
        RunMethodCuts(chain_data, chain_mc, side=args.side)
    if args.step3 or args.all:
        ComputeSignificance()
    if args.kinplots or args.all:
        KinematicPlots(chain_data, chain_mc, side=args.side)
