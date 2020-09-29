# Fit bins
ptbins_coarse = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
ptbins_fine = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]
ybins = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.0, 2.25]

fit_cuts = []
cut_strings = {}
fit_text = {} # Text for plots
cut_strings["inclusive"] = "1"
fit_text["inclusive"] = "Inclusive"
for ipt in range(len(ptbins_coarse) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1]).replace(".", "p")
	fit_cuts.append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<pT<{}".format(ptbins_coarse[ipt], ptbins_coarse[ipt+1])

for iy in range(len(ybins) - 1):
	cut_str = "(pt > 5.0) && (pt < 30.0) && ({} < abs(y)) && (abs(y) < {})".format(ybins[iy], ybins[iy+1])
	cut_name = "ybin_{}_{}".format(ybins[iy], ybins[iy+1]).replace(".", "p")
	fit_cuts.append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<|y|<{}".format(ybins[iy], ybins[iy+1])

fit_cuts_fine = []
for ipt in range(len(ptbins_fine) - 1):
	cut_str = "(abs(y) < 2.25) && (pt > {}) && (pt < {})".format(ptbins_fine[ipt], ptbins_fine[ipt+1])
	cut_name = "ptbin_{}_{}".format(ptbins_fine[ipt], ptbins_fine[ipt+1]).replace(".", "p")
	fit_cuts_fine.append(cut_name)
	cut_strings[cut_name] = cut_str
	fit_text[cut_name] = "{}<pT<{}".format(ptbins_fine[ipt], ptbins_fine[ipt+1])

# Binning for binned fits
roobinning = {
	"Bu": 
}