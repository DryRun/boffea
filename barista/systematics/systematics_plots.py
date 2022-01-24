import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)
plt.tight_layout()

btypes = ["Bu", "Bd", "Bs"]
legend_entries = {
    "Bu": r"$B^{\pm}\rightarrow K^{\pm} J/\psi(\mu^{+}\mu^{-})$",
    "Bd": r"$B^{0}\rightarrow K^{*}J/\psi\rightarrow K^{\pm}\pi^{\mp}\mu^{+}\mu^{-}$",
    "Bs": r"$B^{0}_{s}\rightarrow \phi J/\psi \rightarrow K^{+}K^{-}\mu^{+}\mu^{-}$",
}
rlegend_entries = {
    "Bu": "B^{#pm}#rightarrow K^{#pm} J/#psi(#mu^{+}#mu^{-})",
    "Bd": "B^{0}#rightarrow K^{*}J/#psi#rightarrow K^{#pm}#pi^{#mp}#mu^{+}#mu^{-}",
    "Bs": "B^{0}_{s}#rightarrow #phi J/#psi #rightarrow K^{+}K^{-}#mu^{+}#mu^{-}",
}
style = {
    "Bu":{
        "marker_style": "o", 
        "marker_size": ".", 
        "marker_color": "blue", #seaborn_colors.get_root_color("Blues_d", 2),
        "line_color": "blue", #seaborn_colors.get_root_color("Blues_d", 3),
        "line_width": 1,
    },
    "Bd":{
        "marker_style": "s", 
        "marker_size": "s", 
        "marker_color": "green", #seaborn_colors.get_root_color("Greens_d", 2),
        "line_color": "green", #seaborn_colors.get_root_color("Greens_d", 3),
        "line_width": 1,
    },
    "Bs":{
        "marker_style": "D", 
        "marker_size": "D", 
        "marker_color": "orange", #seaborn_colors.get_root_color("Oranges_d", 2),
        "line_color": "orange", #seaborn_colors.get_root_color("Oranges_d", 3),
        "line_width": 1,
    }
}
axis_labels = {
    "pt": r"$p_{\mathrm{T}}$ [GeV]", 
    "y": "|y|",
    "absy": "|y|",
}

def make_twopanels(figsize=(7, 8)):
    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1], "hspace":0.1}, constrained_layout=True)
    fig.subplots_adjust(wspace=0)
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='y')
    ax[1].tick_params(axis='x')
    ax[1].tick_params(axis='y')
    fig.subplots_adjust(wspace=0)   
    return fig, ax

