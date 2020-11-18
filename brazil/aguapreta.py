import numpy as np
import math

# Masses in GeV
ELECTRON_MASS = 0.00051099895
MU_MASS = 0.1056583745
MUON_MASS = MU_MASS
TAU_MASS =  1.77686
PI_MASS = 0.13957018 # charged
PI0_MASS = 0.1349766
K_MASS = 0.493677
KS_MASS = 0.497611
KL_MASS = 0.497611
KSTAR_892_MASS = 0.89166
JPSI_1S_MASS = 3.0969
UPSILON_1S_MASS = 9.4603
PHI_1020_MASS = 1.019461

# B mesons
BU_MASS = 5.27931
BD_MASS = 5.27962
BS_MASS = 5.3663

# Cuts (see AN2017_168)
JPSI_WINDOW = 0.150
PHI_WINDOW = 0.010
KSTAR_WINDOW = 0.10
B0_PHI_VETO_WINDOW = 0.01
BS_KSTAR_VETO_WINDOW = 0.05

# Optimized cuts
final_cuts = {
	"Bs": {
		"sv_pt": 3.0, 
		"sv_prob": 0.07, 
		"cos2D": 0.997, 
		"l_xy_sig": 4.0, 
		"dm_phi": 0.01, 
		"k1_pt": 0.85, 
		"k2_pt": 0.6, 
		"l1_pt": 1.5, # Not optimized 
		"l2_pt": 1.5, # Not optimized
		"kstar_veto": BS_KSTAR_VETO_WINDOW,
	}, 
	"Bu": {
		"sv_pt": 3.0, 
		"sv_prob": 0.07, 
		"cos2D": 0.997, 
		"l_xy_sig": 4.0, 
		"k_pt": 0.85,
		"l1_pt": 1.5, # Not optimized 
		"l2_pt": 1.5, # Not optimized
	}, 
	"Bd": {
		"sv_pt": 3.0, 
		"sv_prob": 0.07, 
		"cos2D": 0.997, 
		"l_xy_sig": 4.0, 
		"k1_pt": 0.85, 
		"k2_pt": 0.6, 
		"l1_pt": 1.5, # Not optimized 
		"l2_pt": 1.5, # Not optimized
		"kstar_window": KSTAR_WINDOW,
		"phi_veto": B0_PHI_VETO_WINDOW,
	}, 
}

def where(predicate, iftrue, iffalse):
    predicate = predicate.astype(np.bool)   # just to make sure they're 0/1
    return predicate*iftrue + (1 - predicate)*iffalse

def delta_phi(phi1, phi2):
    return (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi

def delta_r(eta1, eta2, phi1, phi2):
    return ((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)**0.5
