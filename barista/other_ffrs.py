'''
Create FFR objects from other analyses
- Shares simple get methods with FFRData
- Otherwise, the only computation performed is R_AB = yieldA/yieldB
yields_dict = {"Bu": [(yield1, dyield1), (yield2, dyield2), ...]}
'''
import os
import sys
import copy
import numpy as np
import math
import pickle

from brazil.aguapreta import BR_phiKK, BR_KstarKpi, BR_BuKJpsi, BR_BdKstarJpsi, BR_BsPhiJpsi
#BR_phiKK = 0.492
#BR_KstarKpi = 0.665
#BR_BuKJpsi = 1.020e-3
#BR_BdKstarJpsi = 1.27e-3
#BR_BsPhiJpsi = 1.08e-3

def main():
	ffrs = {}

	### BPH-21-001, pt
	xbins = [12, 13, 14, 15, 16, 18, 20, 23, 26, 29, 34, 45, 70]
	yields_dict = {}
	yields_dict["Bd"] = [(13261, 520), (25320, 857), (32496, 923), (41243, 695), (95348, 1038), (101216, 1321), (141747, 2040), (115830, 1565), (91951, 1028), (104570, 1130), (109976, 896), (61313, 1153)]
	yields_dict["Bu"] = [(208436, 529), (238697, 554), (251434, 574), (252905, 570), (472530, 769), (407594, 706), (482561, 768), (349881, 661), (250986, 554), (268446, 581), (267778, 609), (138089, 433)]
	yields_dict["Bs"] = [(4395, 75), (6997, 100), (8901, 108), (10273, 118), (22164, 171), (21567, 170), (28385, 191), (22152, 167), (16304, 145), (18314, 154), (19048, 156), (9869, 116)]
	eff_dict = {}
	eff_dict["Bu"] = [(0.01586, 0.00012), (0.02360, 0.00016), (0.03203, 0.00022), (0.04095, 0.00028), (0.05377, 0.00027), (0.07143, 0.00039), (0.09110, 0.00045), (0.11465, 0.00065), (0.13306, 0.00089), (0.15502, 0.00097), (0.18159, 0.00111), (0.21194, 0.00169)]
	eff_dict["Bd"] = [(0.00121, 0.00002), (0.00276, 0.00003), (0.00495, 0.00005), (0.00788, 0.00007), (0.01279, 0.00008), (0.02043, 0.00013), (0.03075, 0.00017), (0.04369, 0.00026), (0.05597, 0.00038), (0.06934, 0.00044), (0.08990, 0.00053), (0.11420, 0.00085)]
	eff_dict["Bs"] = [(0.00263, 0.00003), (0.00562, 0.00006), (0.00974, 0.00009), (0.01450, 0.00013), (0.02240, 0.00014), (0.03425, 0.00022), (0.04824, 0.00027), (0.06513, 0.00041), (0.08085, 0.00058), (0.09673, 0.00065), (0.11781, 0.00077), (0.13636, 0.00119)]
	ffrs[("BPH21001", "pt")] = FFRDataFromYields(
					yields_dict=yields_dict, 
					eff_dict=eff_dict,
					axis="pt", 
					name="BPH21001", 
					xbins=xbins)
	ffrs[("BPH21001", "pt")].save()

	### BPH-21-001, rapidity
	xbins = [0.00, 0.25, 0.50, 0.75, 1.00, 1.30, 1.60, 2.40]
	yields_dict = {}
	yields_dict["Bs"] = [(29578, 186), (30854, 192), (32434, 198), (29034, 196), (26040, 192), (17953, 167), (22590, 222)]
	yields_dict["Bu"] = [(541692, 764 ), (559411, 784), (593143, 821), (535407, 810), (520667, 919), (393101, 910), (479677, 1168)]
	yields_dict["Bd"] = [(148284, 1261), (155504, 1450), (163290, 1322), (142595, 1246), (125410, 1068), (86048, 823), (107838, 584)]

	eff_dict = {}
	eff_dict["Bu"] = [(0.06114, 0.00032), (0.06319, 0.00033), (0.06875, 0.00035), (0.06720, 0.00035), (0.06240, 0.00031), (0.05470, 0.00029), (0.03063, 0.00013)]
	eff_dict["Bs"] = [(0.03007, 0.00018), (0.03157, 0.00019), (0.03418, 0.00020), (0.03237, 0.00020), (0.02793, 0.00016), (0.02336, 0.00015), (0.01312, 0.00007)]
	eff_dict["Bd"] = [(0.01982, 0.00012), (0.02051, 0.00012), (0.02224, 0.00013), (0.02083, 0.00012), (0.01773, 0.00010), (0.01444, 0.00009), (0.00838, 0.00004)]
	ffrs[("BPH21001", "y")] = FFRDataFromYields(
					yields_dict=yields_dict, 
					eff_dict=eff_dict, 
					axis="y", 
					name="BPH21001", 
					xbins=xbins)
	ffrs[("BPH21001", "y")].save()


	''' OLD: Sometime before approval
	### BPH-21-001, pt
	xbins = [12, 13, 14, 15, 16, 18, 20, 23, 26, 29, 34, 45, 70]
	yields_Bs = [(7206, 95), (9668, 114), (11480, 124), (12482, 129), (25734, 186), (24024, 178), (30795, 200), (23563, 175), (17140, 151), (18951, 160), (19480, 166), (9878, 117)]
	yields_Bu = [(208436, 529),(238697, 554),(251434, 574),(252905, 570),(472530, 769),(407594, 706),(482561, 768),(349881, 661),(250986, 554),(268446, 581),(267778, 609),(138089, 433)]
	yields_Bd = [(29719, 184 ), (43163, 223 ), (53079, 245 ), (59509, 259 ), (127180, 380), (123760, 377), (161770, 426), (126530, 377), (96648, 329 ), (108612, 351), (113422, 362), (62188, 271 ), ]
	eff_Bu = [(0.01586, 0.00012), (0.02360, 0.00016), (0.03203, 0.00022), (0.04095, 0.00028), (0.05377, 0.00027), (0.07143, 0.00039), (0.09110, 0.00045), (0.11465, 0.00065), (0.13306, 0.00089), (0.15502, 0.00097), (0.18159, 0.00111), (0.21194, 0.00169)]
	eff_Bs = [(0.00441, 0.00005), (0.00803, 0.00007), (0.01273, 0.00011), (0.01800, 0.00015), (0.02616, 0.00015), (0.03834, 0.00023), (0.05229, 0.00029), (0.06911, 0.00043), (0.08452, 0.00060), (0.10027, 0.00067), (0.12046, 0.00078), (0.13689, 0.00120)]
	eff_Bd = [(0.00306, 0.00003),(0.00563, 0.00005),(0.00891, 0.00007),(0.01285, 0.00010),(0.01917, 0.00011),(0.02839, 0.00017),(0.04030, 0.00021),(0.05475, 0.00031),(0.06756, 0.00044),(0.08171, 0.00050),(0.10209, 0.00058),(0.12372, 0.00090),]

	ffrs[("BPH21001", "pt")] = FFRDataFromYields(
					yields_dict={"Bu": yields_Bu, "Bd": yields_Bd, "Bs": yields_Bs}, 
					eff_dict={"Bu": eff_Bu, "Bd": eff_Bd, "Bs": eff_Bs}, 
					axis="pt", 
					name="BPH21001", 
					xbins=xbins)
	ffrs[("BPH21001", "pt")].save()


	### BPH-21-001, rapidity
	xbins = [0.00, 0.25, 0.50, 0.75, 1.00, 1.30, 1.60, 2.40]
	yields_Bs = [(33218, 196), (34649, 203), (36469, 209), (32600, 205), (29261, 200), (20280, 177), (25514, 224)]
	yields_Bu = [(541692, 764 ), (559411, 784 ), (593143, 821 ), (535407, 810 ), (520667, 919 ), (393101, 910 ), (479677, 1168), ]
	yields_Bd = [(177174, 433), (183225, 511), (194064, 460), (170113, 434), (151523, 425), (105941, 374), (130271, 444), ]
	eff_Bu = [(0.06114, 0.00032), (0.06319, 0.00033), (0.06875, 0.00035), (0.06720, 0.00035), (0.06240, 0.00031), (0.05470, 0.00029), (0.03063, 0.00013), ]
	eff_Bs = [(0.03375, 0.00020), (0.03546, 0.00021), (0.03838, 0.00022), (0.03622, 0.00021), (0.03138, 0.00018), (0.02628, 0.00016), (0.01462, 0.00007), ]
	eff_Bd = [(0.02638, 0.00015), (0.02743, 0.00015), (0.02974, 0.00016), (0.02788, 0.00016), (0.02383, 0.00013), (0.01953, 0.00011), (0.01105, 0.00005), ]

	ffrs[("BPH21001", "y")] = FFRDataFromYields(
					yields_dict={"Bu": yields_Bu, "Bd": yields_Bd, "Bs": yields_Bs}, 
					eff_dict={"Bu": eff_Bu, "Bd": eff_Bd, "Bs": eff_Bs}, 
					axis="y", 
					name="BPH21001", 
					xbins=xbins)
	ffrs[("BPH21001", "y")].save()
	'''

	ffrs[("LHCb_13TeV", "pt")] = FFRDataLHCb(
		name="LHCb_13TeV", 
		axis="pt",
		xbins=np.array([0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0, 40.0]), 
		Rsu=np.array([0.133, 0.132, 0.134, 0.132, 0.134, 0.129, 0.131, 0.129, 0.128, 0.128, 0.118, 0.121]), 
		dRsu=np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]), 
	)
	ffrs[("LHCb_13TeV", "pt")].save()

	ffrs[("LHCb_combined", "pt")] = FFRDataLHCb(
		name="LHCb_combined", 
		axis="pt",
		xbins=np.array([0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0, 40.0]), 
		Rsu=np.array([0.125, 0.127, 0.125, 0.128, 0.128, 0.127, 0.127, 0.126, 0.125, 0.125, 0.118, 0.120]),
		dRsu=np.array([0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002]),
	)
	ffrs[("LHCb_combined", "pt")].save()


	# LHCb
class FFRDataLHCb:
	def __init__(self, name, axis, xbins, Rsu, dRsu):
		self._xbins = xbins
		self._Rsu = Rsu
		self._dRsu = dRsu

		# Dummies
		self._axis  = axis
		self._name  = name
		self._yields = {}
		self._dyields = {}
		self._efficiencies = {}
		self._defficiencies = {}
		self._n = {}
		self._dn = {}
		for btype in ["Bu", "Bs", "Bd"]:
			self._yields[btype] = None
			self._dyields[btype] = None

			self._efficiencies[btype] = None
			self._defficiencies[btype] = None

			self._n[btype] = None
			self._dn[btype] = None
		self._total_n = None
		self._dtotal_n = None

		# Divisions
		self._Rsd = None
		self._dRsd = None

		#self._Rsu = 
		#self._dRsu = 

		self._Rdu = None
		self._dRdu = None

		self._fsfd = None
		self._dfsfd = None

		self._fsfu = self._Rsu * BR_phiKK / BR_KstarKpi * 2.0
		self._dfsfu = self._dRsu * BR_phiKK / BR_KstarKpi * 2.0

		self._fdfu = None
		self._dfdfu = None

		# Compute averages
		self._sum_n = {}
		self._dsum_n = {}
		for btype in ["Bu", "Bd", "Bs"]:
			self._sum_n[btype] = None
			self._dsum_n[btype] = None

		self._avg_Rsu = None
		self._avg_dRsu = None

		self._avg_Rsd = None
		self._avg_dRsd = None

		self._avg_Rdu = None
		self._avg_dRdu = None

		self._avg_fsfu = None
		self._avg_dfsfu = None

		self._avg_fsfd = None
		self._avg_dfsfd = None

		self._avg_fdfu = None
		self._avg_dfdfu = None

	def save(self):
		output_path = f"/home/dyu7/BFrag/data/ffrs/external/ffrs_lhcb_{self._axis}_{self._name}.pkl"
		with open(output_path, "wb") as f:
			pickle.dump(self, f)

	def axis(self):
		return self._axis

	def xbins(self):
		return self._xbins

	def Rsu(self):
		return self._Rsu

	def Rsd(self):
		return self._Rsd

	def Rdu(self):
		return self._Rdu

	def dRsu(self):
		return self._dRsu

	def dRsd(self):
		return self._dRsd

	def dRdu(self):
		return self._dRdu

	def fsfu(self):
		return self._fsfu

	def fsfd(self):
		return self._fsfd

	def fdfu(self):
		return self._fdfu

	def dfsfu(self):
		return self._dfsfu

	def dfsfd(self):
		return self._dfsfd

	def dfdfu(self):
		return self._dfdfu

	def N(self, btype):
		return self._n[btype]

	def dN(self, btype):
		return self._dn[btype]

	def TotalN(self, btype):
		return self._total_n[btype]

	def dTotalN(self, btype):
		return self._dtotal_n[btype]

	def Yield(self, btype):
		return self._yields[btype]

	def dYield(self, btype):
		return self._dyields[btype]

	def Eff(self, btype):
		return self._efficiencies[btype]

	def dEff(self, btype):
		return self._defficiencies[btype]

	def TotalEff(self, btype):
		return self._total_efficiencies[btype]

	def dTotalEff(self, btype):
		return self._dtotal_efficiencies[btype]

	# Averages
	def Rsu_avg(self):
		return self._avg_Rsu

	def Rsd_avg(self):
		return self._avg_Rsd

	def Rdu_avg(self):
		return self._avg_Rdu

	def dRsu_avg(self):
		return self._avg_dRsu

	def dRsd_avg(self):
		return self._avg_dRsd

	def dRdu_avg(self):
		return self._avg_dRdu

	def fsfu_avg(self):
		return self._avg_fsfu

	def fsfd_avg(self):
		return self._avg_fsfd

	def fdfu_avg(self):
		return self._avg_fdfu

	def dfsfu_avg(self):
		return self._avg_dfsfu

	def dfsfd_avg(self):
		return self._avg_dfsfd

	def dfdfu_avg(self):
		return self._avg_dfdfu

	def N_avg(self, btype):
		return self._sum_n[btype]

	def dN_avg(self, btype):
		return self._dsum_n[btype]

	# Dummy functions
	def side(self):
		return "noside"

	def binned(self):
		return "nobinned"

	def trigger_strategy(self):
		return "notrigger"

	def fitfunc(self):
		return "nofitfunc"

	def selection(self):
		return "noselection"


class FFRDataFromYields:
	def __init__(self, yields_dict, eff_dict, xbins, axis="pt", name="default"):
		self._axis  = copy.deepcopy(axis)
		self._xbins = copy.deepcopy(xbins)
		self._name  = copy.deepcopy(name)

		self._yields = {}
		self._dyields = {}
		self._efficiencies = {}
		self._defficiencies = {}
		self._n = {}
		self._dn = {}
		for btype in ["Bu", "Bs", "Bd"]:
			self._yields[btype] = np.array([x[0] for x in yields_dict[btype]])
			self._dyields[btype] = np.array([x[1] for x in yields_dict[btype]])

			self._efficiencies[btype] = np.array([x[0] for x in eff_dict[btype]])
			self._defficiencies[btype] = np.array([x[1] for x in eff_dict[btype]])

			self._n[btype] = self._yields[btype] / self._efficiencies[btype]
			self._dn[btype] = self._n[btype] * np.sqrt(
				(self._dyields[btype] / self._yields[btype])**2
				+ (self._defficiencies[btype] / self._efficiencies[btype])**2)

		self._total_n = self._n # No distinction between total and nontotal, unliked FFRData where probefilter effs are assumed to cancel but might still be interesting
		self._dtotal_n = self._dn

		# Divisions
		self._Rsu = self._total_n["Bs"] / self._total_n["Bu"] #/ BR_phiKK / 2.0
		self._dRsu = self._Rsu * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2) / 2.0

		self._Rsd = self._total_n["Bs"] / self._total_n["Bd"] #/ BR_phiKK * BR_KstarKpi / 2.0
		self._dRsd = self._Rsd * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bd"] / self._total_n["Bd"])**2) / 2.0

		self._Rdu = self._total_n["Bd"] / self._total_n["Bu"] #/ BR_KstarKpi / 2.0
		self._dRdu = self._Rdu * np.sqrt((self._dtotal_n["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2) / 2.0

		self._fsfu = self._total_n["Bs"] / self._total_n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
		self._dfsfu = self._fsfu * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)

		self._fsfd = self._total_n["Bs"] / self._total_n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
		self._dfsfd = self._fsfd * np.sqrt((self._dtotal_n["Bs"] / self._total_n["Bs"])**2 + (self._dtotal_n["Bd"] / self._total_n["Bd"])**2)

		self._fdfu = self._total_n["Bd"] / self._total_n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
		self._dfdfu = self._fdfu * np.sqrt((self._dtotal_n["Bd"] / self._total_n["Bd"])**2 + (self._dtotal_n["Bu"] / self._total_n["Bu"])**2)

		# Compute averages
		self._sum_n = {}
		self._dsum_n = {}
		for btype in ["Bu", "Bd", "Bs"]:
			self._sum_n[btype] = self._total_n[btype].sum()
			self._dsum_n[btype] = math.sqrt((self._dn[btype] * self._dn[btype]).sum())

		self._avg_Rsu = self._sum_n["Bs"] / self._sum_n["Bu"] #/ BR_phiKK / 2.0
		self._avg_dRsu = self._avg_Rsu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2) / 2.0

		self._avg_Rsd = self._sum_n["Bs"] / self._sum_n["Bd"] #/ BR_phiKK * BR_KstarKpi / 2.0
		self._avg_dRsd = self._avg_Rsd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2) / 2.0

		self._avg_Rdu = self._sum_n["Bd"] / self._sum_n["Bu"] #/ BR_KstarKpi / 2.0
		self._avg_dRdu = self._avg_Rdu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2) / 2.0

		self._avg_fsfu = self._sum_n["Bs"] / self._sum_n["Bu"] / (BR_BsPhiJpsi * BR_phiKK) * BR_BuKJpsi
		self._avg_dfsfu = self._avg_fsfu * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

		self._avg_fsfd = self._sum_n["Bs"] / self._sum_n["Bd"] / (BR_BsPhiJpsi * BR_phiKK) * (BR_BdKstarJpsi * BR_KstarKpi)
		self._avg_dfsfd = self._avg_fsfd * np.sqrt((self._dsum_n["Bs"] / self._sum_n["Bs"])**2 + (self._dsum_n["Bd"] / self._sum_n["Bd"])**2)

		self._avg_fdfu = self._sum_n["Bd"] / self._sum_n["Bu"] / (BR_BdKstarJpsi * BR_KstarKpi) * BR_BuKJpsi
		self._avg_dfdfu = self._avg_fdfu * np.sqrt((self._dsum_n["Bd"] / self._sum_n["Bd"])**2 + (self._dsum_n["Bu"] / self._sum_n["Bu"])**2)

	def save(self):
		output_path = f"/home/dyu7/BFrag/data/ffrs/external/ffrs_{self._axis}_{self._name}.pkl"
		with open(output_path, "wb") as f:
			pickle.dump(self, f)

	def axis(self):
		return self._axis

	def xbins(self):
		return self._xbins

	def Rsu(self):
		return self._Rsu

	def Rsd(self):
		return self._Rsd

	def Rdu(self):
		return self._Rdu

	def dRsu(self):
		return self._dRsu

	def dRsd(self):
		return self._dRsd

	def dRdu(self):
		return self._dRdu

	def dRsu_total(self):
		return self._dRsu

	def dRsd_total(self):
		return self._dRsd

	def dRdu_total(self):
		return self._dRdu

	def fsfu(self):
		return self._fsfu

	def fsfd(self):
		return self._fsfd

	def fdfu(self):
		return self._fdfu

	def dfsfu(self):
		return self._dfsfu

	def dfsfd(self):
		return self._dfsfd

	def dfdfu(self):
		return self._dfdfu

	def dfsfu_total(self):
		return self._dfsfu

	def dfsfd_total(self):
		return self._dfsfd

	def dfdfu_total(self):
		return self._dfdfu

	def N(self, btype):
		return self._n[btype]

	def dN(self, btype):
		return self._dn[btype]

	def TotalN(self, btype):
		return self._total_n[btype]

	def dTotalN(self, btype):
		return self._dtotal_n[btype]

	def Yield(self, btype):
		return self._yields[btype]

	def dYield(self, btype):
		return self._dyields[btype]

	def Eff(self, btype):
		return self._efficiencies[btype]

	def dEff(self, btype):
		return self._defficiencies[btype]

	def TotalEff(self, btype):
		return self._total_efficiencies[btype]

	def dTotalEff(self, btype):
		return self._dtotal_efficiencies[btype]

	# Averages
	def Rsu_avg(self):
		return self._avg_Rsu

	def Rsd_avg(self):
		return self._avg_Rsd

	def Rdu_avg(self):
		return self._avg_Rdu

	def dRsu_avg(self):
		return self._avg_dRsu

	def dRsd_avg(self):
		return self._avg_dRsd

	def dRdu_avg(self):
		return self._avg_dRdu

	def fsfu_avg(self):
		return self._avg_fsfu

	def fsfd_avg(self):
		return self._avg_fsfd

	def fdfu_avg(self):
		return self._avg_fdfu

	def dfsfu_avg(self):
		return self._avg_dfsfu

	def dfsfd_avg(self):
		return self._avg_dfsfd

	def dfdfu_avg(self):
		return self._avg_dfdfu

	def N_avg(self, btype):
		return self._sum_n[btype]

	def dN_avg(self, btype):
		return self._dsum_n[btype]

	# Dummy functions
	def side(self):
		return "noside"

	def binned(self):
		return "nobinned"

	def trigger_strategy(self):
		return "notrigger"

	def fitfunc(self):
		return "nofitfunc"

	def selection(self):
		return "noselection"


if __name__ == "__main__":
	main()