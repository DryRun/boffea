import os
import sys
import glob

samples = ["BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen", 
		   "BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen"]

for sample in samples:
	filelist =glob.glob(f"/home/dryu/store/BParkingNANO/v2_7_0/{sample}/*/*/*/*.root")

	with open(f"/home/dryu/BFrag/boffea/barista/filelists/v2_7/files_{sample}.txt", "w") as f:
		for file in filelist:
			f.write(file + "\n")