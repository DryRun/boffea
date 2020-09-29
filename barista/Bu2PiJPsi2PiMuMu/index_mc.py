import os
import sys
import glob

filelist =glob.glob(f"/home/dryu/store/BParkingNANO/v2_7_0/Bu2PiJPsi2PiMuMu/*.root")

with open(f"/home/dryu/BFrag/boffea/barista/filelists/v2_7/files_Bu2PiJPsi2PiMuMu.txt", "w") as f:
	for file in filelist:
		f.write(file + "\n")			