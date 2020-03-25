import os
import sys
import glob
from pprint import pprint

files_per_job = 25

runps = [
	"ParkingBPH5/crab_data_Run2018D_part5",
	"ParkingBPH6/crab_data_Run2018B_part6",
	"ParkingBPH2/crab_data_Run2018A_part2",
	"ParkingBPH5/crab_data_Run2018A_part5",
	"ParkingBPH1/crab_data_Run2018D_part1",
	"ParkingBPH3/crab_data_Run2018A_part3",
	"ParkingBPH5/crab_data_Run2018B_part5",
	"ParkingBPH1/crab_data_Run2018C_part1",
	"ParkingBPH2/crab_data_Run2018D_part2",
	"ParkingBPH3/crab_data_Run2018B_part3",
	"ParkingBPH2/crab_data_Run2018C_part2",
	"ParkingBPH5/crab_data_Run2018C_part5",
	"ParkingBPH4/crab_data_Run2018D_part4",
	"ParkingBPH4/crab_data_Run2018B_part4",
	"ParkingBPH1/crab_data_Run2018B_part1",
	"ParkingBPH1/crab_data_Run2018A_part1",
	"ParkingBPH3/crab_data_Run2018D_part3",
	"ParkingBPH6/crab_data_Run2018A_part6",
	"ParkingBPH4/crab_data_Run2018A_part4",
	"ParkingBPH2/crab_data_Run2018B_part2",
	"ParkingBPH3/crab_data_Run2018C_part3",
	"ParkingBPH4/crab_data_Run2018C_part4"
]
index_files = {}
all_index_files = {}
for runp in runps:
	runp_short = runp.split("/")[1].replace("crab_", "")
	files = glob.glob("/home/dryu/store/BParkingNANO/v1_7/{}/*/*/*root".format(runp))
	print("{} => {} files".format(runp, len(files)))
	job_files = []
	index_files[runp] = []
	subjob_idx = 0
	for i, file in enumerate(files):
		job_files.append(file)
		if len(job_files) >= files_per_job or (i == len(files) - 1):
			this_idx_file = "/home/dryu/BFrag/boffea/barista/filelists/v1_7/partialdatasplit/{}_subjob{}.txt".format(runp_short, subjob_idx)
			index_files[runp].append(this_idx_file)
			all_index_files[os.path.basename(this_idx_file).replace(".txt", "").replace("data_", "")] = this_idx_file
			with open(this_idx_file, "w") as f:
				for file2 in job_files:
					f.write("{}\n".format(file2))
			job_files = []
			subjob_idx += 1
pprint(all_index_files)
with open("data_index.py", "w") as f:
	pprint(all_index_files, f)
