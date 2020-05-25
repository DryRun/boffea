import os
import sys
import glob
from pprint import pprint

files_per_job = 5

runps = [
	"Run2018D_part5",
	"Run2018B_part6",
	"Run2018A_part2",
	"Run2018A_part5",
	"Run2018D_part1",
	"Run2018A_part3",
	"Run2018B_part5",
	"Run2018C_part1",
	"Run2018D_part2",
	"Run2018B_part3",
	"Run2018C_part2",
	"Run2018C_part5",
	"Run2018D_part4",
	"Run2018B_part4",
	"Run2018B_part1",
	"Run2018A_part1",
	"Run2018D_part3",
	"Run2018A_part6",
	"Run2018A_part4",
	"Run2018B_part2",
	"Run2018C_part3",
	"Run2018C_part4"
]
runps_patch = [
	"Run2018B_part1",
	"Run2018D_part1",
	"Run2018D_part5",
	"Run2018D_part2",
]
index_files = {}
all_index_files = {}
for runp in runps:
	#runp_short = runp.split("/")[1].replace("crab_", "")
	if runp in runps_patch:
		version = "v1_7_1"
	else:
		version = "v1_7"
	# Raw files, before hadding. Inefficient for coffea, too few events
	#files = glob.glob("/home/dryu/store/BParkingNANO/{}/{}/*/*/*root".format(version, runp))
	
	# Hadded files
	print("glob pattern:")
	print(f"/home/dryu/store/BParkingNANO/{version}/hadd/{runp}/*root")
	files = glob.glob(f"/home/dryu/store/BParkingNANO/{version}/hadd/{runp}/*root")
	print("{} => {} files".format(runp, len(files)))
	job_files = []
	index_files[runp] = []
	subjob_idx = 0
	for i, file in enumerate(files):
		job_files.append(file)
		if len(job_files) >= files_per_job or (i == len(files) - 1):
			this_idx_file = "/home/dryu/BFrag/boffea/barista/filelists/v1_7/haddsplit/{}_subjob{}.txt".format(runp, subjob_idx)
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
