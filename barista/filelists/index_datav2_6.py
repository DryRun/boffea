import os
import sys
import glob
from pprint import pprint

files_per_job = 7

runps = [
	"data_Run2018A_part1",
	"data_Run2018A_part2",
	"data_Run2018A_part3",
	"data_Run2018A_part4",
	"data_Run2018A_part5",
	"data_Run2018A_part6",
	"data_Run2018B_part1",
	"data_Run2018B_part2",
	"data_Run2018B_part3",
	"data_Run2018B_part4",
	"data_Run2018B_part5",
	"data_Run2018B_part6",
	"data_Run2018C_part1",
	"data_Run2018C_part2",
	"data_Run2018C_part3",
	"data_Run2018C_part4",
	"data_Run2018C_part5",
	"data_Run2018D_part1",
	"data_Run2018D_part2",
	"data_Run2018D_part4",
	"data_Run2018D_part3",
	"data_Run2018D_part5",
]
index_files = {}
all_index_files = {}
for runp in runps:	
	# Hadded files
	print("glob pattern:")
	print(f"/mnt/hadoop/store/user/dryu/BParkingNANO/v2_6/hadd/{runp}*root")
	files = glob.glob(f"/mnt/hadoop/store/user/dryu/BParkingNANO/v2_6/hadd/{runp}*root")
	print("{} => {} files".format(runp, len(files)))
	job_files = []
	index_files[runp] = []
	subjob_idx = 0
	for i, file in enumerate(files):
		job_files.append(file)
		if len(job_files) >= files_per_job or (i == len(files) - 1):
			this_idx_file = "/home/dryu/BFrag/boffea/barista/filelists/v2_6/haddsplit/{}_subjob{}.txt".format(runp, subjob_idx)
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
