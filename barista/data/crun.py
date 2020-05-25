import datetime
import os

'''
  parser.add_argument("--datasets", "-d", type=str, help="List of datasets to run (comma-separated")
  parser.add_argument("--workers", "-w", type=int, default=16, help="Number of workers")
  parser.add_argument("--quicktest", "-q", action="store_true", help="Run a small test job")
  parser.add_argument("--save_tag", "-s", type=str, help="Save tag for output file")
'''
import argparse
parser = argparse.ArgumentParser(description="Make histograms for B FFR data")
parser.add_argument("-r", "--runps", type=str, help="List of run-periods i.e. data_Run2018C_part3. Comma-separated.")
parser.add_argument("--retar_venv", action="store_true", help="Retar venv (takes a while)")
args = parser.parse_args()

from data_index import in_txt
if args.runps:
	runps = args.runps.split(",")
else:
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

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
working_directory = os.path.expandvars(f"$HOME/BFrag/data/histograms/condor/job{ts}")
os.system(f"mkdir -pv {working_directory}")
cwd = os.getcwd()
os.chdir(working_directory)

# Tar working area
tarball_dir = "/home/dryu/BFrag/tarballs"
print("Tarring user code...")
os.system(f"tar -czf {tarball_dir}/usercode.tar.gz -C $HOME/BFrag boffea/barista boffea/brazil --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf'")
print("...done.")
if args.retar_venv:
	print("Tarring virtualenv...")
	os.system(f"tar -czf {tarball_dir}/venv.tar.gz -C $HOME/BFrag boffea/venv boffea/setup.py boffea/env.sh --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf'")
	print("...done.")

# Loop over datasets
for runp in runps:
	working_subdir = f"{working_directory}/{runp}/"
	os.system(f"mkdir -pv {working_subdir}")

	subjobs = [x for x in in_txt.keys() if runp in x]

	# Make run script (to be run on batch node)
	with open(f"{working_subdir}/run.sh", "w") as run_script:
		run_script.write("#!/bin/bash\n")
		run_script.write("lscpu\n")
		run_script.write("tar -xzf usercode.tar.gz\n")
		run_script.write("tar -xzf venv.tar.gz\n")
		run_script.write("ls -lrth\n")
		run_script.write("cd boffea\n")
		run_script.write("ls -lrth\n")
		#run_script.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh\n")
		#run_script.write("source venv/bin/activate\n")
		run_script.write("source env.sh\n")
		run_script.write("DATASET_SUBJOBS=({})\n".format(" ".join(subjobs)))
		run_script.write(f"python barista/data/data_processor.py --dataset ${{DATASET_SUBJOBS[$1]}} -s ${{DATASET_SUBJOBS[$1]}}\n")
		run_script.write("mv *coffea ${_CONDOR_SCRATCH_DIR}\n")

	files_to_transfer = [f"{tarball_dir}/usercode.tar.gz", f"{tarball_dir}/venv.tar.gz"]

	# Make condor command
	csub_command = f"csub {working_subdir}/run.sh -F {','.join(files_to_transfer)} -n {len(subjobs)} -d {working_subdir}"
	print(csub_command)
	os.system(csub_command)



