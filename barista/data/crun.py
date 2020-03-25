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
parser.add_argument("--retar_venv", action="store_true", help="Retar venv (takes a while)")
args = parser.parse_args()

from data_index import in_txt
datasets = in_txt.keys()

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
working_directory = os.path.expandvars(f"$HOME/BFrag/data/histograms/condor/job{ts}")
os.system(f"mkdir -pv {working_directory}")
cwd = os.getcwd()
os.chdir(working_directory)

# Tar working area
tarball_dir = "/home/dryu/BFrag/tarballs"
os.system(f"tar -czf {tarball_dir}/usercode.tar.gz -C $HOME/BFrag boffea/barista boffea/brazil --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf'")
if args.retar_venv:
	os.system(f"tar -czf {tarball_dir}/venv.tar.gz -C $HOME/BFrag boffea/venv boffea/setup.py --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf'")

# Make run script (to be run on batch node)
with open(f"{working_directory}/run.sh", "w") as run_script:
	run_script.write("#!/bin/bash\n")
	run_script.write("tar -xzf usercode.tar.gz\n")
	run_script.write("tar -xzf venv.tar.gz\n")
	run_script.write("ls -lrth\n")
	run_script.write("cd boffea\n")
	run_script.write("ls -lrth\n")
	run_script.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh\n")
	run_script.write("source venv/bin/activate\n")
	run_script.write("DATASETS=({})\n".format(" ".join(datasets)))
	run_script.write(f"python barista/data/data_processor.py --dataset ${{DATASETS[$1]}} -s ${{DATASETS[$1]}}\n")
	run_script.write("mv *coffea ${_CONDOR_SCRATCH_DIR}\n")

files_to_transfer = [f"{tarball_dir}/usercode.tar.gz", f"{tarball_dir}/venv.tar.gz"]

# Make condor command
csub_command = f"csub {working_directory}/run.sh -F {','.join(files_to_transfer)} -n {len(datasets)} -d {working_directory}"
print(csub_command)
os.system(csub_command)



