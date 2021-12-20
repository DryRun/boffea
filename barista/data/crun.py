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
tarball_dir = "/home/dyu7/BFrag/tarballs"
print("Tarring user code...")
os.system(f"tar -czf {tarball_dir}/usercode.tar.gz -C $HOME/BFrag boffea/barista boffea/brazil --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf' --exclude='*.tar.gz'")
print("...done.")
if args.retar_venv:
	print("Tarring virtualenv...")
	os.system(f"tar -czf {tarball_dir}/venv.tar.gz -C $HOME/BFrag boffea/venv boffea/setup.py boffea/env.sh --exclude='*.root' --exclude='*.coffea' --exclude='*.png' --exclude='*.pdf'")
	print("...done.")

# Loop over datasets
global_resubmit_script = open(f"{working_directory}/resubmit_all.sh", "w")
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
		run_script.write("RETRY_COUNTER=0\n")
		run_script.write("while [[ \"$RETRY_COUNTER\" -lt 5 && ! \"$(find . -maxdepth 1 -name '*coffea' -print -quit)\" ]]; do\n")
		run_script.write( "    echo \"Retry $RETRY_COUNTER\"\n")
		run_script.write(f"    python barista/data/data_processor.py --dataset ${{DATASET_SUBJOBS[$1]}} -s ${{DATASET_SUBJOBS[$1]}} --condor -w 8 \n")
		run_script.write( "    RETRY_COUNTER=$((RETRY_COUNTER+1))\n")
		run_script.write("done\n")
		run_script.write("if [[ \"$RETRY_COUNTER\" -eq 5 && ! \"$(find . -maxdepth 1 -name '*coffea' -print -quit)\" ]]; then\n")
		run_script.write("    echo \"FATAL: Hit max retries\"\n")
		run_script.write("fi\n")
		run_script.write("mv *coffea ${_CONDOR_SCRATCH_DIR}\n")
		run_script.write("mkdir -pv hide\n")
		run_script.write("mv data_Run*root hide\n")
		run_script.write("echo 'Done with run script'\n")

	with open(f"{working_subdir}/localrun.sh", "w") as localrun_script:
		localrun_script.write("#!/bin/bash\n")
		for subjob in subjobs:
			localrun_script.write(f"python /home/dyu7/BFrag/boffea/barista/data/data_processor.py --dataset {subjob} -s {subjob} -w 8\n")

	files_to_transfer = [f"{tarball_dir}/usercode.tar.gz", f"{tarball_dir}/venv.tar.gz"]

	# Make condor command
	csub_command = f"csub {working_subdir}/run.sh -F {','.join(files_to_transfer)} -n {len(subjobs)} -d {working_subdir}"
	print(csub_command)
	with open(f"{working_subdir}/csub_command.sh", "w") as csub_script:
		csub_script.write("#!/bin/bash\n")
		csub_script.write(csub_command + "\n")

	# Make resubmit command
	# - Lazy: rerun whole job if not all coffea outputs are present
	with open(f"{working_subdir}/resubmit.sh", "w") as resubmit_script:
		resubmit_script.write("""
#!/bin/bash
NSUCCESS=$(ls -l *coffea | wc -l)
NTOTAL={}
if [ \"$NSUCCESS\" -ne \"$NTOTAL\" ]; then
    source csub_command.sh
fi
""".format(len(subjobs)))
	global_resubmit_script.write(f"cd {working_subdir}\n")
	global_resubmit_script.write(f"source resubmit.sh\n")

	os.system(csub_command)



