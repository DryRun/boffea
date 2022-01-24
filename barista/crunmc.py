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
parser.add_argument("-d", "--datasets", type=str, help="List of datasets, see in_txt. Comma-separated.")
parser.add_argument("-n", "--nsubjobs", type=int, default=25, help="Number of subjobs for each dataset")
parser.add_argument("--retar_venv", action="store_true", help="Retar venv (takes a while)")
args = parser.parse_args()

in_txt = {
    "Bd2KsJpsi2KPiMuMu_probefilter":"/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BdToKstarJpsi_ToKPiMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KsJpsi2KPiMuMu_inclusive":"/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BdToJpsiKstar_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
    "Bd2KsJpsi2KPiMuMu_mufilter":"/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BdToKstarJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",	

	"Bs2PhiJpsi2KKMuMu_probefilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToPhiJpsi_ToKKMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
	"Bs2PhiJpsi2KKMuMu_inclusive": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToJpsiPhi_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),
	"Bs2PhiJpsi2KKMuMu_mufilter": os.path.expandvars("$HOME/BFrag/boffea/barista/filelists/frozen/files_BsToPhiJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt"),

	"Bu2KJpsi2KMuMu_probefilter": "/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BuToKJpsi_ToMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
	"Bu2KJpsi2KMuMu_inclusive": "/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BuToJpsiK_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
	"Bu2KJpsi2KMuMu_mufilter": "/home/dyu7/BFrag/boffea/barista/filelists/frozen/files_BuToKJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen.txt",
}
dataset_files = {}
for dataset_name, filelistpath in in_txt.items():
	with open(filelistpath, 'r') as filelist:
		dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

if args.datasets:
	datasets = args.datasets.split(",")
else:
	datasets = [
		"Bd2KsJpsi2KPiMuMu_probefilter",
		"Bd2KsJpsi2KPiMuMu_inclusive",
		"Bd2KsJpsi2KPiMuMu_mufilter",
		"Bs2PhiJpsi2KKMuMu_probefilter",
		"Bs2PhiJpsi2KKMuMu_inclusive",
		"Bs2PhiJpsi2KKMuMu_mufilter",
		"Bu2KJpsi2KMuMu_probefilter",
		"Bu2KJpsi2KMuMu_inclusive",
		"Bu2KJpsi2KMuMu_mufilter",
	]


ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
working_directory = os.path.expandvars(f"$HOME/BFrag/data/histograms/condor/mc_job{ts}")
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
for dataset in datasets:
	working_subdir = f"{working_directory}/{dataset}/"
	os.system(f"mkdir -pv {working_subdir}")

	# Make sure nsubjobs <= nfiles
	this_nsubjobs = min(args.nsubjobs, len(dataset_files[dataset]))

	# Choose executable
	if "Bd" in dataset: 
		executable = "barista/Bd2KsJpsi2KPiMuMu/mc_efficiency_processor.py"
	elif "Bu" in dataset: 
		executable = "barista/Bu2KJpsi2KMuMu/mc_efficiency_processor.py"
	elif "Bs" in dataset: 
		executable = "barista/Bs2PhiJpsi2KKMuMu/mc_efficiency_processor.py"


	# Make run script (to be run on batch node)
	with open(f"{working_subdir}/run.sh", "w") as run_script:
		run_script.write("#!/bin/bash\n")
		run_script.write("lscpu\n")
		run_script.write("cat /etc/os-release\n\n")
		run_script.write("lsb_release -a\n\n")
		run_script.write("hostnamectl\n\n")
		run_script.write("uname -r\n\n")
		#run_script.write("export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH\n")
		run_script.write("tar -xzf usercode.tar.gz\n")
		run_script.write("tar -xzf venv.tar.gz\n")
		run_script.write("ls -lrth\n")
		run_script.write("cd boffea\n")
		run_script.write("ls -lrth\n")
		#run_script.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh\n")
		#run_script.write("source venv/bin/activate\n")
		run_script.write("source env.sh\n")
		run_script.write("echo $LD_LIBRARY_PATH\n")
		#run_script.write("DATASET_SUBJOBS=({})\n".format(" ".join(subjobs)))
		run_script.write("RETRY_COUNTER=0\n")
		run_script.write("while [[ \"$RETRY_COUNTER\" -lt 5 && ! \"$(find . -maxdepth 1 -name '*coffea' -print -quit)\" ]]; do\n")
		run_script.write( "    echo \"Retry $RETRY_COUNTER\"\n")
		run_script.write(f"    python {executable} --dataset {dataset} --subjob $1 {this_nsubjobs} --condor -w 8 \n")
		run_script.write( "    RETRY_COUNTER=$((RETRY_COUNTER+1))\n")
		run_script.write("done\n")
		run_script.write("if [[ \"$RETRY_COUNTER\" -eq 5 && ! \"$(find . -maxdepth 1 -name '*coffea' -print -quit)\" ]]; then\n")
		run_script.write("    echo \"FATAL: Hit max retries\"\n")
		run_script.write("fi\n")
		run_script.write("mv *coffea ${_CONDOR_SCRATCH_DIR}\n")
		run_script.write("mkdir -pv hide\n")
		run_script.write("mv *root hide\n")
		run_script.write("mv $_CONDOR_SCRATCH_DIR/BPark*root hide\n")
		run_script.write("echo 'Done with run script'\n")

	with open(f"{working_subdir}/localrun.sh", "w") as localrun_script:
		localrun_script.write("#!/bin/bash\n")
		for isubjob in range(this_nsubjobs):
			localrun_script.write(f"python /home/dyu7/BFrag/boffea/{executable} --datasets {dataset} --subjob {isubjob} {this_nsubjobs} -w 1\n")

	files_to_transfer = [f"{tarball_dir}/usercode.tar.gz", f"{tarball_dir}/venv.tar.gz"]

	# Make condor command
	csub_command = f"csub {working_subdir}/run.sh -F {','.join(files_to_transfer)} -n {this_nsubjobs} -d {working_subdir} --mem 6000 2>&1"
	print(csub_command)
	with open(f"{working_subdir}/csub_command.sh", "w") as csub_script:
		csub_script.write("#!/bin/bash\n")
		csub_script.write(csub_command + "\n")

	# Make resubmit command
	# - Lazy: rerun whole job if not all coffea outputs are present
	with open(f"{working_subdir}/resubmit.sh", "w") as resubmit_script:
		resubmit_script.write("""
#!/bin/bash
# NSUCCESS=$(ls -l *coffea | wc -l)
NFAIL=$(find . -name "*coffea" -size -100c -exec stat -c "%s %n" {{}} \; | wc -l)
NTOTAL={}
#if [ \"$NSUCCESS\" -ne \"$NTOTAL\" ]; then
if [ \"$NFAIL\" -ge 1 ]; then 
    source csub_command.sh
fi
""".format(this_nsubjobs))
	global_resubmit_script.write(f"cd {working_subdir}\n")
	global_resubmit_script.write(f"source resubmit.sh\n")

	os.system(csub_command)



