#!/bin/bash
#if uname -r | grep -q el6; then
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-slc6-gcc8-opt/setup.sh
#else
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh
#fi
#source /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.20.02-232c4/x86_64-centos7-gcc8-opt/bin/thisroot.sh
#if uname -r | grep -q el6; then
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_98apython3/x86_64-slc6-gcc8-opt/setup.sh
#else
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_98apython3/x86_64-centos7-gcc8-opt/setup.sh
#fi
#source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc8-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-clang10-opt/setup.sh

source venv/bin/activate
export BHOME=/home/dyu7/BFrag/boffea
export BDATA=/home/dyu7/BFrag/data
