#!/bin/bash
#if uname -r | grep -q el6; then
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_98apython3/x86_64-slc6-gcc8-opt/setup.sh
#else
#  source /cvmfs/sft.cern.ch/lcg/views/LCG_98apython3/x86_64-centos7-gcc8-opt/setup.sh
#fi
source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-clang10-opt/setup.sh

python -m venv --copies venv
echo "export PYTHONPATH=/user_data/dryu/BFrag/boffea/venv/lib/python3.8/site-packages:/user_data/dryu/BFrag/boffea:$PYTHONPATH
" >> venv/bin/activate
cp venv_95/bin/csub venv/bin/

source venv/bin/activate
#python -m pip install setuptools pip --upgrade
#python -m pip install coffea
#python -m pip install xxhash
pip install --editable . 
