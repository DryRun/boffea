#!/bin/bash
#python fitmc_Bu2KMuMu.py --fits --plots --tables --all
#python fitmc_Bs2KKMuMu.py --fits --plots --tables --all
#python fitmc_Bd2KPiMuMu.py --prefits --save_prefits --all

python fitmc_Bu2KMuMu.py --fits --plots --tables --all --selection nominal >& log_Bu_nominal.txt
python fitmc_Bs2KKMuMu.py --fits --plots --tables --all --selection nominal >& log_Bs_nominal.txt
python fitmc_Bd2KPiMuMu.py --prefits --save_prefits --all --selection nominal >& log_Bd_nominal.txt

python fitmc_Bu2KMuMu.py --fits --plots --tables --all --fitfunc hypatia --selection nominal >& log_Bu_nominal_hypatia.txt
python fitmc_Bs2KKMuMu.py --fits --plots --tables --all --fitfunc hypatia --selection nominal >& log_Bs_nominal_hypatia.txt
python fitmc_Bd2KPiMuMu.py --prefits --save_prefits --all --fitfunc hypatia --selection nominal >& log_Bd_nominal_hypatia.txt

#python fitmc_Bu2KMuMu.py --fits --plots --tables --all --fitfunc 2gauss --selection nominal >& log_Bu_nominal_2gauss.txt
#python fitmc_Bs2KKMuMu.py --fits --plots --tables --all --fitfunc 2gauss --selection nominal >& log_Bs_nominal_2gauss.txt
#python fitmc_Bd2KPiMuMu.py --prefits --save_prefits --all --fitfunc 2gauss --selection nominal >& log_Bd_nominal_2gauss.txt

python fitmc_Bu2KMuMu.py --fits --plots --tables --all --fitfunc 3gauss --selection nominal >& log_Bu_nominal_3gauss.txt
python fitmc_Bs2KKMuMu.py --fits --plots --tables --all --fitfunc 3gauss --selection nominal >& log_Bs_nominal_3gauss.txt
python fitmc_Bd2KPiMuMu.py --prefits --save_prefits --all --fitfunc 3gauss --selection nominal >& log_Bd_nominal_3gauss.txt
