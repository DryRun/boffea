#!/bin/bash
#python fitdata_Bs2KKMuMu_hypatia.py --all --fits --plots --tables
#python fitdata_Bu2KMuMu_hypatia.py --all --fits --plots --tables
#python fitdata_Bd2KPiMuMu_hypatia.py --all --fits --plots --tables

python fitdata_Bs2KKMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc johnson >& log_data_Bs_nominal.txt & 
python fitdata_Bu2KMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc johnson >& log_data_Bu_nominal.txt 
python fitdata_Bd2KPiMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc johnson >& log_data_Bd_nominal.txt &

#python fitdata_Bs2KKMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc hypatia >& log_data_Bs_nominal_hypatia.txt & 
#python fitdata_Bu2KMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc hypatia >& log_data_Bu_nominal_hypatia.txt 
#python fitdata_Bd2KPiMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc hypatia >& log_data_Bd_nominal_hypatia.txt &

python fitdata_Bs2KKMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc 3gauss >& log_data_Bs_nominal_3gauss.txt & 
python fitdata_Bu2KMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc 3gauss >& log_data_Bu_nominal_3gauss.txt 
python fitdata_Bd2KPiMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc 3gauss >& log_data_Bd_nominal_3gauss.txt &

python fitdata_Bs2KKMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc poly >& log_data_Bs_nominal_poly.txt & 
python fitdata_Bu2KMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc poly >& log_data_Bu_nominal_poly.txt 
python fitdata_Bd2KPiMuMu.py --all --fits --plots --tables --binned --selection nominal --fitfunc poly >& log_data_Bd_nominal_poly.txt &

#python fitdata_Bs2KKMuMu.py --all --fits --plots --tables --binned --selection VarMuonPt --fitfunc johnson >& log_data_Bs_VarMuonPt.txt & 
#python fitdata_Bu2KMuMu.py --all --fits --plots --tables --binned --selection VarMuonPt --fitfunc johnson >& log_data_Bu_VarMuonPt.txt 
#python fitdata_Bd2KPiMuMu.py --all --fits --plots --tables --binned --selection VarMuonPt --fitfunc johnson >& log_data_Bd_VarMuonPt.txt &
