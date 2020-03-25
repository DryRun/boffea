#!/bin/bash
python coffea2rootfit.py -i ../Bs2PhiJpsiToKKMuMu/MCEfficiencyHistograms.coffea -j Bcands_tagmatch,Bcands_probematch -o MCEfficiency_Bs.root
python coffea2rootfit.py -i ../Bu2KJpsi2KMuMu/MCEfficiencyHistograms.coffea -j Bcands_tagmatch,Bcands_probematch -o MCEfficiency_Bu.root
python coffea2rootfit.py -i ../Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms.coffea -j Bcands_tagmatch,Bcands_probematch -o MCEfficiency_Bd.root
