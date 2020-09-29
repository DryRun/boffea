#!/bin/bash
python coffea2roofit.py -i ../Bs2PhiJpsi2KKMuMu/MCEfficiencyHistograms.coffea -j Bcands_recomatch -o MCEfficiency_Bs.root
python coffea2roofit.py -i ../Bu2KJpsi2KMuMu/MCEfficiencyHistograms.coffea -j Bcands_recomatch -o MCEfficiency_Bu.root
python coffea2roofit.py -i ../Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms.coffea -j Bcands_recomatch,Bcands_recomatchswap -o MCEfficiency_Bd.root

#Bcands_tagmatch_HLT_Mu7_IP4,Bcands_probematch_HLT_Mu7_IP4
#Bcands_tagmatch_HLT_Mu7_IP4,Bcands_probematch_HLT_Mu7_IP4,Bcands_tagmatchswap_HLT_Mu7_IP4,Bcands_probematchswap_HLT_Mu7_IP4
python coffea2roofit.py -i ../Bu2PiJPsi2PiMuMu/MCEfficiencyHistograms.coffea -j Bcands_recomatch -o MCEfficiency_Bu2PiJpsi.root


python coffea2roofit.py -i ../Bs2PhiJpsi2KKMuMu/MCEfficiencyHistograms.coffea -j Bcands_Bs_opt -o optimization_Bs_mc.root
