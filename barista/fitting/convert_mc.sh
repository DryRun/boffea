#!/bin/bash
python coffea2roofit.py -i /home/dyu7/BFrag/boffea/barista/Bs2PhiJpsi2KKMuMu/MCEfficiencyHistograms_Bs.coffea -j all -o MCEfficiency_Bs.root
python coffea2roofit.py -i /home/dyu7/BFrag/boffea/barista/Bu2KJpsi2KMuMu/MCEfficiencyHistograms_Bu.coffea -j all -o MCEfficiency_Bu.root
python coffea2roofit.py -i /home/dyu7/BFrag/boffea/barista/Bd2KsJpsi2KPiMuMu/MCEfficiencyHistograms_Bd.coffea -j all  -o MCEfficiency_Bd.root

# Bcands_recomatch,Bcands_tagmatch_HLT_Mu7_IP4,Bcands_probematch_HLT_Mu7_IP4
# Bcands_recomatch,Bcands_tagmatch,Bcands_probematch,Bcands_recomatchswap,Bcands_tagmatchswap,Bcands_probematchswap
#Bcands_tagmatch_HLT_Mu7_IP4,Bcands_probematch_HLT_Mu7_IP4
#Bcands_tagmatch_HLT_Mu7_IP4,Bcands_probematch_HLT_Mu7_IP4,Bcands_tagmatchswap_HLT_Mu7_IP4,Bcands_probematchswap_HLT_Mu7_IP4
python coffea2roofit.py -i /home/dyu7/BFrag/boffea/barista/Bu2PiJPsi2PiMuMu/MCEfficiencyHistograms.coffea -j Bcands_recomatch -o MCEfficiency_Bu2PiJpsi.root


#python coffea2roofit.py -i /home/dyu7/BFrag/boffea/barista/Bs2PhiJpsi2KKMuMu/MCEfficiencyHistograms.coffea -j Bcands_Bs_opt -o optimization_Bs_mc.root
