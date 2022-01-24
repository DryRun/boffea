#!/bin/bash
cd /home/dyu7/BFrag/boffea/barista/Bd2KsJpsi2KPiMuMu
python mc_efficiency_processor.py >& log.txt
cd /home/dyu7/BFrag/boffea/barista/Bs2PhiJpsi2KKMuMu
python mc_efficiency_processor.py >& log.txt
cd /home/dyu7/BFrag/boffea/barista/Bu2KJpsi2KMuMu
python mc_efficiency_processor.py >& log.txt
#cd /home/dyu7/BFrag/boffea/barista/Bu2PiJPsi2PiMuMu
#python mc_efficiency_processor.py
cd /home/dyu7/BFrag/boffea/barista
