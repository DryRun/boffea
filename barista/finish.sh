#!/bin/bash
python efficiency.py --selection nominal
python ffrs.py --all --selection nominal --fitfunc poly
python ffrs.py --all --selection nominal --fitfunc 3gauss
python ffrs.py --all --selection nominal --fitfunc johnson
#python efficiency.py --selection HiTrkPt
#python ffrs.py --selection HiTrkPt --fitfunc johnson
