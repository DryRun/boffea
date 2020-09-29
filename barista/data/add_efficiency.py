#! /usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import os
import sys
import math
import concurrent.futures
import gzip
import pickle
import json
import time
import numexpr
import array
from functools import partial
import re

import uproot
import numpy as np
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
import awkward
import copy
from coffea.analysis_objects import JaggedCandidateArray

from brazil.aguapreta import *
import brazil.dataframereader as dataframereader
from brazil.Bcand_accumulator import Bcand_accumulator

np.set_printoptions(threshold=np.inf)

def add_efficiency(coffea_file):
	# Load efficiency histograms into coffea evaluator
	for btype in ["Bu", "Bs", "Bd"]: 
		

	coffea_stuff = util.load(coffea_file)
	for key in coffea_stuff:



	util.save(coffea_stuff, coffea_file.replace(".coffea", "_eff.coffea"))




  def __init__(self, input_arr, eff_hist):
  	ext = extractor()
  	ext.add_weight_sets["eff eff_pt_y ]

