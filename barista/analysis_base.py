#! /usr/bin/env python

import ROOT
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import time
import datetime
import uproot
import awkward
import sys
import os
import numpy as np

from enum import Enum
class DataSource(Enum):
  kDATA = 1
  kMC = 2

class AnalysisBase(object):
  def __init__(self, inputfiles, outputfile, inputbranches, hist=False):
    __metaclass__ = ABCMeta
    self._file_out_name = outputfile.replace('.root','').replace('.h5','')
    self._file_in_name = inputfiles
    self._num_files = len(self._file_in_name)
    self._inputbranches = inputbranches
    self._hist = hist
    self._ifile = 0

  def fill_output(self):
    print('[AnalysisBase::fill_output] INFO: FILE: {}/{}. Filling the output {}...'.format(self._ifile+1, self._num_files, 'histograms' if self._hist else 'tree'))
    if self._hist:
      for hist_name, hist_bins in sorted(self._outputbranches.items()):
        if hist_name in self._branches.keys():
          branch_np = self._branches[hist_name].values
          fill_hist(self._hist_list[hist_name], branch_np[np.isfinite(branch_np)])
    else:
      self._branches = self._branches[self._outputbranches.keys()]
      #self._branches.to_hdf(self._file_out_name+'.h5', 'branches', mode='a', format='table', append=True)
      self._branches.to_root(self._file_out_name+'.root', key='tree', mode='a')

  def finish(self):
    print('[AnalysisBase::finish] INFO: Merging the output files...')
    #os.system("hadd -k -f {}.root {}_subset*.root".format(self._file_out_name, self._file_out_name))
    #os.system("rm {}_subset*.root".format(self._file_out_name))
    if self._hist:
      for hist_name, hist in sorted(self._hist_list.items()):
        hist.write()
      self._file_out.close()
    else:
      pass


  def print_timestamp(self):
    ts_start = time.time()
    print("[BParkingNANOAnalysis::print_timestamp] INFO : Time: {}".format(datetime.datetime.fromtimestamp(ts_start).strftime('%Y-%m-%d %H:%M:%S')))


  @abstractmethod
  def run(self):
    pass








