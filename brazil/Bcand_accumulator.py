from six import with_metaclass
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy as np
import coffea.processor as processor
import uproot


class Bcand_accumulator(dict, processor.AccumulatorABC):
    '''A dictionary of column_accumulators for storing B candidates

    Parameters
    ----------
        cols : list[str]
            List of columns to save
    '''
    def __init__(self, cols=[]): # , "l_xy", "l_xy_unc", "sv_prob", "cos2D"]):#, name="Bcands", outputfile=None, reuseoutputfile=None):
        self._initialized = False
        self._cols = []

        if len(cols) > 0:
            self.initialize_columns(cols)

        #if outputfile:
        #    print("Saving {} output to {}".format(name, outputfile))
        #    self._outputfile = uproot.recreate(outputfile)
        #    branchdict = {}
        #    for col in cols:
        #        branchdict[col] = np.float64
        #    self._outputfile[self._name] = uproot.newtree(branchdict)
        #elif reuseoutputfile:
        #    self._outputfile = reuseoutputfile
        #else:
        #    self._outputfile = None

    def initialize_columns(self, cols):
        self._cols = cols
        for col in self._cols:
            self[col] = processor.column_accumulator(np.array([]))
        self._initialized = True

    def identity(self):
        return Bcand_accumulator(self._cols)#, name=self._name, reuseoutputfile=self._outputfile)

    def add(self, other):
        if not isinstance(other, Bcand_accumulator):
            raise ValueError("Bcand_accumulator cannot be added to %r" % type(other))            

        if not self._initialized:
            print("DEBUG : initializing columns from other")
            print(other.keys())
            self.initialize_columns(other.keys())

        if sorted(other.keys()) != sorted(self.keys()):
            print("DEBUG : self.keys() = ")
            print(sorted(self.keys()))
            print("DEBUG : other.key() = ")
            print(sorted(other.keys()))
            raise ValueError("Cannot add two Bcand_accumulator objects with different columns")

        for col in self._cols:
            self[col].add(other[col])

        #if self._outputfile:
        #    branchdict = {}
        #    for col in self._cols:
        #        branchdict[col] = self[col].value()
        #    self._outputfile[self._name].extend(branchdict)


    def set(self, col, arr):
        if not col in self._cols:
            return ValueError(f"Nonexistent column {col}")
        self[col] = processor.column_accumulator(arr)

    def extend(self, branchdict):
        for col, val in branchdict.items():
            if not col in self._cols:
                raise ValueError(f"Nonexistent column {col}")
            #print("DEBUG : col = {}".format(col))
            self[col].add(processor.column_accumulator(val))

    def add_column(self, colname, arr):
        self._cols.append(colname)
        self[colname] = processor.column_accumulator(arr)
