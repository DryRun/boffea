from six import with_metaclass
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy


class Bcand_accumulator(AccumulatorABC):
    '''A dictionary of column_accumulators n appendable numpy ndarray

    Parameters
    ----------
        value : numpy.ndarray
            The identity value array, which should be an empty ndarray
            with the desired row shape. The column dimension will correspond to
            the first index of `value` shape.

    Examples
    --------
    If a set of accumulators is defined as::

        a = column_accumulator(np.array([]))
        b = column_accumulator(np.array([1., 2., 3.]))
        c = column_accumulator(np.array([4., 5., 6.]))

    then:

    >>> a + b
    column_accumulator(array([1., 2., 3.]))
    >>> c + b + a
    column_accumulator(array([4., 5., 6., 1., 2., 3.]))
    '''
    def __init__(self, value):
        if not isinstance(value, numpy.ndarray):
            raise ValueError("column_accumulator only works with numpy arrays")
        self._empty = numpy.zeros(dtype=value.dtype, shape=(0,) + value.shape[1:])
        self._value = value

    def __repr__(self):
        return "column_accumulator(%r)" % self.value

    def identity(self):
        return column_accumulator(self._empty)

    def add(self, other):
        if not isinstance(other, column_accumulator):
            raise ValueError("column_accumulator cannot be added to %r" % type(other))
        if other._empty.shape != self._empty.shape:
            raise ValueError("Cannot add two column_accumulator objects of dissimilar shape (%r vs %r)"
                             % (self._empty.shape, other._empty.shape))
        self._value = numpy.concatenate((self._value, other._value))

    @property
    def value(self):
        '''The current value of the column

        Returns a numpy array where the first dimension is the column dimension
        '''
        return self._value

