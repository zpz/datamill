from collections import Counter
from datetime import datetime
import logging

import numpy as np

cimport cython


logger = logging.getLogger(__name__)


cdef class Transformer:
    cdef readonly str feature_name
    cdef readonly list predictor_names
    cdef readonly int n_predictors

    def __init__(self, feature_name, predictor_names=None):
        self.feature_name = feature_name
        if predictor_names is None:
            predictor_names = [feature_name]
        self.predictor_names = predictor_names
        self.n_predictors = len(predictor_names)

    cpdef int _fill(self, double[:] out, x, int start=0):
        """
        Fill non-zero values in `out` at appropriate indices.

        Args:
            x: a single value corresponding to `feature_name` of this object.

            out: a 1-D (slice into a) array of floats.
                The array should have been zeroed before
                being passed in.

        Returns:
            Number of elements of `out` processed by this function.

        This method should always modify values in `out`
        using single index syntax so that the changes
        will be made to one element of `out` at a time.
        E.g.
            out[0] = 3.0
            out[2] = 1.0

        Never directly assign values to `out`, like
            out = 3
        as that will create a new local variable `out`.

        Do not assign to multiple elements at once using `numpy` syntax, like
            out[2:5] = 1.0
        as this does not work with other iterable types, esp `memoryview`s.
        """
        raise NotImplementedError

    cpdef int _fill_row(self, double[:, :] out, x, int row=0, int col_start=0):
        """
        Analogous to `_fill`, but the output space `out` is 2D, hence the filling
        is in spacified row.
        """
        return self._fill(out[row, :], x, col_start)

    cpdef int _fill_col(self, double[:, :] out, x, int row_start=0, int col_start=0):
        """
        In contrast to `_fill`, which concerns a single observation,
        this method processes multiple observations in `x` and fill in a block of `out`,
        one row per observation.
        """
        cdef int irow
        for irow, xx in enumerate(x):
            self._fill(out[row_start + irow, :], xx, col_start)
        return self.n_predictors

    cpdef double[:] one(self, x):
        """
        Analogous to `_fill`, but output space is not give, hence
        is created internally and returned.
        """
        out = np.zeros(self.n_predictors)
        self._fill(out, x)
        return out

    cpdef double[:,:] many_by_row(self, x):
        out = np.zeros((len(x), self.n_predictors), order='C')
        for irow, v in enumerate(x):
            self._fill(out[irow, :], v)
        return out

    cpdef double[:, :] many_by_col(self, x):
        out = np.zeros((len(x), self.n_predictors), order='C')
        self._fill_col(out, x)
        return out

    cpdef double[:, :] many(self, x):
        return self.many_by_row(x)

    def __repr__(self):
        return '{}(feature_name={}, predictor_names={})'.format(
            self.__class__.__name__,
            repr(self.feature_name),
            repr(self.predictor_names)
        )

    def __getstate__(self):
        return {
            'feature_name': self.feature_name,
            'predictor_names': self.predictor_names,
        }

    def __setstate__(self, state):
        self.feature_name = state['feature_name']
        self.predictor_names = state['predictor_names']
        self.n_predictors = len(self.predictor_names)


cdef class FloatTransformer(Transformer):
    pass


cdef class IntTransformer(Transformer):
    pass


cdef class StrTransformer(Transformer):
    pass


cdef class DirectFeature(FloatTransformer):
    """
    A 'direct' feature is a meaningful numeric value that is used directly
    as a predictor, without any transformations, except for treatment of 'missingness'.
    As per numerical modeling software, this must be a `float` type.

    If an integer is intended to be used as a 'direct' feature,
    it's better that the user casts it to float type in pre-processing,
    rather than passing it as an 'indirect' feature that requires an `int`-to-`float`
    type conversion.

    There is no class for 'indirect' feature---any feature that is not 'direct'
    is in effect 'indirect'.
    """

    cpdef int _fill(self, double[:] out, x, int start=0):
        out[start] = x
        return 1  # self.n_predictors

    cpdef int _fill_row(self, double[:,:] out, x, int row=0, int col_start=0):
        out[row, col_start] = x
        return 1  # self.n_predictors

    cpdef int _fill_col(self, double[:,:] out, x, int row_start=0, int col_start=0):
        cdef int n = len(x)
        cdef int irow
        cdef double xx
        for irow in range(n):
            xx = x[irow]
            out[row_start + irow, col_start] = xx
        return 1  # self.n_predictors


cdef class OS(StrTransformer):
    def __init__(self, feature_name):
        predictor_names = ['apple', 'google', 'ms']
        super().__init__(feature_name, predictor_names)

    cpdef int _fill(self, double[:] out, x, int start=0):
        xx = x.lower()
        if 'ios' in xx or 'os x' in xx or 'osx' in xx or 'mac' in xx:
            out[start] = 1.0
        elif 'android' in xx or 'chrome' in xx or 'google' in xx:
            out[start + 1] = 1.0
        elif 'win' in xx:
            out[start + 2] = 1.0
        return self.n_predictors


cdef class Browser(StrTransformer):
    def __init__(self, feature_name):
        predictor_names = ['safari', 'chrome', 'ie', 'firefox']
        super().__init__(feature_name, predictor_names)

    cpdef int _fill(self, double[:] out, x, int start=0):
        xx = x.lower()
        if 'safari' in xx:
            out[start] = 1.0
        elif 'chrome' in xx:
            out[start + 1] = 1.0
        elif 'ie' in xx:
            out[start + 2] = 1.0
        elif 'firefox' in xx:
            out[start + 3] = 1.0
        return self.n_predictors


cdef class Dummifier(StrTransformer):
    cdef readonly dict values

    def __init__(self, feature_name, values, max_values=32, value_freq_cutoff=0.0001):
        """
        Args:
            values: iterable of strings containing values of the categorical variable.
        """

        c = Counter(v for v in values if v != '')
        if len(c) > max_values:
            cc = c.most_common(max_values)
            for i in range(1, max_values):
                if cc[i][1] <= cc[0][1] * value_freq_cutoff:
                    cc = cc[:i]
                    break
            logger.debug('feature "%s": %d most common values out of %d: %s',
                         feature_name, len(cc), len(c), cc)
            uvalues = [v[0] for v in cc]
        else:
            uvalues = [v[0] for v in c.most_common()]

        predictor_names = [feature_name + '_' + '_'.join(str(v).split()) for v in uvalues]
        super().__init__(feature_name, predictor_names)
        self.values = {v:i for i,v in enumerate(uvalues)}
            # Although 'uvalues' order values by decreasing frequency,
            # this dict may destroy that order and lost potential benefit of that.

    def __repr__(self):
        return '{}(feauture_name={}, values={})'.format(
            self.__class__.__name__,
            repr(self.feature_name),
            repr(list(self.values.keys()))
        )

    def __str__(self):
        values = list(self.values.keys())
        if len(values) > 64:
            nn = len(values) - 64
            values = '[' + ', '.join(values[:64]) + ',  ... and %d more values ... ]' % nn
        else:
            values = str(values)
        return '{}(feauture_name={}, values={})'.format(
            self.__class__.__name__,
            str(self.feature_name),
            values
        )

    # TODO: this is a time consumer
    cpdef int _fill(self, double[:] out, x, int start=0):
        cdef int idx
        idx = self.values.get(x, -1)
        if idx >= 0:
            out[start+idx] = 1.0
        return self.n_predictors

    cpdef int _fill_col(self, double[:,:] out, x, int row_start=0, int col_start=0):
        cdef int idx
        cdef int n = len(x)
        cdef int irow
        values = self.values
        for irow in range(n):
            idx = values.get(x[irow], -1)
            if idx >= 0:
                out[row_start + irow, col_start + idx] = 1.0
        return self.n_predictors


    def __getstate__(self):
        s = super().__getstate__()
        return (s, {'values': self.values})

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.values = state[1]['values']


cdef class LengthChecker(StrTransformer):
    cdef int length

    def __init__(self, feature_name, length):
        super().__init__(feature_name)
        self.length = length

    def __repr__(self):
        return '{}(feature_name={}, length={})'.format(
            self.__class__.__name__,
            repr(self.feature_name),
            repr(self.length)
        )

    cpdef int _fill(self, double[:] out, x, int start=0):
        if len(x) == self.length:
            out[start] = 1.0
        return self.n_predictors

    def __getstate__(self):
        s = super().__getstate__()
        return (s, {'length': self.length})

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.length = state[1]['length']


cdef class HasValueChecker(StrTransformer):
    def __init__(self, feature_name):
        super().__init__(feature_name)

    cpdef int _fill(self, double[:] out, x, int start=0):
        if x != '':
            out[start] = 1.0
        return self.n_predictors


cdef class KeywordsChecker(StrTransformer):
    cdef list keywords

    def __init__(self, feature_name, keywords):
        """
        Args:
            keywords: iterable of strings containing keywords.
        """
        keywords = sorted(list(set(x for x in keywords if x != '')))
        predictor_names = [feature_name + '_' + '_'.join(v.split()) for v in keywords]
        super().__init__(feature_name, predictor_names)
        self.keywords = keywords

    def __repr__(self):
        return '{}(feature_name={}, keywords={})'.format(
            self.__class__.__name__,
            repr(self.feature_name),
            repr(self.keywords)
        )

    cpdef int _fill(self, double[:] out, x, int start=0):
        cdef int idx
        xx = x.split()
        for idx, k in enumerate(self.keywords):
            if k in xx:
                out[start + idx] = 1.0
        return self.n_predictors

    def __getstate__(self):
        s = super().__getstate__()
        return (s, {'keywords': self.keywords})

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.keywords = state[1]['keywords']


cdef class WeekdayDummifier(IntTransformer):
    """
    Input is an ISO weekday number, i.e. 1,..., 7 for Monday,..., Sunday.
    """

    def __init__(self, feature_name):
        WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        predictor_names = [feature_name + '_is_' + d for d in WEEKDAYS]
        super().__init__(feature_name, predictor_names)

    cpdef int _fill(self, double[:] out, x, int start=0):
        cdef int xx = x
        out[start + xx - 1] = 1.0
        return self.n_predictors


cdef class HourDummifier(IntTransformer):
    """
    Input is hour number 0,..., 23.
    """
    # cdef str feature_name
    # cdef list predictor_names
    # cdef int n_predictors

    def __init__(self, feature_name):
        predictor_names = ['{}_is_{:0>2}'.format(feature_name, i) for i in range(24)]
        super().__init__(feature_name, predictor_names)

    cpdef int _fill(self, double[:] out, x, int start=0):
        cdef int xx = x
        out[start + xx] = 1.0
        return self.n_predictors


cdef class Timestamp2WeekdayHourDummifier(IntTransformer):
    cdef WeekdayDummifier wd_dummifier
    cdef HourDummifier h_dummifier

    def __init__(self, feature_name):
        wd_dummifier = WeekdayDummifier(feature_name + '_weekday')
        h_dummifier = HourDummifier(feature_name + '_hour')
        predictor_names = wd_dummifier.predictor_names + h_dummifier.predictor_names
        super().__init__(feature_name, predictor_names)
        self.wd_dummifier = wd_dummifier
        self.h_dummifier = h_dummifier

    # TODO: this is a time consumer
    cpdef int _fill(self, double[:] out, x, int start=0):
        """
        :type x: int (timestamp such as 1483666858)
        """
        cdef int weekday, hour, n
        weekday, hour = shift_timestamp_to(x)
        n = self.wd_dummifier._fill(out, weekday, start)
        n += self.h_dummifier._fill(out, hour, start + n)
        return n

    def __setstate__(self, state):
        super().__setstate__(state)
        wd_dummifier = WeekdayDummifier(self.feature_name + '_weekday')
        h_dummifier = HourDummifier(self.feature_name + '_hour')
        self.wd_dummifier = wd_dummifier
        self.h_dummifier = h_dummifier


cdef tuple shift_timestamp_to(long ts):
    cdef long ts0 = 1489363200   # 2017-03-13 0:0:0 UTC, Monday
    cdef long weekday0 = 1   # ISO weekday: Monday is 1, Sunday is 7

    cdef long HOUR_SECONDS = 3600
    cdef long DAY_SECONDS = 86400
    cdef long WEEK_SECONDS = 604800

    cdef long ts_delta = ts - ts0
    if ts_delta < 0:
        ts_delta += ((-ts_delta) // WEEK_SECONDS + 1) * WEEK_SECONDS

    cdef long td, nday, seconds
    cdef int weekday, hour

    td = ts_delta % WEEK_SECONDS
    nday, seconds = divmod(td, DAY_SECONDS)
    weekday = weekday0 + nday
    if weekday > 7:
        weekday = weekday - 7
    hour = seconds // HOUR_SECONDS
    if hour > 23:
        hour = 23
    return weekday, hour   # 1,..., 7; 0, 1,..., 23


def fill_one_float(double[:] out, list float_transformers, data, int start):
    """
    Args:
        data: iterable of feature values, one value per transformer in `float_transformers`.

    Take values in `data`, corresponding transformers in `float_transformers`,
    generate predictors and fill into `out` starting at index `start`.
    """
    cdef int n = 0
    cdef int nt = len(float_transformers)
    cdef int it
    cdef FloatTransformer t
    cdef double val
    for it in range(nt):
        t = float_transformers[it]
        val = data[it]
        n += t._fill(out, val, start + n)
    return n


def fill_one_int(double[:] out, list int_transformers, data, int start):
    cdef int n = 0
    cdef int nt = len(int_transformers)
    cdef int it
    cdef IntTransformer t
    cdef long val
    for it in range(nt):
        t = int_transformers[it]
        val = data[it]
        n += t._fill(out, val, start + n)
    return n


def fill_one_str(double[:] out, list str_transformers, data, int start):
    cdef int n = 0
    cdef int nt = len(str_transformers)
    cdef int it
    cdef StrTransformer t
    cdef str val
    for it in range(nt):
        t = str_transformers[it]
        val = data[it]
        n += t._fill(out, val, start + n)
    return n


def fill_one_float_row(double[:,:] out, list float_transformers, double[:,:] data, int row, int col_start):
    """
    Args:
        data: 2-D array.

    Use one row of `data` to generate predictors and fill corresponding row of `out`.
    """
    cdef int n = 0
    cdef int nt = len(float_transformers)
    cdef FloatTransformer t
    cdef int it
    for it in range(nt):
        t = float_transformers[it]
        n += t._fill_row(out, data[row, it], row, col_start + n)
    return n


def fill_one_int_row(double[:,:] out, list int_transformers, long[:,:] data, int row, int col_start):
    cdef int n = 0
    cdef int nt = len(int_transformers)
    cdef IntTransformer t
    cdef int it
    for it in range(nt):
        t = int_transformers[it]
        n += t._fill_row(out, data[row, it], row, col_start + n)
    return n


#@cython.binding(True)
def fill_one_str_row(double[:,:] out, list str_transformers, data, int row, int col_start):
    cdef int n = 0
    cdef int nt = len(str_transformers)
    cdef StrTransformer t
    cdef int it
    for it in range(nt):
        t = str_transformers[it]
        n += t._fill_row(out, data[row, it], row, col_start + n)
    return n
