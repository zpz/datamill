import logging

from ._transformer import (
    FloatTransformer, IntTransformer, StrTransformer,
    DirectFeature, Dummifier, LengthChecker, HasValueChecker, KeywordsChecker,
    Timestamp2WeekdayHourDummifier, OS, Browser,
    fill_one_float, fill_one_int, fill_one_str,
    fill_one_float_row, fill_one_int_row, fill_one_str_row
)


logger = logging.getLogger(__name__)


# Notes on categorical features:
#
# Categorical features should be of string type;
# this should be guaranteed by the user;
# we do not do string type conversions on the input values.
#
# As all input values should be strings, there will be no
# values like `None` or `numpy.nan`.
# User should ensure that missing values are consistently
# represented by a fixed value, such as the empty string, '',
# or the likes of 'na', 'unknown', 'missing', 'n/a', etc.
#
# Different features do not need to use the same string to represent
# a missing value.

# In general, the correct type of a feature---`float` or `int` or `str`---should
# be guaranteed in pre-processing. Once passed into this module, such basic type check
# is not conducted.
#
# Missing values should be represented by special values that are not too Pythonic
# like `None` and `numpy.nan`.
# For example, one can use -1 for numerical features and '' (empty string) for textual features.



class Feature:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}(name=\'{}\')'.format(self.__class__.__name__, self.name)


class FloatFeature(Feature):
    pass


class IntFeature(Feature):
    pass


class StrFeature(Feature):
    pass


class Transplexer:
    """
    A `Transplexer` turns one or more 'features' (i.e. input values)
    to one or more 'predictors'.

    This should be contrasted to `Transformer`, which turns one 'feature'
    to one or more 'predictors'.
    """
    def __str__(self):
        return '{}: {} ==> {}'.format(
            self.__class__.__name__,
            self.features, self.predictor_names)

    @property
    def float_features(self):
        return [f for f in self.features if isinstance(f, FloatFeature)]

    @property
    def int_features(self):
        return [f for f in self.features if isinstance(f, IntFeature)]

    @property
    def str_features(self):
        return [f for f in self.features if isinstance(f, StrFeature)]


class TimestampRecency(Transplexer):
    def __init__(self, timestamp1, timestamp2, predictor_name=None):
        """
        Args:
            timestamp1, timestamp2: `IntFeature`s specifying the timestamp features.
                `timestamp1` is an earlier time than `timestamp2`.

        Arguments to methods `_fill` and `_fill_row` other than the common ones including
        `self`, `out`, `start`, `row`, `col_start`, must be defined in the same order
        as `timestamp1` and `timestamp2` as arguments to `__init__`.
        """
        assert isinstance(timestamp1, IntFeature)
        assert isinstance(timestamp2, IntFeature)
        self.features = [timestamp1, timestamp2]
        self.feature_names = [f.name for f in self.features]
        self.predictor_names = [predictor_name or timestamp1.name + '_recency']
        self.n_features = len(self.features)
        self.n_predictors = len(self.predictor_names)

    def __repr__(self):
        return '{}(timestamp1={}, timestamp2={}, predictor_name=\'{}\')'.format(
            self.__class__.__name__,
            self.features[0],
            self.features[1],
            self.predictor_names[0]
        )

    def _fill(self, out, ts1, ts2, start=0):
        out[start] = (ts2 - ts1) / 3600.0    # hours
        return 1  # self.n_predictors

    def _fill_row(self, out, ts1, ts2, row=0, col_start=0):
        out[row, col_start] = (ts2 - ts1) / 3600.0    # hours
        return 1  # self.n_predictors

    def _fill_col(self, out, ts1, ts2, row_start=0, col_start=0):
        n = len(ts1)
        out[row_start : (row_start + n), col_start] = (ts2 - ts1) / 3600.0  # hours
        return 1  # self.n_predictors
