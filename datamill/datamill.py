import itertools
import logging

import numpy as np

from .transformer import (
    FloatTransformer, IntTransformer, StrTransformer,
    DirectFeature, Dummifier, LengthChecker, HasValueChecker, KeywordsChecker,
    Timestamp2WeekdayHourDummifier, OS, Browser,
    fill_one_float, fill_one_int, fill_one_str,
    fill_one_float_row, fill_one_int_row, fill_one_str_row
)
from .transformer import (
    FloatFeature, IntFeature, StrFeature,
    Transplexer, TimestampRecency
)

from .util import pretty_log_batch_size


from .util import _locate_features

logger = logging.getLogger(__name__)


def _find_transplexer_features(transplexer, float_feature_names, int_feature_names, str_feature_names):
    args = []
    for feature in transplexer.features:
        if isinstance(feature, FloatFeature):
            args.append((
                0, float_feature_names.index(feature.name)
            ))
        elif isinstance(feature, IntFeature):
            args.append((
                1, int_feature_names.index(feature.name)
            ))
        elif isinstance(feature, StrFeature):
            args.append((
                2, str_feature_names.index(feature.name)
            ))
        else:
            raise Exception('un-recognized Feature object %s' % feature)
    return args


class DataMill:
    def __init__(self, transformers, transplexers=None):
        """
        The order of elements in `transformers` and `transplexers` does not (completely)
        determine the order of features that this object will later
        accept or produce in processing.
        """
        assert transformers
        if transplexers is None:
            transplexers = []

        self.float_transformers = [t for t in transformers if isinstance(t, FloatTransformer)]
        self.int_transformers = [t for t in transformers if isinstance(t, IntTransformer)]
        self.str_transformers = [t for t in transformers if isinstance(t, StrTransformer)]
        self.transplexers = transplexers

        self.n_predictors = sum(t.n_predictors for t in transformers) + \
                            sum(t.n_predictors for t in transplexers)
        self._one_buffer = np.zeros(self.n_predictors)

        transformer_float_features = [t.feature_name for t in self.float_transformers]
        transplexer_float_features = [f.name for t in self.transplexers for f in t.float_features]
        self._float_feature_names = list(set(transformer_float_features + transplexer_float_features))
        self._float_transformers_feature_idx = _locate_features(
            self._float_feature_names, transformer_float_features)

        transformer_int_features = [t.feature_name for t in self.int_transformers]
        transplexer_int_features = [f.name for t in self.transplexers for f in t.int_features]
        self._int_feature_names = list(set(transformer_int_features + transplexer_int_features))
        self._int_transformers_feature_idx = _locate_features(
            self._int_feature_names, transformer_int_features)

        transformer_str_features = [t.feature_name for t in self.str_transformers]
        transplexer_str_features = [f.name for t in self.transplexers for f in t.str_features]
        self._str_feature_names = list(set(transformer_str_features + transplexer_str_features))
        self._str_transformers_feature_idx = _locate_features(
            self._str_feature_names, transformer_str_features)

        transplexer_args = []
        feature_names = [self._float_feature_names, self._int_feature_names, self._str_feature_names]
        for t in transplexers:
            transplexer_args.append(_find_transplexer_features(t, *feature_names))
        self._transplexer_args = transplexer_args

    def __repr__(self):
        transformers = self.float_transformers + self.int_transformers + self.str_transformers
        transplexers = self.transplexers
        return '{}(transformers=[{}], transplexers=[{}])'.format(
            self.__class__.__name__,
            ', '.join(map(repr, transformers)),
            ', '.join(map(repr, transplexers))
        )

    def __str__(self):
        transformers = self.float_transformers + self.int_transformers + self.str_transformers
        transplexers = self.transplexers
        return '{} with transformers and transplexers\n{}'.format(
            self.__class__.__name__,
            '\n'.join('  ' + str(v) for v in transformers + transplexers),
        )

    @property
    def float_feature_names(self):
        return self._float_feature_names

    @property
    def n_float_features(self):
        return len(self._float_feature_names)

    @property
    def int_feature_names(self):
        return self._int_feature_names

    @property
    def n_int_features(self):
        return len(self._int_feature_names)

    @property
    def str_feature_names(self):
        return self._str_feature_names

    @property
    def n_str_features(self):
        return len(self._str_feature_names)

    @property
    def feature_names(self):
        return self.float_feature_names + self.int_feature_names + self.str_feature_names

    @property
    def n_features(self):
        return self.n_float_features + self.n_int_features + self.n_str_features

    @property
    def predictor_names(self):
        return list(itertools.chain.from_iterable(
            t.predictor_names for t in (
                self.float_transformers +
                self.int_transformers +
                self.str_transformers +
                self.transplexers)))

    def _fill(self, out, x_float=None, x_int=None, x_str=None, start=0):
        """
        Args:
            out (1-d numpy array or memoryview): caller must
                guarantee it has the right shape (and enough size) and is zeroed.
            x_float (iterable of floats): elements must correspond to
                `self.float_feature_names`.
            x_int (iterable of ints): elements must correspond to
                `self.int_feature_names`.
            x_str (iterable of strings): elements must correspond to
                `self.str_feature_names`.
            start: index of `out` to start the filling.

        The input data elements will be processed by their respective
        `Transformer`s and `Transplexer`s to produce 'predictors', which will fill `out`
        in a determined order.

        Returns:
            number of elements of `out` that have been filled in this function.
            It's more precise to say 'processed' instead of 'filled' because
            a `Transformer` may skip spots in `out`, knowing it's zero and needs no change.
        """
        n = 0
        if self.float_transformers:
            n += fill_one_float(
                out,
                self.float_transformers,
                [x_float[i] for i in self._float_transformers_feature_idx],
                start + n)
        if self.int_transformers:
            n += fill_one_int(
                out,
                self.int_transformers,
                [x_int[i] for i in self._int_transformers_feature_idx],
                start + n)
        if self.str_transformers:
            n += fill_one_str(
                out,
                self.str_transformers,
                [x_str[i] for i in self._str_transformers_feature_idx],
                start + n)
        if self.transplexers:
            features = [x_float, x_int, x_str]
            for t, args in zip(self.transplexers, self._transplexer_args):
                values = [features[i][j] for i,j in args]
                n += t._fill(out, *values, start + n)
        return n

    def one(self, x_float=None, x_int=None, x_str=None):
        """
        Given 'raw' input data, produce final feature values
        to be fed directly to models for training and prediction.

        Args:
            x_float, x_int, x_str: 'raw' feature values.
                These values combined represent a single observation.

        Returns:
            `numpy` array of floats containing final feature values which
            are directly usable by models. The elements are in certain order
            that this object knows how to handle.

            The return is the reference to a pre-allocated `numpy` array
            in order to avoid repeated re-allocation.
            The caller usually should treat the return as read-only.
        """
        out = self._one_buffer
        out[:] = 0.0
        self._fill(out, x_float=x_float, x_int=x_int, x_str=x_str)
        return out

    def _fill_row(self, out, x_float=None, x_int=None, x_str=None, row=0, col_start=0):
        """
        Args:
            out: 2D numpy array or memoryview of floats, zeroed.
            x_float, x_int, x_str: `numpy` matrices.
        """
        n = 0
        if self.float_transformers:
            n += fill_one_float_row(
                out,
                self.float_transformers,
                x_float[:, self._float_transformers_feature_idx],
                row,
                col_start + n)
        if self.int_transformers:
            n += fill_one_int_row(
                out,
                self.int_transformers,
                x_int[:, self._int_transformers_feature_idx],
                row,
                col_start + n)
        if self.str_transformers:
            n += fill_one_str_row(
                out,
                self.str_transformers,
                x_str[:, self._str_transformers_feature_idx],
                row,
                col_start + n)
        if self.transplexers:
            features = [x_float[row, :], x_int[row, :], x_str[row, :]]
            for t, args in zip(self.transplexers, self._transplexer_args):
                values = [features[i][j] for i,j in args]
                n += t._fill_row(out, *values, row, col_start + n)
        return n

    def _fill_col(self, out, x_float=None, x_int=None, x_str=None, row_start=0, col_start=0):
        """
        Args:
            out: 2D numpy array or memoryview of floats, zeroed.
            x_float, x_int, x_str: `numpy` matrices.

        Return: number of columns filled.
        """
        n = 0
        for i, f in enumerate(self.float_transformers):
            n += f._fill_col(out, x_float[:, self._float_transformers_feature_idx[i]],
                             row_start, col_start + n)
        for i, f in enumerate(self.int_transformers):
            n += f._fill_col(out, x_int[:, self._int_transformers_feature_idx[i]],
                             row_start, col_start + n)
        for i, f in enumerate(self.str_transformers):
            n += f._fill_col(out, x_str[:, self._str_transformers_feature_idx[i]],
                             row_start, col_start + n)
        if self.transplexers:
            features = [x_float, x_int, x_str]
            for t, args in zip(self.transplexers, self._transplexer_args):
                values = [features[i][:, j] for i,j in args]
                n += t._fill_col(out, *values, row_start, col_start + n)
        return n

    def _get_data_kwargs(self, x_float=None, x_int=None, x_str=None):
        kwargs = {}
        if x_float is not None:
            assert x_float.ndim == 2
            if x_float.size > 0:
                assert x_float.shape[1] == self.n_float_features
                kwargs['x_float'] = x_float
        if x_int is not None:
            assert x_int.ndim == 2
            if x_int.size > 0:
                assert x_int.shape[1] == self.n_int_features
                kwargs['x_int'] = x_int
        if x_str is not None:
            assert x_str.ndim == 2
            if x_str.size > 0:
                assert x_str.shape[1] == self.n_str_features
                kwargs['x_str'] = x_str
        assert kwargs

        n_obs = [v.shape[0] for v in kwargs.values()]
        n = n_obs[0]
        if len(n_obs) > 1:
            assert all(v == n for v in n_obs[1:])
        return kwargs, n

    def grind(self, data):
        """
        Args:
            data: pandas DataFrame.
        """
        out = {}
        if self.float_feature_names:
            out['x_float'] = data[self.float_feature_names].copy().astype(float).as_matrix()
            # TODO: is this `copy()` needed?
        if self.int_feature_names:
            out['x_int'] = data[self.int_feature_names].copy().fillna(-1).astype(int).as_matrix()
        if self.str_feature_names:
            out['x_str'] = data[self.str_feature_names].copy().fillna('').astype(str).as_matrix()

        return out

    def many_by_row(self, *, data=None, x_float=None, x_int=None, x_str=None):
        """
        Matrix counterpart of `self.one`.

        Args:
            data: pandas DataFrame.
                If not `None`, then `x_float`, `x_int`, and `x_str` are ignored.
            x_float: numpy matrix if not `None`.
            x_int: likewise.
            x_str: likewise.

        Each row of the inputs represents one observation.
        """
        if data is not None:
            kwargs = self.grind(data)
        else:
            kwargs = {'x_float': x_float, 'x_int': x_int, 'x_str': x_str}
        kwargs, n_obs = self._get_data_kwargs(**kwargs)

        out = np.zeros((n_obs, self.n_predictors), order='C')

        logger.debug('filling data row-by-row...')
        batch = pretty_log_batch_size(n_obs)
        for i in range(n_obs):
            self._fill_row(out, **kwargs, row=i)
            if (i+1) % batch == 0:
                logger.debug('processed %d obs', i+1)
        return out

    def many_by_col(self, *, data=None, x_float=None, x_int=None, x_str=None):
        """
        Matrix counterpart of `self.one`.

        Args:
            data: pandas DataFrame.
                If not `None`, then `x_float`, `x_int`, and `x_str` are ignored.
            x_float: numpy matrix if not `None`.
            x_int: likewise.
            x_str: likewise.

        Each row of the inputs represents one observation.
        """
        if data is not None:
            kwargs = self.grind(data)
        else:
            kwargs = {'x_float': x_float, 'x_int': x_int, 'x_str': x_str}
        kwargs, n_obs = self._get_data_kwargs(**kwargs)

        if n_obs * self.n_predictors > 1000000:
            logger.debug('creating zeroed numpy matrix of shape (%d, %d)', n_obs, self.n_predictors)
        out = np.zeros((n_obs, self.n_predictors), order='C')
        self._fill_col(out, **kwargs)
        return out

    def many(self, **kwargs):
        return self.many_by_col(**kwargs)


FEATURES = {
    "float": FloatFeature,
    "int": IntFeature,
    "str": StrFeature
}

# The labels are used in config files to identify the transformers and transplexers.
TRANSFORMERS = {
    "direct": DirectFeature,
    "dummifier": Dummifier,
    "length_checker": LengthChecker,
    "has_value_checker": HasValueChecker,
    "keywords_checker": KeywordsChecker,
    "recency": TimestampRecency,
    "timestamp_2_weekday_hour_dummifier": Timestamp2WeekdayHourDummifier,
    'os': OS,
    'browser': Browser
}


def parse_feature_config(feature_config):
    """
    `feature_config` is a list of feature definitions, as exemplified in 'train.json'.
    """
    features = [
        {
            'name': feature['name'],
            'action': TRANSFORMERS[feature.get('action', 'direct')],
            'kwargs': feature.get('parameters', {})
        }
        for feature in feature_config
        ]
    return features


def make_datamill(features_config, data=None):
    """
    `data` is a pandas DataFrame. Can be `None` if it is not actually needed.
    """
    features = parse_feature_config(features_config)
    transformers = []
    transplexers = []
    for f in features:
        name = f['name']
        cls = f['action']
        kwargs = f['kwargs']
        if issubclass(cls, Transplexer):
            kw = {}
            for k, v in kwargs.items():
                vv = v.split()
                if len(vv) > 1:
                    assert len(vv) == 2
                    kw[k] = FEATURES[vv[0]](vv[1])
                else:
                    kw[k] = v
            tr = cls(**kw)
            transplexers.append(tr)
        else:
            if cls is Dummifier and 'values' not in kwargs:
                values = data[name].copy().fillna('').values.astype(str)
                tr = cls(feature_name=name, values=values, **kwargs)
            else:
                tr = cls(feature_name=name, **kwargs)
            transformers.append(tr)
        #logger.debug(str(tr))

    data_mill = DataMill(transformers, transplexers)
    return data_mill


def get_features(feature_config):
    """
    Args:
        feature_config: the 'features' section of the config JSON.
    """
    float_features = []
    int_features = []
    str_features = []
    features = parse_feature_config(feature_config)
    for f in features:
        name = f['name']
        cls = f['action']
        kwargs = f['kwargs']
        if issubclass(cls, Transplexer):
            for v in kwargs.values():
                vv = v.split()
                if len(vv) > 1:
                    if vv[0] == 'float':
                        float_features.append(vv[1])
                    elif vv[0] == 'int':
                        int_features.append(vv[1])
                    else:
                        str_features.append(vv[1])
        else:
            if issubclass(cls, FloatTransformer):
                float_features.append(name)
            elif issubclass(cls, IntTransformer):
                int_features.append(name)
            else:
                str_features.append(name)

    return float_features, int_features, str_features


def get_config_features(config, sections):
    float_features = []
    int_features = []
    str_features = []

    for sec in sections:
        f, i, s = get_features(config[sec]['features'])
        float_features.extend(f)
        int_features.extend(i)
        str_features.extend(s)

    float_features = sorted(list(set(float_features)))
    int_features = sorted(list(set(int_features)))
    str_features = sorted(list(set(str_features)))

    return float_features, int_features, str_features