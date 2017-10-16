"""
This module, esp `MetaModel`, shows the intended usage of `DataMill`.
"""

import textwrap

import numpy as np
from sklearn.utils import check_array, check_X_y, check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator


class StratifiedModel(BaseEstimator):
    def __init__(self, model_cls, **kwargs):
        self.model_class = model_cls
        self.model_kwargs = kwargs.copy()

    def fit(self, X, y, strata):
        """
        Args:
            X: numpy matrix of predictors.
            y: numpy array.
            strata: numpy array of strata. It's sensible to envision the need of a
                multi-column matrix strata, but there is no such need for now.
        """
        X, y = check_X_y(X, y,
                         ensure_2d=True, copy=False,
                         y_numeric=True, multi_output=False)
        strata = check_array(strata, ensure_2d=False)
        check_consistent_length(y, strata)

        uniq = np.unique(strata)
        models = {}
        residues = 0.0
        for key in uniq:
            model = self.model_class(**self.model_kwargs)
            idx = np.where(strata == key)[0]
            model.fit(X[idx, :], y[idx])
            models[key] = model
            residues += model._residues
        self.models_ = models
        self._residues = residues
        self._n_obs = X.shape[0]

        return self


class StratifiedRegressor(StratifiedModel):
    def _decision_function(self, X, strata):
        check_is_fitted(self, 'models_')
        strata = check_array(strata, ensure_2d=False)
        check_consistent_length(X, strata)
        uniq = np.unique(strata)
        yhat = np.zeros(X.shape[0])
        for key in uniq:
            idx = np.where(strata == key)[0]
            model = self.models_.get(key, None)
            if model is None:
                raise Exception('model for stratum %s is not found' % key)
            yhat[idx] = model._decision_function(X[idx, :])
        return yhat

    def predict(self, X, strata):
        """
        Args:
            X: numpy matrix.
        """
        return self._decision_function(X, strata)

    def predict_one(self, x, stratum):
        """
        Make prediction on a single data point.

        Args:
            x: 1-d numpy array of predictors.
        """
        model = self.models_.get(stratum, None)
        if model is None:
            raise Exception('model for stratum %s is not found' % stratum)
        if hasattr(model, 'predict_one'):
            return model.predict_one(x)
        else:
            return model.predict(x.reshape(-1, 1))

    # def __repr__(self):
    #     if self.model_kwargs:
    #         args = ', ' + ', '.join(k + '=' + str(v) for k, v in self.model_kwargs.items())
    #     else:
    #         args = ''
    # return '{}({}{})'.format(self.__class__.__name__,
    # self.model_class.__name__, args)


class StratifiedClassifier(StratifiedModel):
    def predict_proba(self, X, strata):
        raise NotImplementedError

    def predict_proba_one(self, x, stratum):
        raise NotImplementedError


class MetaModel(BaseEstimator):
    def __init__(self, model, datamill):
        """
        `datamill` is either a `DataMill` instance or a `DoubleDataMill` instance.
        The user needs to know which is the case, and call the methods accordingly
        with appropriate data input parameters.
        """
        self.model = model
        self.datamill = datamill

    def fit(self, x, y, **kwargs):
        XX = self.datamill.many(**x)
        self.model.fit(XX, y, **kwargs)
        return self

    def __repr__(self):
        return '{}(model={}, datamill={})'.format(
            self.__class__.__name__,
            repr(self.model),
            repr(self.datamill)
        )

    def __str__(self):
        return '{}:\n  model:\n{}\n  datamill:\n{}'.format(
            self.__class__.__name__,
            textwrap.indent(str(self.model), '    '),
            textwrap.indent(str(self.datamill), '    '))


class MetaRegressor(MetaModel):
    def predict(self, x, **kwargs):
        XX = self.datamill.many(**x)
        return self.model.predict(XX, **kwargs)

    def predict_one(self, x, **kwargs):
        xx = self.datamill.one(**x)
        if hasattr(self.model, 'predict_one'):
            return self.model.predict_one(xx, **kwargs)
        else:
            return self.model.predict(np.array([xx]), **kwargs)


class MetaClassifier(MetaModel):
    def predict_proba(self, x, **kwargs):
        XX = self.datamill.many(**x)
        return self.model.predict_proba(XX, **kwargs)

    def predict_proba_one(self, x, **kwargs):
        xx = self.datamill.one(**x)
        if hasattr(self.model, 'predict_proba_one'):
            return self.model.predict_proba_one(xx, **kwargs)
        else:
            return self.model.predict_proba(np.array([xx]), **kwargs)[0]
