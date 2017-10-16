import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from faker import Faker

from datamill.transformer import (
    DirectFeature, Dummifier,
    IntFeature, TimestampRecency)
from datamill.datamill import DataMill


fake = Faker()

direct = DirectFeature('direct')

dumm_values = [fake.name() for _ in range(10)]
dummif = Dummifier('dumm', dumm_values)

recency = TimestampRecency(IntFeature('ts1'), IntFeature('ts2'))

dm = DataMill([direct, dummif], [recency])


def test_init():
    print('')
    print('repr:')
    print(repr(dm))
    print('str:')
    print(str(dm))
    print('')
    print('float_feature_names:', dm.float_feature_names)
    print('int_feature_names:', dm.int_feature_names)
    print('str_feature_names:', dm.str_feature_names)


def make_one():
    x_float = [3.8]
    x_int = [3600, 7200]
    x_str = [dumm_values[2]]

    expected = np.zeros(1 + len(dummif.values) + 1)
    expected[0] = x_float[0]
    expected[1 + dummif.values[x_str[0]]] = 1.0
    ts1_idx = dm.int_feature_names.index(recency.feature_names[0])
    ts2_idx = 1 - ts1_idx
    expected[-1] = (x_int[ts2_idx] - x_int[ts1_idx]) / 3600.

    return x_float, x_int, x_str, expected


def make_many(n):
    x_float, x_int, x_str, expected = make_one()
    ts1_idx = dm.int_feature_names.index(recency.feature_names[0])
    ts2_idx = 1 - ts1_idx
    data = pd.DataFrame(
        {direct.feature_name: x_float * n,
         recency.feature_names[0]: [x_int[ts1_idx]] * n,
         recency.feature_names[1]: [x_int[ts2_idx]] * n,
         dummif.feature_name: x_str * n
        }
    )
    expected = np.tile(expected, [n, 1])
    return data, expected


def test_one():
    x_float, x_int, x_str, expected = make_one()

    z = dm.one(x_float=x_float, x_int=x_int, x_str=x_str)
    assert_array_equal(z, expected)


def test_many():
    data, expected = make_many(3)
    z1 = dm.many_by_row(data = data)
    assert_array_equal(z1, expected)

    z2 = dm.many_by_col(data = data)
    assert_array_equal(z2, expected)


def do_row(data):
    z = dm.many_by_row(data = data)
    return z


def do_col(data):
    z = dm.many_by_col(data = data)
    return z


def test_many_large():
    data, expected = make_many(10000)

    z1 = do_row(data)
    assert_array_equal(z1, expected)

    z2 = do_col(data)
    assert_array_equal(z2, expected)
