import numpy as np
from numpy.testing import assert_array_equal

from datamill.transformer import (
    Dummifier,
    IntFeature, TimestampRecency
)

def test_Dummifier():
    dum = Dummifier('myfeature', ['ab', 'c', 'ab', 'dd', 'c'])

    y = 'c'
    wanted_1 = np.array([0., 1., 0.])
    got = dum.one(y)
    assert_array_equal(got, wanted_1)

    y = 'd'
    wanted_2 = np.array([0., 0., 0.])
    got = dum.one(y)
    assert_array_equal(got, wanted_2)

    y = np.array(['c', 'd'])
    wanted = np.vstack([wanted_1.reshape(1, -1), wanted_2.reshape(1, -1)])
    got = dum.many(y)
    assert_array_equal(got, wanted)


def test_recency():
    rec = TimestampRecency(IntFeature('click_ts'), IntFeature('ts'), 'click_recency')
    out = np.zeros(3)
    n = rec._fill(out, 3600*3, 3600*10, 1)
    assert n == 1
    assert out[1] == 7
