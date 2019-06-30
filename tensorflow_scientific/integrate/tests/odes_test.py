# Copyright 2019, Mukesh Mithrakumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from tensorflow_scientific.integrate import odes
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import errors_impl
# from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class OdeIntFixedTest(test.TestCase):

    def _test_integrate_sine(self, method, t, dt=None):

        def evol_func(y, t):
            del t
            return array_ops.stack([y[1], -y[0]])

        y0 = [0., 1.]
        y_grid = odes.odeint_fixed(evol_func, y0, t, dt, method=method)
        np.testing.assert_allclose(y_grid[:, 0], np.sin(t), rtol=1e-2, atol=1e-2)

    def _test_integrate_gaussian(self, method, t, dt=None):

        def evol_func(y, t):
            return -math_ops.cast(t, dtype=y.dtype) * y[0]

        y0 = [1.]
        y_grid = odes.odeint_fixed(evol_func, y0, t, dt, method=method)
        np.testing.assert_allclose(y_grid[:, 0], np.exp(-t**2 / 2), rtol=1e-2, atol=1e-2)

    def _test_integrate_sine_all(self, method):
        uniform_time_grid = np.linspace(0., 10., 200)
        non_uniform_time_grid = np.asarray([0.0, 0.4, 4.7, 5.2, 7.0])
        uniform_dt = 0.02
        non_uniform_dt = np.asarray([0.01, 0.001, 0.05, 0.03])

        self._test_integrate_sine(method, uniform_time_grid)
        self._test_integrate_sine(method, non_uniform_time_grid, uniform_dt)
        self._test_integrate_sine(method, non_uniform_time_grid, non_uniform_dt)

    def _test_integrate_gaussian_all(self, method):
        uniform_time_grid = np.linspace(0., 2., 100)
        non_uniform_time_grid = np.asarray([0.0, 0.1, 0.7, 1.2, 2.0])
        uniform_dt = 0.01
        non_uniform_dt = np.asarray([0.01, 0.001, 0.1, 0.03])

        self._test_integrate_gaussian(method, uniform_time_grid)
        self._test_integrate_gaussian(method, non_uniform_time_grid, uniform_dt)
        self._test_integrate_gaussian(method, non_uniform_time_grid, non_uniform_dt)

    def _test_everything(self, method):
        self._test_integrate_sine_all(method)
        self._test_integrate_gaussian_all(method)

    # TODO: Skipping the next two functions because of the following error
    @pytest.mark.skip(reason="np assert_allclose with ufunc isfinite not supported for the input types")
    def test_midpoint(self):
        self._test_everything('midpoint')

    @pytest.mark.skip(reason="np assert_allclose with ufunc isfinite not supported for the input types")
    def test_rk4(self):
        self._test_everything('rk4')

    def test_dt_size_exceptions(self):
        times = np.linspace(0., 2., 100)
        dt = np.ones(99) * 0.01
        dt_wrong_length = np.asarray([0.01, 0.001, 0.1, 0.03])
        dt_wrong_dim = np.expand_dims(np.linspace(0., 2., 99), axis=0)
        times_wrong_dim = np.expand_dims(np.linspace(0., 2., 100), axis=0)

        with self.assertRaises(ValueError):
            self._test_integrate_gaussian('midpoint', times, dt_wrong_length)

        with self.assertRaises(ValueError):
            self._test_integrate_gaussian('midpoint', times, dt_wrong_dim)

        with self.assertRaises(ValueError):
            self._test_integrate_gaussian('midpoint', times_wrong_dim, dt)


if __name__ == '__main__':
    test.main()
