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

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

from tensorflow_scientific.integrate import odes

# flake8: noqa: E731
class TestOdeInt(test.TestCase):

    def setUp(self):
        """simple defaults (solution is a sin-wave)."""
        super(TestOdeInt, self).setUp()
        matrix = constant_op.constant([[0, 1], [-1, 0]], dtype=dtypes.float64)
        self.func = lambda y, t: math_ops.matmul(matrix, y)
        self.y0 = np.array([[1.0], [0.0]])

    def test_odeint_exp(self):
        """Test odeint by an exponential function: dy / dt = y,  y(0) = 1.0. Its analytical solution is y = exp(t)."""
        func = lambda y, t: y
        y0 = constant_op.constant(1.0, dtype=dtypes.float64)
        t = np.linspace(0.0, 1.0, 11)
        y_solved = odes.odeint(func, y0, t)

        self.assertEqual(y_solved.get_shape(), tensor_shape.TensorShape([11]))
        y_true = np.exp(t)
        self.assertAllClose(y_true, y_solved)

    def test_odeint_complex(self):
        """Test a complex, linear ODE: dy / dt = k * y,  y(0) = 1.0. Its analytical solution is y = exp(k * t)."""
        k = 1j - 0.1
        func = lambda y, t: k * y
        t = np.linspace(0.0, 1.0, 11)
        y_solved = odes.odeint(func, 1.0 + 0.0j, t)

        y_true = np.exp(k * t)
        self.assertAllClose(y_true, y_solved)

    def test_odeint_riccati(self):
        """The Ricatti equation is: dy / dt = (y - t) ** 2 + 1.0,  y(0) = 0.5.
        Its analytical solution is y = 1.0 / (2.0 - t) + t."""
        func = lambda t, y: (y - t)**2 + 1.0
        t = np.linspace(0.0, 1.0, 11)
        y_solved = odes.odeint(func, np.float64(0.5), t)

        y_true = 1.0 / (2.0 - t) + t
        self.assertAllClose(y_true, y_solved)

    def test_odeint_2d_linear(self):
        """Solve the 2D linear differential equation:
        dy1 / dt = 3.0 * y1 + 4.0 * y2,
        dy2 / dt = -4.0 * y1 + 3.0 * y2,
        y1(0) = 0.0,
        y2(0) = 1.0.
        Its analytical solution is
        y1 = sin(4.0 * t) * exp(3.0 * t),
        y2 = cos(4.0 * t) * exp(3.0 * t).
        """
        matrix = constant_op.constant([[3.0, 4.0], [-4.0, 3.0]], dtype=dtypes.float64)
        func = lambda y, t: math_ops.matmul(matrix, y)
        y0 = constant_op.constant([[0.0], [1.0]], dtype=dtypes.float64)
        t = np.linspace(0.0, 1.0, 11)
        y_solved = odes.odeint(func, y0, t)

        y_true = np.zeros((len(t), 2, 1))
        y_true[:, 0, 0] = np.sin(4.0 * t) * np.exp(3.0 * t)
        y_true[:, 1, 0] = np.cos(4.0 * t) * np.exp(3.0 * t)
        self.assertAllClose(y_true, y_solved, atol=1e-5)

    def test_odeint_higher_rank(self):
        func = lambda y, t: y
        y0 = constant_op.constant(1.0, dtype=dtypes.float64)
        t = np.linspace(0.0, 1.0, 11)
        for shape in [(), (1,), (1, 1)]:
            expected_shape = (len(t),) + shape
            y_solved = odes.odeint(func, array_ops.reshape(y0, shape), t)
            self.assertEqual(y_solved.get_shape(), tensor_shape.TensorShape(expected_shape))
            self.assertEqual(y_solved.shape, expected_shape)

    def test_odeint_all_dtypes(self):
        func = lambda y, t: y
        t = np.linspace(0.0, 1.0, 11)
        for y0_dtype in [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]:
            for t_dtype in [dtypes.float32, dtypes.float64]:
                y0 = math_ops.cast(1.0, y0_dtype)
                y_solved = odes.odeint(func, y0, math_ops.cast(t, t_dtype))
                expected = np.asarray(np.exp(t))
                self.assertAllClose(y_solved, expected, rtol=1e-5)
                self.assertEqual(dtypes.as_dtype(y_solved.dtype), y0_dtype)

    def test_odeint_required_dtypes(self):
        with self.assertRaisesRegexp(TypeError, '`y0` must have a floating point'):
            odes.odeint(self.func, math_ops.cast(self.y0, dtypes.int32), [0, 1])

        with self.assertRaisesRegexp(TypeError, '`t` must have a floating point'):
            odes.odeint(self.func, self.y0, math_ops.cast([0, 1], dtypes.int32))

    def test_odeint_runtime_errors(self):
        with self.assertRaisesRegexp(ValueError, 'cannot supply `options` without'):
            odes.odeint(self.func, self.y0, [0, 1], options={'first_step': 1.0})

        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError, 'max_num_steps'):
            odes.odeint(self.func, self.y0, [0, 1], method='dopri5', options={'max_num_steps': 0})

        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError, 'monotonic increasing'):
            odes.odeint(self.func, self.y0, [1, 0])

    def test_odeint_different_times(self):
        times0 = np.linspace(0, 10, num=11, dtype=float)
        times1 = np.linspace(0, 10, num=101, dtype=float)

        y_solved_0, info_0 = odes.odeint(self.func, self.y0, times0, full_output=True)
        y_solved_1, info_1 = odes.odeint(self.func, self.y0, times1, full_output=True)

        self.assertAllClose(y_solved_0, y_solved_1[::10])
        self.assertAllEqual(info_0['num_func_evals'], info_1['num_func_evals'])
        self.assertAllEqual(info_0['integrate_points'], info_1['integrate_points'])
        self.assertAllEqual(info_0['error_ratio'], info_1['error_ratio'])

    def test_odeint_5th_order_accuracy(self):
        t = [0, 20]
        kwargs = dict(full_output=True, method='dopri5', options=dict(max_num_steps=2000))
        _, info_0 = odes.odeint(self.func, self.y0, t, rtol=0, atol=1e-6, **kwargs)
        _, info_1 = odes.odeint(self.func, self.y0, t, rtol=0, atol=1e-9, **kwargs)

        self.assertAllClose(array_ops.size(info_0['integrate_points'], out_type=dtypes.float32) * 1000**0.2,
                            array_ops.size(info_1['integrate_points'], out_type=dtypes.float32),
                            rtol=0.01)


class TestStepSize(test.TestCase):

    def test_error_ratio_one(self):
        new_step = odes._optimal_step_size(last_step=constant_op.constant(1.0), error_ratio=constant_op.constant(1.0))
        self.assertAllClose(new_step, 0.9)

    def test_ifactor(self):
        new_step = odes._optimal_step_size(last_step=constant_op.constant(1.0), error_ratio=constant_op.constant(0.0))
        self.assertAllClose(new_step, 10.0)

    def test_dfactor(self):
        new_step = odes._optimal_step_size(last_step=constant_op.constant(1.0), error_ratio=constant_op.constant(1e6))
        self.assertAllClose(new_step, 0.2)


class TestInterpolation(test.TestCase):

    def test_5th_order_polynomial(self):
        f = lambda x: x**4 + x**3 - 2 * x**2 + 4 * x + 5
        f_prime = lambda x: 4 * x**3 + 3 * x**2 - 4 * x + 4
        coeffs = odes._interp_fit(f(0.0), f(10.0), f(5.0), f_prime(0.0), f_prime(10.0), 10.0)
        times = np.linspace(0, 10, dtype=np.float32)
        y_fit = array_ops.stack([odes._interp_evaluate(coeffs, 0.0, 10.0, t) for t in times])
        y_expected = f(times)

        self.assertAllClose(y_expected, y_fit)
        with self.assertRaises(errors_impl.InvalidArgumentError):
            odes._interp_evaluate(coeffs, 0.0, 10.0, 100.0)


class TestOdeIntFixed(test.TestCase):

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

    def test_midpoint(self):
        self._test_everything('midpoint')

    def test_rk4(self):
        self._test_everything('rk4')

    def test_size_exceptions(self):
        times = np.linspace(0., 2., 100)
        dt = np.ones(99) * 0.01
        dt_wrong_length = np.asarray([0.01, 0.001, 0.1, 0.03])
        dt_wrong_dim = np.expand_dims(np.linspace(0., 2., 99), axis=0)
        times_wrong_dim = np.expand_dims(np.linspace(0., 2., 100), axis=0)

        with self.assertRaises(errors_impl.InvalidArgumentError):
            self._test_integrate_gaussian('midpoint', times, dt_wrong_length)

        with self.assertRaises(ValueError):
            self._test_integrate_gaussian('midpoint', times, dt_wrong_dim)

        with self.assertRaises(errors_impl.InvalidArgumentError):
            self._test_integrate_gaussian('midpoint', times_wrong_dim, dt)


if __name__ == '__main__':
    test.main()
