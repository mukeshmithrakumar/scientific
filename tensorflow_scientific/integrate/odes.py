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
"""ODE solvers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha beta c_sol c_mid c_error')

# Parameters from Shampine (1986), section 4.
_DORMAND_PRINCE_TABLEAU = _ButcherTableau(
    alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    beta=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_mid=[
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
    ],
    c_error=[
        1951 / 21600 - 35 / 384,
        0,
        22642 / 50085 - 500 / 1113,
        451 / 720 - 125 / 192,
        -12231 / 42400 - -2187 / 6784,
        649 / 6300 - 11 / 84,
        1 / 60,
    ])


def _assert_increasing(t):
    assert_increasing = control_flow_ops.Assert(
        math_ops.reduce_all(t[1:] > t[:-1]), ['`t` must be monotonic increasing'])
    return ops.control_dependencies([assert_increasing])


def _check_input_sizes(t, dt):
    if len(t.get_shape().as_list()) > 1:
        raise ValueError('`t` must be a 1D tensor')

    if len(dt.get_shape().as_list()) > 1:
        raise ValueError('`dt` must be a 1D tensor')

    if t.get_shape()[0] != dt.get_shape()[0] + 1:
        raise ValueError('`t` and `dt` have incompatible lengths, must be N and N-1')


def _check_input_types(y0, t, dt=None):
    if not(y0.dtype.is_floating or y0.dtype.is_complex):
        raise TypeError('`y0` must have a floating point or complex floating point dtype')

    if not(t.dtype.is_floating):
        raise TypeError('`t` must have a floating point dtype')

    if dt is not None and not dt.dtype.is_floating:
        raise TypeError('`dt` must have a floating point dtype')


# flake8: noqa: F831
@six.add_metaclass(abc.ABCMeta)
class _FixedGridIntegrator(object):
    """Base class for fixed-grid ODE integrators."""

    def integrate(self, evol_func, y0, time_grid, dt_grid, steps_on_intervals):
        """Returns integrated values of differential equations on the `time grid`.
        Numerically integrates differential equation defined via time derivative evaluator `evol_func` using
        fixed time steps specified in dt_grid.

        Args:
            evol_func (callable)        : evaluates time derivative of y at a given time.
            y0 (float)                  : N-D Tensor holds initial values of the solution
            time_grid (float)           : 1-D Tensor holding the time points at which the solution will be recorded
            dt_grid (float)             : 1-D Tensor holds fixed time steps to be used on time_grid intervals.
                                          Must have one less element than that of the time_grid
            steps_on_intervals (int)    : 1-D Tensor, must have the same size as dt_grid. Specifies number of
                                          steps needed for every interval.
                                          Assumes steps_on_intervals*dt_grid == time_intervals
        Returns:
            (N+1)-D tensor, where the first dimension corresponds to different time points.
            Contains the solved value of y for each desired time point in `t`, with the initial value `y0` being the first element along the first dimension.
        """

        iteration_func = self._make_iteration_func(evol_func, dt_grid)
        integrate_interval = self._make_interval_integrator(iteration_func, steps_on_intervals)

        num_times = array_ops.size(time_grid)
        current_time = time_grid[0]
        solution_array = tensor_array_ops.TensorArray(y0.dtype, num_times)
        solution_array = solution_array.write(0, y0)

        solution_array, _, _, _ = control_flow_ops.while_loop(
            lambda _a, _b, _c, i: i < num_times,
            integrate_interval,
            (solution_array, y0, current_time, 1)
        )

        solution_array = solution_array.stack()
        solution_array.set_shape(time_grid.get_shape().concatenate(y0.get_shape()))
        return solution_array

    def _make_iteration_func(self, evol_func, dt_grid):
        """Returns a function that builds operations of a single time step."""

        def iteration_func(y, t, dt_step, interval_step):
            """Performs a single time step advance."""

            dt = dt_grid[interval_step - 1]
            dy = self._step_function(evol_func, t, dt, y)
            dy = math_ops.cast(dy, dtype=y.dtype)
            return y + dy, t + dt, dt_step + 1, interval_step
        return iteration_func

    def _make_interval_integrator(self, iteration_func, interval_sizes):
        """Returns a function that builds operations for interval integration."""

        def integrate_interval(solution_array, y, t, interval_num):
            """Integrates y with fixed time step on interval `interval_num`."""

            y, t, _, _ = control_flow_ops.while_loop(
                lambda _a, _b, j, interval_num: j < interval_sizes[interval_num - 1],
                iteration_func,
                (y, t, 0, interval_num)
            )

            return solution_array.write(interval_num, y), y, t, interval_num + 1
        return integrate_interval

    @abc.abstractmethod
    def _step_function(self, evol_func, t, dt, y):
        pass


class _MidpointFixedGridIntegrator(_FixedGridIntegrator):
    """Fixed grid integrator implementing midpoint method. We will be using the explicit midpoint method.
    The local error at each step of the midpoint method is of order O(h^3), giving a global error of  order O(h^2).

    Args:
        _FixedGridIntegrator: Base class for fixed-grid ODE integrators.
    """

    def _step_function(self, evol_func, t, dt, y):
        dt_cast = math_ops.cast(dt, y.dtype)
        # yn1 = yn + h * f((tn + h)/2, (yn + f(tn, yn) * h/2))
        return dt_cast * evol_func(y + evol_func(y, t) * dt_cast / 2, t + dt / 2)


class _RK4FixedGridIntegrator(_FixedGridIntegrator):
    """Fixed grid integrator implementing RK4 method.
    This scheme is the most widely known member of the Runge-Kutta family and called "the Runge–Kutta method".
    The RK4 method is a fourth-order method, meaning that the local truncation error is on the order of O(h^5),
    while the total accumulated error is on the order of O(h^{4}).

    Args:
        _FixedGridIntegrator: Base class for fixed-grid ODE integrators.
    """

    def _step_function(self, evol_func, t, dt, y):
        k1 = evol_func(y, t)
        half_step = t + dt / 2
        dt_cast = math_ops.cast(dt, y.dtype)

        k2 = evol_func(y + dt_cast * k1 / 2, half_step)
        k3 = evol_func(y + dt_cast * k2 / 2, half_step)
        k4 = evol_func(y + dt_cast * k3, t + dt)
        return math_ops.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)


def odeint_fixed(func, y0, t, dt=None, method='rk4', name=None):
    """ODE integration on a fixed grid (with no step size control).
        Useful in certain scenarios to avoid the overhead of adaptive step size
        control, e.g. when differentiation of the integration result is desired and/or
        the time grid is known a priori to be sufficient.

    Args:
        func (function)             : Function that maps a Tensor holding the state `y` and a scalar Tensor
                                      `t` into a Tensor of state derivatives with respect to time.
        y0 (float or complex)       : N-D Tensor giving starting value of `y` at time point `t[0]`.
        t (float)                   : 1-D Tensor holding a sequence of time points for which to solve for and each time
                                      must be larger than the previous time. May have any floating point dtype.
        dt (float, optional)        : 0-D or 1-D Tensor providing time step suggestion to be used on time
                                      integration intervals in `t`. 1-D Tensor should provide values
                                      for all intervals, must have 1 less element than that of `t`.
                                      If given a 0-D Tensor, the value is interpreted as time step suggestion
                                      same for all intervals. If passed None, then time step is set to be the
                                      t[1:] - t[:-1]. Defaults to None. The actual step size is obtained by
                                      insuring an integer number of steps per interval, potentially reducing the
                                      time step. Defaults to None.
        method (str, optional)      :  One of 'midpoint' or 'rk4'. Defaults to 'rk4'.
        name (str, optional)        :  Optional name for the resulting operation. Defaults to None.

    Returns:
        y (Tensor)                  : (N+1)-D tensor, where the first dimension corresponds to different
                                      time points. Contains the solved value of y for each desired time point in
                                      `t`, with the initial value `y0` being the first element along the first
                                      dimension.
    Raises:
        ValueError                  : Upon caller errors.
    """

    with ops.name_scope(name, 'odeint_fixed', [y0, t, dt]):
        t = ops.convert_to_tensor(t, dtype=dtypes.float64, name='t')
        y0 = ops.convert_to_tensor(y0, name='y0')

        intervals = t[1:] - t[:-1]
        if dt is None:
            dt = intervals
        dt = ops.convert_to_tensor(dt, preferred_dtype=dtypes.float64, name='dt')

        steps_on_intervals = math_ops.ceil(intervals / dt)
        dt = intervals / steps_on_intervals
        steps_on_intervals = math_ops.cast(steps_on_intervals, dtype=dtypes.int32)

        _check_input_types(y0, t, dt)
        _check_input_sizes(t, dt)

        with _assert_increasing(t):
            with ops.name_scope(method):
                if method == 'midpoint':
                    return _MidpointFixedGridIntegrator().integrate(func, y0, t, dt, steps_on_intervals)
                elif method == 'rk4':
                    return _RK4FixedGridIntegrator().integrate(func, y0, t, dt, steps_on_intervals)
                else:
                    raise ValueError('method not supported: {!s}'.format(method))
