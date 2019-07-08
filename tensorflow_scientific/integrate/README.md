Solves the initial value problem for a non-stiff system of first order ODEs:
```
dy/dt = func(y, t), y(t[0]) = y0
```
where y is a Tensor of any shape.
For example:
```
# solve `dy/dt = -y`, corresponding to exponential decay
tf.contrib.integrate.odeint(lambda y, _: -y, 1.0, [0, 1, 2])
=> [1, exp(-1), exp(-2)]
```
Output dtypes and numerical precision are based on the dtypes of the inputs
`y0` and `t`.
Currently, implements 5th order Runge-Kutta with adaptive step size control
and dense output, using the Dormand-Prince method. Similar to the 'dopri5'
method of `scipy.integrate.ode` and MATLAB's `ode45`.
Based on: Shampine, Lawrence F. (1986), "Some Practical Runge-Kutta Formulas",
Mathematics of Computation, American Mathematical Society, 46 (173): 135-150,
doi:10.2307/2008219
