<p align="center"><img width="90%" src="https://raw.githubusercontent.com/mukeshmithrakumar/scientific/master/logo/tfs.png" /></p>

<h1 id="TensorflowScientific" align="center" >Tensorflow Scientific</h1>

<p align="center">
    <a href="https://www.tensorflow.org/beta">
    <img src="https://img.shields.io/badge/Tensorflow-2.0-orange.svg" alt="Tensorflow 2.0">
    </a>
    <a href="https://travis-ci.com/mukeshmithrakumar/scientific.svg?branch=master">
    <img src="https://travis-ci.com/mukeshmithrakumar/scientific.svg?branch=master" alt="Build Status">
    </a>
    <a href="https://pypi.org/project/tensorflow-scientific/">
    <img src="https://badge.fury.io/py/tensorflow-scientific.svg" alt="PyPI Status Badge">
    </a>
    <a href="https://pypi.python.org/pypi/tensorflow-scientific/">
    <img src="https://img.shields.io/pypi/pyversions/tensorflow-scientific.svg" alt="PyPI pyversions">
    </a>
    <a href="https://www.codacy.com/app/mukesh_4/scientific?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mukeshmithrakumar/scientific&amp;utm_campaign=Badge_Grade"><img src="https://api.codacy.com/project/badge/Grade/eb5acb88325245e9b7265da0c4f11db8"/>
    </a>
    <a href='https://coveralls.io/github/mukeshmithrakumar/scientific?branch=master'><img src='https://coveralls.io/repos/github/mukeshmithrakumar/scientific/badge.svg?branch=master' alt='Coverage Status'/>
    </a>
</p>


<h2 align="center">Introduction</h2>

TensorFlow Scientific (TFS) is a Python library built on TensorFlow for scientific computing.
TensorFlow Scientific contains modules for integration, ODE solvers and other tasks common in science and engineering and a sub package on quantum mechanics.


<h2 align="center">Installation</h2>

#### Stable Builds

To install the latest version, run the following:

```
pip install tensorflow-scientific
```

**Note:** [`tensorflow==2.0.0-beta0`](https://www.tensorflow.org/beta) will be installed with the package if you don't have it.

To use TensorFlow Scientific:

```python
import tensorflow as tf
import tensorflow_scientific as tfs
```

#### Installing from Source

**WORK IN PROGRESS**

You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```
git clone https://github.com/mukeshmithrakumar/scientific.git
cd addons

# This script links project with TensorFlow dependency
./configure.sh

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_scientific-*.whl
```


<h2 align="center">Subpackages</h2>
<p align="right"><a href="#TensorflowScientific"><sup>▴ Back to top</sup></a></p>

[tfs.integrate](tensorflow_scientific/integrate/README.md)

- tfs.integrate.odeint
- tfs.integrate.odeint_fixed

[tfs.solvers](tensorflow_scientific/solvers/README.md)

**WORK IN PROGRESS**

[tfs.quantum](tensorflow_scientific/quantum/README.md)

**WORK IN PROGRESS**


<h2 align="center">Examples</h2>
<p align="right"><a href="#TensorflowScientific"><sup>▴ Back to top</sup></a></p>

**WORK IN PROGRESS**


<h2 align="center">Upcoming Releases</h2>
<p align="right"><a href="#TensorflowScientific"><sup>▴ Back to top</sup></a></p>

:fire: 0.3.0 Developer Alpha

- tfs.solvers
- support for linux build
- install via conda
- examples on tfs.integrate
- examples on tfs.solvers

:fire: 0.4.0 Developer Alpha

- tfs.quantum
- examples on tfs.quantum


<h2 align="center">FAQ</h2>
<p align="right"><a href="#TensorflowScientific"><sup>▴ Back to top</sup></a></p>

Q1. How do I contribute?

TF-Scientific is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please
see [contribution guidelines](CONTRIBUTING.md) for a guide on how to
contribute. This project adheres to [TensorFlow's code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
