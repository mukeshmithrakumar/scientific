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
"""TensorFlow Scientific

TensorFlow Scientific (TFS) is a Python library built on TensorFlow for scientific computing.
TensorFlow Scientific contains modules for integration, ODE solvers and other tasks common in science and engineering.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup
import os

version = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "tensorflow_scientific", "version.py")) as fp:
    exec(fp.read(), version)


project_name = 'tensorflow-scientific'

REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'tensorflow==2.5.0rc0'
]

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name=project_name,
    version=version['__version__'],
    description='Scientific modeling in TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mukesh Mithrakumar',
    author_email='mukesh@mukeshmithrakumar.com',
    url='https://github.com/mukeshmithrakumar/scientific',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    project_urls={
        'Source': 'https://github.com/mukeshmithrakumar/scientific'
    },
    license='Apache 2.0',
    keywords=('tensorflow', 'scientific')
)
