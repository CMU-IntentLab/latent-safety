#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("LCRL", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        "gymnasium==0.28.1",
        "tqdm",
        "numpy==1.26.4",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch==2.4.0",
        "numba>=0.51.0",
        "h5py>=2.10.0",  # to match tensorflow's minimal requirements
        "packaging",
        "matplotlib",
        "gym",
        "seaborn",
        "clarabel",
        "ipykernel",
    ]


def get_extras_require() -> str:
    req = {
        "dev": [
            "sphinx",
            "sphinx_rtd_theme",
            "jinja2",
            "sphinxcontrib-bibtex",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0",
            "wandb>=0.12.0",
            "networkx",
            "mypy",
            "pydocstyle",
            "doc8",
            "scipy",
            "pillow",
            "pygame>=2.1.0",  # pettingzoo test cases pistonball
            "pymunk>=6.2.1",  # pettingzoo test cases pistonball
            "nni>=2.3,<3.0",  # expect breaking changes at next major version
            "pytorch_lightning",
            "gym>=0.22.0",
            "shimmy",
            "ffmpeg-python",
            "gymnasium[mujoco]>=0.26.0",
            "seaborn",
        ],
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    }
    if sys.platform == "linux":
        req["dev"].append("envpool>=0.7.0")
    return req


setup(
    name="LCRL",
    version=get_version(),
    description="A Library for Learning Reachability Sets, based on Tianshou (version 0.5.1)",
    url="https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning",
    author="james.jingqi.li",
    author_email="james.jingqi.li@gmail.com",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
