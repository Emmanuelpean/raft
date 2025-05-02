<p align="center">
  <img src="https://github.com/Emmanuelpean/raft/blob/main/resources/medias/logo.svg" alt="Raft" width="150">
</p>

<h1 align="center">RAFT</h1>
<h2 align="center">Universal Data File Plotter</h2>

<div align="center">

  [![Passing](https://github.com/Emmanuelpean/raft/actions/workflows/test.yml/badge.svg?branch=type-hints&event=push)](https://github.com/Emmanuelpean/raft/actions/workflows/test.yml)
  [![Tests Status](./reports/tests/tests-badge.svg?dummy=8484744)](https://emmanuelpean.github.io/raft/reports/tests/report.html?sort=result)
  [![Coverage Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](https://emmanuelpean.github.io/raft/reports/coverage/htmlcov/index.html)
  [![Last Commit](https://img.shields.io/github/last-commit/emmanuelpean/raft/type-hints)](https://github.com/emmanuelpean/raft/commits/type-hints)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

*Raft* is a user-friendly web app designed to give you a quick look at your data without needing to write a single line 
of code or spend time reformatting files. It supports a wide range of data file formats and takes care of the behind-
the-scenes work of reading and processing the data, so you can jump straight to visualising and exploring your results.

The idea came from a simple frustration: too often, valuable time is lost just trying to open and make sense of a data 
file. This app removes that barrier. Whether you are checking raw instrument output, exploring simulation results, or 
comparing experimental datasets, it lets you plot and inspect the data in just a few clicks. It is especially handy in 
fast-paced research environments where quick decisions depend on a fast understanding of the data.

## Installation

Create and activate a virtual environment, activate it, and run:
```console
$ pip install -e .[dev]
```

## Usage
To run the app locally, run:
```console
$ cd app
$ streamlit run ./main.py
```

To profile the app, run:
```console
$ streamlit run ./main.py -- --profile=true
```
