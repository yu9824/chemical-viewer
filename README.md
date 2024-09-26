# matplotlib-interactive

[![CI](https://github.com/yu9824/chemical-viewer/actions/workflows/CI.yaml/badge.svg)](https://github.com/yu9824/chemical-viewer/actions/workflows/CI.yaml)
[![docs](https://github.com/yu9824/chemical-viewer/actions/workflows/docs.yaml/badge.svg)](https://github.com/yu9824/chemical-viewer/actions/workflows/docs.yaml)
<!--
[![python_badge](https://img.shields.io/pypi/pyversions/chemical-viewer)](https://pypi.org/project/chemical-viewer/)
[![license_badge](https://img.shields.io/pypi/l/chemical-viewer)](https://pypi.org/project/chemical-viewer/)
[![PyPI version](https://badge.fury.io/py/chemical-viewer.svg)](https://pypi.org/project/chemical-viewer/)
[![Downloads](https://static.pepy.tech/badge/chemical-viewer)](https://pepy.tech/project/chemical-viewer)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/chemical-viewer.svg)](https://anaconda.org/conda-forge/chemical-viewer)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/chemical-viewer.svg)](https://anaconda.org/conda-forge/chemical-viewer)
-->

## How to use

### CLI

After `pip install`

```bash
chemical-viewer ./data/data.csv

```

Please check help (`chemical-viewer --help`)

### Python API

```python
from chemical_viewer import InteractiveViewer
from rdkit import Chem

viewer = InteractiveViewer()
viewer.scatter([1], [1], mols=[Chem.MolFromSmiles("CC")], texts=["test"])

```
