# ADCP Python Toolbox

Python tools for reading, processing, and plotting data from **Nortek Signature ADCPs**. Supports native AD2CP files and serial telemetry, conversion to NetCDF, and visualization of ADCP measurements.

## Installation

Requires Python 3.10 or newer and Git. Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/hverholt/adcp-py-toolbox.git"
```

This installs the `adcp_toolbox` package and its required dependencies.

## Usage

```python
from adcp_toolbox import ad2cp, plot
```

Example workflows are available in `notebooks/`.

## Development

To modify the package locally, clone the repository and install it in editable mode with notebook dependencies:

```bash
git clone https://github.com/hverholt/adcp-py-toolbox.git
cd adcp-py-toolbox
python -m pip install -e ".[notebooks]"
```

## Collaboration

Developed in collaboration with the **University of Iceland** and **Röst Marine Research Center** (https://www.rostrannsoknir.is).

## Maintainer

Developed and maintained by **Sindri Bernholt**  
[sindri@rostrannsoknir.is](mailto:sindri@rostrannsoknir.is)
