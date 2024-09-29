#!/bin/bash
set -e

# Clean previous builds
rm -rf dist/

# Build the package
python setup.py sdist bdist_wheel

# Run tests
pytest

# Check installation
pip install dist/*.whl

# Lint the package
flake8 --ignore=E501 .

# Upload using twine
twine upload dist/* --username __token__ --password $PYPI_TOKEN
