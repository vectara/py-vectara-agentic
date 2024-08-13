#!/bin/bash
set -e

# Clean previous builds
rm -rf dist/

# Build the package
python setup.py sdist bdist_wheel

# Upload using twine
twine upload dist/*
