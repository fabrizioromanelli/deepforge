#!/bin/bash

python -m build
twine upload -r testpypi dist/*
sleep 5 # wait for the project to be "digested" by the PyPI system
pip install -i https://test.pypi.org/simple/ deep-model
