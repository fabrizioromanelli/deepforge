#!/bin/bash

python -m build
twine upload -r testpypi dist/*
pip install -i https://test.pypi.org/simple/ deep-model