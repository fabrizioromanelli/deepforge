#!/bin/bash

# To install the library locally in editable mode, run:
# python -m pip install -e .

rm -rf dist/*

if [ $1 = 'test' ]
then
  echo 'Building for test environment'
  python -m build
  twine upload -r testpypi dist/*
  sleep 5 # wait for the project to be "digested" by the PyPI system
  pip install -i https://test.pypi.org/simple/ deepforge
elif [ $1 = 'prod' ]
then
  echo 'Building for production environment'
  python -m build
  twine upload -r pypi dist/*
  sleep 5 # wait for the project to be "digested" by the PyPI system
  pip install deepforge
else
  echo 'Usage: ./build.sh [test/prod]'
fi