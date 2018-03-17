#!/bin/bash
set -e
if [ ! -d 'venv' ]; then
  virtualenv venv
  source venv/bin/activate
  pydir="$(which python)"
  if [[ "$pydir" != *"/venv/bin/python" ]]
  then
    echo "Something went wrong. Please use the manual method."
    exit
  fi
  pip install pillow scipy tensorflow==0.12.0
  deactivate
  clear
fi
source venv/bin/activate
./enhancenet.py
deactivate
set +e
