#!/bin/bash

export OPENBLAS_NUM_THREADS="1"


set -e # if error then crash


python3 process_data.py --nrows=1000000 
