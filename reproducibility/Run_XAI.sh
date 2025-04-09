#!/bin/bash

CONDA_ENV=gnn
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

path=$1
name=$2
matrix=$3
label=$4
graph=$5
hpo=$6

python 3_XAI.py $path $name $matrix $label $graph $hpo

echo "Script 3_XAI.py completed"
