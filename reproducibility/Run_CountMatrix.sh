#!/bin/bash

CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

path=$1
gtf_file=$2
frag_file=$3
featurespace=$4
feature_file=$5
metadata=$6
source=$7
rep=$8
target_label=$9

python 2_CountMatrix.py $path $gtf_file $frag_file $featurespace $feature_file $metadata $source $rep $target_label

echo "Script 2_Run_CountMatrix.py completed"
