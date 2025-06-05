#!/bin/bash

CONDA_ENV=gnn
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

aes=$1
ds=$2
fs=$3

echo "Aes: $aes"
echo "Dataset: $ds"
echo "Feature space: $fs" 


echo "Master Rob $(date +"%T")"
cat $aes | parallel --gnu -j 4 "./Dropout_Slave.sh {} $ds $fs"
