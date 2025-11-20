#!/bin/bash

CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

python ${1}.py $2 $3 $4 $5 $6 $7 $8 $9  ${10}  ${11} 

echo "Script ${exp}.py completed"
