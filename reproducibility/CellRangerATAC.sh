#!/bin/sh

CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

id=$1
ref=$2
fastq=$3
mem=$4
cores=$5
working_dir=$6

mkdir -p $working_dir
cd $working_dir

echo "Dataset: $1"
cellranger-atac count --id=${1}_CellRanger \
--reference=$ref \
--fastqs=$fastq \
--sample=$1 \
--localmem=$mem \
--localcores=$cores
