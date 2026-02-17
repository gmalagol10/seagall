#!/bin/bash

CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

module load ngs/MACS2/2.1.2

frag_file=$1
name=$2
outdir=$3
gsize=$4

echo "Fragment file: $frag_file"
echo "Name: $name "
echo "Out dir: $outdir"
echo "Genome size: $gsize"

echo "Peak calling $(date +"%T"): Run MACS2"
macs2 callpeak \
--nomodel \
--keep-dup all \
--extsize 200 \
--shift -100 \
-f BED \
--gsize $gsize \
--name $name \
--outdir $outdir \
-t  $frag_file
echo "Peak calling $(date +"%T"): Done MACS2"
