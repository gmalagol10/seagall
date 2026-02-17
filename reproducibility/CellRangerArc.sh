#!/bin/sh

CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

id=$1
ref=$2
fastqgex=$3
fastqatac=$4
mem=$5
cores=$6
output_dir=$7

mkdir -p $output_dir
cd $output_dir

rm -rf library_${id}.csv

echo "Dataset: $id"
echo "Reference: $ref"
echo "Fastq: $fastqgex (GEX) and $fastqatac (ATAC)"
echo "Memory and cores: $mem and $cores"
echo "Output dir $output_dir"

echo "fastqs,sample,library_type" > library_${id}.csv
echo "${fastqgex},${id},Gene Expression" >> library_${id}.csv
echo "${fastqatac},${id},Chromatin Accessibility" >> library_${id}.csv



cellranger-arc count --id=${1}_CellRangerArc \
--reference=$ref \
--libraries=library_${id}.csv \
--localcores=$cores \
--localmem=$mem
