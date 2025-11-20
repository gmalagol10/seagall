import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import pandas as pd
import snapatac2
import numpy as np
import polars as pl
import sys
import time
import os
from pathlib import Path


warnings.filterwarnings("ignore")
print( time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI script started", flush=True)

path = sys.argv[1]
name = sys.argv[2]
matrix = sys.argv[3]
label = sys.argv[4]

names = ["Script -->", "Path -->", "Name -->", "Matrix -->", "Label -->"]
for n,arg in zip(names,sys.argv):
	print(n, arg, flush=True)


Path(f"{path}/SnapATAC2").mkdir(parents=True, exist_ok=True)
snp2_path = f"{path}/SnapATAC2/{name}"
	
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Creating dataset", flush=True)					
adata = sc.read(matrix)
adata=adata[adata.obs[label].dropna().index]

diz={}
for ct in sorted(set(adata.obs[label])):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Testing {ct}", flush=True)					
	g1=adata.obs[label] == ct
	g2=adata.obs[label] != ct
	diff_peaks = snapatac2.tl.diff_test(adata, cell_group1=g1, cell_group2=g2, direction="both")
	diff_peaks=diff_peaks.to_pandas()
	diff_peaks=diff_peaks[(diff_peaks["log2(fold_change)"]>0.5) ^ (diff_peaks["log2(fold_change)"]<0.5)]
	diff_peaks=diff_peaks[(diff_peaks["adjusted p-value"]<0.05)]
	diz[ct]=diff_peaks
	
fts, lfc, pvs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for ct in diz.keys():
	fts[ct]=diz[ct]["feature name"]
	lfc[ct]=diz[ct]["log2(fold_change)"]
	pvs[ct]=diz[ct]["adjusted p-value"]

fts.to_csv(f"{snp2_path}_SnapATAC2_Features.tsv.gz", sep="\t", compression="gzip")
lfc.to_csv(f"{snp2_path}_SnapATAC2_FeaturesLFC.tsv.gz", sep="\t", compression="gzip")
pvs.to_csv(f"{snp2_path}_SnapATAC2_FeaturesPvsAdj.tsv.gz", sep="\t", compression="gzip")
