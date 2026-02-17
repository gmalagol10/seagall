import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

import time
import sys
import scipy
import torch
import os

from pathlib import Path

import ML_utils as mlu

print(f"Robustness Script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

AE = sys.argv[1]
dataset = sys.argv[2]
featurespace = sys.argv[3]

names = ["Script", "AE", "Data set", "Feature space"]
for n,arg in zip(names,sys.argv):
	print(n,arg, flush=True)
	
path = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/Dropout"
matrix = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/CM/{dataset}_{featurespace}_Dropout.h5ad"
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Loading {matrix}", flush=True)

adata = sc.read_h5ad(matrix)
adata = adata[adata.obs[adata.obs["CellType"].isin(list(adata.obs.value_counts("CellType")[adata.obs.value_counts("CellType")>50].index))].index]
y = np.array(adata.obs.target).astype(int)
Path(f"{path}/{AE}").mkdir(parents=True, exist_ok=True)
for run in range(0, 10):
	for d in np.linspace(0, 50, 6).astype(int):

		M = adata.layers[f"X_{str(d)}_{str(run)}"].copy()

		print(run, d, adata.X.getnnz()/(adata.shape[0]*adata.shape[1]), M.getnnz()/(adata.shape[0]*adata.shape[1]), flush=True)

		Path(f"{path}/{AE}").mkdir(parents=True, exist_ok=True)
		if os.path.isfile(f"{path}/{AE}/{dataset}_{featurespace}_{AE}_{str(d)}_{str(run)}.pth") == False:
			out = mlu.ApplyAE(M=M, representation=AE, y=y, model_name=f"{path}/{AE}/{dataset}_{featurespace}_{AE}_{str(d)}_{str(run)}")
			del out
		else:
			print(f"{path}/{AE}/{dataset}_{featurespace}_{AE}_{str(d)}_{str(run)}.pth already done", flush=True)
