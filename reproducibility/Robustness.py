import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

import time
import sys
import scipy
import torch

from pathlib import Path

import ML_utils as mlu

print(f"Robustness Script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

dataset = sys.argv[2]
featurespace = sys.argv[3]

names = ["Script","Dataset", "Features space", "AE"]
for n,arg in zip(names,sys.argv):
	print(n,arg, flush=True)
	
path = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/Dropout"
matrix = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/CM/{dataset}_{featurespace}_Dropout.h5ad"
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Loading {matrix}", flush=True)					 	
adata = sc.read_h5ad(matrix)
adata = adata[adata.obs[adata.obs["CellType"].isin(list(adata.obs.value_counts("CellType")[adata.obs.value_counts("CellType")>50].index))].index]
y = np.array(adata.obs.target).astype(int)

for run in range(1, 10):
	for d in np.linspace(0, 50, 11).astype(int):

		M =	adata.layers[f"X_{str(d)}_{str(run)}"].copy()

		print(run, d, adata.X.getnnz()/(adata.shape[0]*adata.shape[1]), M.getnnz()/(adata.shape[0]*adata.shape[1]), flush=True)

		Path(f"{path}/GRAE").mkdir(parents=True, exist_ok=True)
		mlu.GR_AE(M=M, y=y, model_name=f"{path}/GRAE/{dataset}_{featurespace}_GRAE_{str(d)}_{str(run)}")


		Path(f"{path}/TopoAE").mkdir(parents=True, exist_ok=True)
		mlu.TopoAE(M=M, y=y, model_name=f"{path}/TopoAE/{dataset}_{featurespace}_TopoAE_{str(d)}_{str(run)}")


		if featurespace == "Peak":
			Path(f"{path}/PeakVI").mkdir(parents=True, exist_ok=True)
			mlu.PeakVI(M=M, model_name=f"{path}/PeakVI/{dataset}_{featurespace}_PeakVI_{str(d)}_{str(run)}")


		else:
			Path(f"{path}/scVI").mkdir(parents=True, exist_ok=True)
			mlu.scVI(M=M, model_name=f"{path}/scVI/{dataset}_{featurespace}_scVI_{str(d)}_{str(run)}")


		Path(f"{path}/VAE").mkdir(parents=True, exist_ok=True)
		mlu.VAE(M=M, y=y, model_name=f"{path}/VAE/{dataset}_{featurespace}_VAE_{str(d)}_{str(run)}") 
