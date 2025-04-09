import warnings
warnings.filterwarnings("ignore")

import episcanpy as epi
import time
import json
import sys

import optuna

from pathlib import Path

import ML_utils as mlu
import Utils as ut
import Models as mod
import HPO as hpo

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"HPO VAE started", flush=True)

matrix = sys.argv[1]
model_name = sys.argv[2]

names=["Script","Matrix", "Model name"]
for n,arg in zip(names,sys.argv):
	print(n,arg, flush=True)
	
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Loading {matrix}",  flush=True)					 	
adata=epi.read_h5ad(matrix)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Starting study", flush=True)					 	
study = hpo.run_HPO_VAE(M=adata.X, y=adata.obs.target.to_numpy(), model_name=model_name)

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Saving results", flush=True)					 	
with open(f"{model_name}_best_params.json", "w") as f:
	json.dump(study.best_params, f)
	
	
	

