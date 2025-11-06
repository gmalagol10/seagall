import numpy as np
import pandas as pd
import scanpy as sc

import sklearn
import os
import scipy
import time

from pathlib import Path

def computehetero(ls, ad):
	ad_ret=sc.AnnData(scipy.sparse.csr_matrix(ls, dtype="float32"))
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
	A=scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32")
	if scipy.sparse.issparse(A):
		A = A.tocsr()

	celltypes = ad.obs["CellType"].astype("category").values
	n_neighbors = np.zeros(A.shape[0], dtype=int)
	n_unique_celltypes = np.zeros(A.shape[0], dtype=int)
	
	# compute per-cell metrics
	for i in range(A.shape[0]):
		neighbors = A[i].indices
		n_neigh = len(neighbors)
		n_neighbors[i] = n_neigh
	
		if n_neigh == 0:
			n_unique_celltypes[i] = np.nan
			heterogeneity_ratio[i] = np.nan
			prop_same_type[i] = np.nan
			continue
	
		neighbor_types = celltypes[neighbors]
		n_unique = len(np.unique(neighbor_types))
	
		n_unique_celltypes[i] = n_unique
	d=pd.DataFrame([n_neighbors, n_unique_celltypes], index=["NN","AbsoluteHetero"]).T
	d["N_CT"]=len(np.unique(celltypes))
	return d

datasets=["10XhsBrain3kMO", "10XhsBrain3kMO","Kidney", "10XhsPBMC10kMO","10XhsPBMC10kMO", "MouseBrain"]
featurespaces=["Peak","GEX","Peak", "Peak", "GEX", "Peak"]
jobs=["BrP", "BrG", "KiP", "PbP", "PbG", "MbP"]
rob=pd.DataFrame(columns=["Dataset","FS","AE","Dropout","MSE", "MedAE","Spearman","Run"])
hetero=pd.DataFrame(columns=["Dataset","FS","Method","AbsoluteHetero","NN","N_CT"])
for dataset, fs, job in zip(datasets, featurespaces, jobs):
	adata=sc.read_h5ad(f"Datasets/{dataset}/FeatureSpaces/{fs}/CM/{dataset}_{fs}_Dropout.h5ad")
	for run in range(0, 10):
		for dp in np.linspace(0, 50, 6).astype(int):
			name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/siVAE/{dataset}_{fs}_siVAE_{dp}_{run}"
			print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> siVAE",  flush=True)
			X_hat=pd.read_csv(f"{name}_x_hat.tsv.gz", index_col=0, sep="\t").values
			latent_space=pd.read_csv(f"{name}_latent.tsv.gz", index_col=0, sep="\t").values  
			X=adata.X.toarray().copy()
			mse=sklearn.metrics.mean_squared_error(X, X_hat)
			medae=sklearn.metrics.median_absolute_error(X, X_hat)
			spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
			d=pd.DataFrame(data=np.array([dataset, fs, AE, dp, mse, medae, spear, run]).T, index=rob.columns).T
			rob=pd.concat([rob,d])  
			
			d=computehetero(latent_space, adata)
			d["Dataset"]=dataset
			d["FS"]=fs
			d["Method"]="siVAE"
			d["Dropout"]=int(dp)
			d["Run"]=int(run)
			hetero=pd.concat([hetero,d])

	rob.reset_index(inplace=True)
	rob.drop("index", axis=1, inplace=True)
	rob.to_csv("Tables/Robustness_siVAE.tsv.gz", sep="\t", compression="gzip")

	hetero.reset_index(inplace=True)
	hetero.drop("index", axis=1, inplace=True)
	hetero.to_csv("Tables/Heteorgeneity_siVAE.tsv.gz", sep="\t", compression="gzip")
