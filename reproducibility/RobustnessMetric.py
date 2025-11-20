import numpy as np
import pandas as pd
import scanpy as sc

import sklearn
import os
import scipy
import torch
import scvi
import grae
import time
from grae.models import GRAE

from pathlib import Path
import matplotlib.patches as mpatches

import Models as mod

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

			#PCA
			print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> PCA",  flush=True)
			sc.pp.pca(adata, layer=f"X_{str(dp)}_{str(run)}", n_comps=int(adata.shape[1]**(1/3)))
			latent_space=adata.obsm["X_pca"].copy() 
			d=computehetero(latent_space, adata)
			d["Dataset"]=dataset
			d["FS"]=fs
			d["Method"]="PCA"
			d["Dropout"]=int(dp)
			d["Run"]=int(run)
			hetero=pd.concat([hetero,d])

			#AE
			model_name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/AE/{dataset}_{fs}_AE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> AE",  flush=True)
				data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
				model = mod.BaseAE(hidden_dim=int(adata.shape[1]**(1/2)), latent_dim=int(adata.shape[1]**(1/3)), input_dim=adata.shape[1])
				model.load_state_dict(torch.load(model_name))
			
				X=adata.X.toarray()
				X_hat=model.decode(model.encode(data)).detach().numpy()
				mse=sklearn.metrics.mean_squared_error(X, X_hat)
				medae=sklearn.metrics.median_absolute_error(X, X_hat)
				spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
				d=pd.DataFrame(data=np.array([dataset, fs, "AE", dp, mse, medae, spear, run]).T, index=rob.columns).T
				rob=pd.concat([rob,d])  

				latent_space=model.encode(data).detach().numpy()
				d=computehetero(latent_space, adata)
				d["Dataset"]=dataset
				d["FS"]=fs
				d["Method"]="AE"
				d["Dropout"]=int(dp)
				d["Run"]=int(run)
				hetero=pd.concat([hetero,d])

			else:
				print(model_name, "IS MISSING")
	   
			#VAE	
			model_name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/VAE/{dataset}_{fs}_VAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> VAE",  flush=True)
				ae_kwargs={}
				ae_kwargs["hidden_dim"]=int(adata.shape[1]**(1/2))
				ae_kwargs["latent_dim"]=int(adata.shape[1]**(1/3))
				ae_kwargs["input_dim"]=adata.shape[1]
				data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
				model = mod.VAutoencoder(ae_kwargs=ae_kwargs)
				model.load_state_dict(torch.load(model_name))

				X=adata.X.toarray()
				X_hat=model.decode(model.encode(data)[0]).detach().numpy()
				mse=sklearn.metrics.mean_squared_error(X, X_hat)
				medae=sklearn.metrics.median_absolute_error(X, X_hat)
				spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
				d=pd.DataFrame(data=np.array([dataset, fs, "VAE", dp, mse, medae, spear, run]).T, index=rob.columns).T
				rob=pd.concat([rob,d])  

				latent_space=model.encode(data)[0].detach().numpy()
				d=computehetero(latent_space, adata)
				d["Dataset"]=dataset
				d["FS"]=fs
				d["Method"]="VAE"
				d["Dropout"]=int(dp)
				d["Run"]=int(run)
				hetero=pd.concat([hetero,d])

			else:
				print(model_name, "IS MISSING")
   
			#TAE			
			model_name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/TAE/{dataset}_{fs}_TAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> TAE", flush=True)
				data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
				ae_kwargs={"input_dim" : adata.shape[1],  "hidden_dim" : int(adata.shape[1]**(1/2)), "latent_dim" : int(adata.shape[1]**(1/3))}
				model = mod.TopologicallyRegularizedAutoencoder(ae_kwargs=ae_kwargs)
				model.load_state_dict(torch.load(model_name))
			
				X=adata.X.toarray()
				X_hat=model.decode(model.encode(data)).detach().numpy()
				mse=sklearn.metrics.mean_squared_error(X, X_hat)
				medae=sklearn.metrics.median_absolute_error(X, X_hat)
				spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
				d=pd.DataFrame(data=np.array([dataset, fs, "TAE", dp, mse, medae, spear, run]).T, index=rob.columns).T
				rob=pd.concat([rob,d])    
				latent_space=model.encode(data).detach().numpy()
				d=computehetero(latent_space, adata)
				d["Dataset"]=dataset
				d["FS"]=fs
				d["Method"]="TAE"
				d["Dropout"]=int(dp)
				d["Run"]=int(run)
				hetero=pd.concat([hetero,d])
			else:
				print(model_name, "IS MISSING")
				
			#GRAE
			model_name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/GRAE/{dataset}_{fs}_GRAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> GRAE",  flush=True)
				model = GRAE(n_components=int(adata.shape[1]**(1/3)))
				model.load(model_name)
				data=grae.data.base_dataset.BaseDataset(adata.X.toarray(), np.ones(adata.shape[0]), "none", 0.85, 42, np.ones(adata.X.shape[0]))
 
				X=adata.X.toarray()
				X_hat=model.inverse_transform(model.transform(data))
				mse=sklearn.metrics.mean_squared_error(X, X_hat)
				medae=sklearn.metrics.median_absolute_error(X, X_hat)
				spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
				d=pd.DataFrame(data=np.array([dataset, fs, "GRAE", dp, mse, medae, spear, run]).T, index=rob.columns).T
				rob=pd.concat([rob,d])    

				latent_space=model.transform(data)
				d=computehetero(latent_space, adata)
				d["Dataset"]=dataset
				d["FS"]=fs
				d["Method"]="GRAE"
				d["Dropout"]=int(dp)
				d["Run"]=int(run)
				hetero=pd.concat([hetero,d])

			else:
				print(model_name, "IS MISSING")
		 
			if fs == "GEX":
				#scVI
				path=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/scVI/{dataset}_{fs}_scVI_{dp}_{run}"
				model_name=f"{path}/model.pt"
				if os.path.isfile(model_name):
					print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> scVI",  flush=True)
					scvi.model.LinearSCVI.setup_anndata(adata=adata)
					model = scvi.model.LinearSCVI(adata=adata)
					model.load(path, adata=adata)
					model.is_trained=True

					X=adata.X.toarray()
					X_hat=model.get_normalized_expression(n_samples=1).values
					mse=sklearn.metrics.mean_squared_error(X, X_hat)
					medae=sklearn.metrics.median_absolute_error(X, X_hat)
					spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
					d=pd.DataFrame(data=np.array([dataset, fs, "scVI", dp, mse, medae, spear, run]).T, index=rob.columns).T
					rob=pd.concat([rob,d])    

					latent_space=model.get_latent_representation()
					d=computehetero(latent_space, adata)
					d["Dataset"]=dataset
					d["FS"]=fs
					d["Method"]="scVI"
					d["Dropout"]=int(dp)
					d["Run"]=int(run)
					hetero=pd.concat([hetero,d])

				else:
					print(model_name, "IS MISSING")
			else:
				#PeakVI
				path=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/PeakVI/{dataset}_{fs}_PeakVI_{dp}_{run}"
				model_name=f"{path}/model.pt"
				if os.path.isfile(model_name):
					print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> PeakVI",  flush=True)
					scvi.model.PEAKVI.setup_anndata(adata=adata)
					model = scvi.model.PEAKVI(adata=adata)
					model.load(path, adata=adata)
					model.is_trained=True

					X=adata.X.toarray()
					X_hat=model.get_accessibility_estimates().values
					mse=sklearn.metrics.mean_squared_error(X, X_hat)
					medae=sklearn.metrics.median_absolute_error(X, X_hat)
					spear=np.nanmean([scipy.stats.spearmanr(X[i,:], X_hat[i,:])[0] for i in range(X.shape[0])])
					d=pd.DataFrame(data=np.array([dataset, fs, "PeakVI", dp, mse, medae, spear, run]).T, index=rob.columns).T
					rob=pd.concat([rob,d])   

					latent_space=model.get_latent_representation()
					d=computehetero(latent_space, adata)
					d["Dataset"]=dataset
					d["FS"]=fs
					d["Method"]="PeakVI"
					d["Dropout"]=int(dp)
					d["Run"]=int(run)
					hetero=pd.concat([hetero,d])

				else:
					print(model_name, "IS MISSING")


	rob.reset_index(inplace=True)
	rob.drop("index", axis=1, inplace=True)
	rob.to_csv("Tables/Robustness.tsv.gz", sep="\t", compression="gzip")

	hetero.reset_index(inplace=True)
	hetero.drop("index", axis=1, inplace=True)
	hetero.to_csv("Tables/Heteorgeneity.tsv.gz", sep="\t", compression="gzip")
