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

from multiprocessing import get_context
from RobustnessMetric import estimate_theta_smoothed, precompute_nb_constants, nb_loss_numba, computehetero, nb_nll_sum_numba
import warnings
warnings.filterwarnings('ignore')


def process_dataset(dataset_info):
	dataset, fs, job = dataset_info
	print(f"Starting processing for {dataset} with feature space {fs}...", flush=True)

	adata = sc.read_h5ad(f"Datasets/{dataset}/FeatureSpaces/{fs}/CM/{dataset}_{fs}_Dropout.h5ad")

	X = adata.X.toarray()
	theta = estimate_theta_smoothed(X)
	nb_consts = precompute_nb_constants(theta)

	rob_rows = []
	hetero_rows = []

	for run in range(0, 10):
		for dp in np.linspace(0, 50, 6).astype(int):

			# ================= siVAE =================
			model_name = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/AE/{dataset}_{fs}_AE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				name=f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/siVAE/{dataset}_{fs}_siVAE_{dp}_{run}"
				print(f"Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "AE is --> siVAE",  flush=True)
				X_hat=pd.read_csv(f"{name}_x_hat.tsv.gz", index_col=0, sep="\t").values
				latent_space=pd.read_csv(f"{name}_latent.tsv.gz", index_col=0, sep="\t").values  
				X=adata.X.toarray().copy()

				top1obs = adata.var.sort_values(by="log_n_cells")[::-1][:int(adata.var.shape[0]/100)].index
				top1obs = adata.var.index.isin(top1obs)
				df_hat = pd.DataFrame(np.log10(np.sum(X_hat, axis=0)), index=adata.var.index, columns=["log_n_cells"])
				top1imp = df_hat.sort_values(by="log_n_cells")[::-1][:int(adata.var.shape[0]/100)].index
				top1imp = adata.var.index.isin(top1imp)

				mse1obs = sklearn.metrics.mean_squared_error(X[:, top1obs], X_hat[:, top1obs])
				mse1imp = sklearn.metrics.mean_squared_error(X[:, top1imp], X_hat[:, top1imp])
				mse = sklearn.metrics.mean_squared_error(X, X_hat)
				spear = np.nanmean([scipy.stats.spearmanr(X[i], X_hat[i])[0] for i in range(X.shape[0])])
				NBloss = nb_loss_numba(X, X_hat, nb_consts, batch_size=1024)
				rob_rows.append([dataset, fs, "siVAE", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

				latent_space = model.encode(data).detach().numpy()
				d = computehetero(latent_space, adata)
				d["Dataset"] = dataset
				d["FS"] = fs
				d["Method"] = "siVAE"
				d["Dropout"] = int(dp)
				d["Run"] = int(run)
				hetero_rows.append(d)


	print(f"Completed processing for {dataset} with feature space {fs}", flush=True)

	rob = pd.DataFrame(rob_rows,columns=["Dataset","FS","AE","Dropout","MSE","MSE1obs","MSE1imp","Spearman", "NBloss", "Run"])
	hetero = pd.DataFrame(hetero_rows, columns=["Dataset", "FS", "Method", "AbsoluteHetero", "NN", "N_CT"])
	return rob, hetero


def main():
	datasets = ["10XhsBrain3kMO","10XhsBrain3kMO","Kidney","10XhsPBMC10kMO","10XhsPBMC10kMO","MouseBrain"]
	featurespaces = ["Peak","GEX","Peak","Peak","GEX","Peak"]
	jobs = ["BrP","BrG","KiP","PbP","PbG","MbP"]

	dataset_info_list = list(zip(datasets, featurespaces, jobs))

	print(f"Starting parallel processing for {len(dataset_info_list)} datasets...")
	start_time = time.time()

	ctx = get_context("spawn")
	n_workers = min(len(dataset_info_list), max(1, os.cpu_count() // 2))

	n_workers = min(len(dataset_info_list), max(1, os.cpu_count() // 2))

	with ctx.Pool(processes=n_workers) as pool:
		results = pool.map(process_dataset, dataset_info_list)

	rob = pd.concat(results[0], ignore_index=True)
	rob.to_csv("Tables/Robustness_siVAE.tsv.gz", sep="\t", compression="gzip", index=False)

	hetero = pd.concat(results[1], ignore_index=True)
	hetero.to_csv("Tables/Heterogeneity_siVAE.tsv.gz", sep="\t", compression="gzip", index=False)


	elapsed_time = time.time() - start_time
	print(f"Processing completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
	main()
