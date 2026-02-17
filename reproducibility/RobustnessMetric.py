import warnings
warnings.filterwarnings('ignore')
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

from scipy.stats import nbinom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.special import gammaln
import math
from numba import njit, prange


import numpy as np
import math
from numba import njit, prange
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def estimate_theta_smoothed(X, degree=2, eps=1e-8, max_theta=1e6):
    X = np.asarray(X, dtype=np.float64)

    mu_gene = X.mean(axis=0)
    var_gene = X.var(axis=0)

    valid = mu_gene > 0
    log_mu = np.log(mu_gene[valid] + eps)
    log_var = np.log(var_gene[valid] + eps)

    poly = PolynomialFeatures(degree, include_bias=False)
    log_mu_poly = poly.fit_transform(log_mu[:, None])

    reg = LinearRegression()
    reg.fit(log_mu_poly, log_var)

    log_var_smooth = reg.predict(log_mu_poly)
    var_smooth = np.exp(log_var_smooth)

    theta = np.zeros_like(mu_gene)
    theta[valid] = mu_gene[valid] ** 2 / np.maximum(
        var_smooth - mu_gene[valid], eps
    )

    theta = np.clip(theta, eps, max_theta)
    return theta.astype(np.float64)


def precompute_nb_constants(theta):
    theta = theta.astype(np.float64)
    log_theta = np.log(theta)
    lgamma_theta = np.array([math.lgamma(t) for t in theta], dtype=np.float64)
    return theta, log_theta, lgamma_theta


@njit(parallel=True, fastmath=True)
def nb_nll_sum_numba(X, MU, theta, log_theta, lgamma_theta):
    n_cells, n_genes = X.shape
    total = 0.0

    for i in prange(n_cells):
        for g in range(n_genes):
            x = X[i, g]
            mu = MU[i, g]
            if mu < 1e-8:
                mu = 1e-8

            th = theta[g]

            log_p = (
                math.lgamma(x + th)
                - lgamma_theta[g]
                - math.lgamma(x + 1.0)
                + th * (log_theta[g] - math.log(th + mu))
                + x * (math.log(mu) - math.log(th + mu))
            )

            total -= log_p

    return total


def nb_loss_numba(X, X_hat, nb_consts, batch_size=256):
    theta, log_theta, lgamma_theta = nb_consts

    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)

    n_cells, n_genes = X.shape
    total_loss = 0.0
    total_count = 0

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)

        total_loss += nb_nll_sum_numba(
            X[start:end],
            X_hat[start:end],
            theta,
            log_theta,
            lgamma_theta,
        )

        total_count += (end - start) * n_genes

    return total_loss / total_count


def computehetero(ls, ad):
	ad_ret = sc.AnnData(scipy.sparse.csr_matrix(ls, dtype="float32"))
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")
	A = scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32")
	if scipy.sparse.issparse(A):
		A = A.tocsr()

	celltypes = ad.obs["CellType"].astype("category").values
	n_neighbors = np.zeros(A.shape[0], dtype=int)
	n_unique_celltypes = np.zeros(A.shape[0], dtype=int)

	for i in range(A.shape[0]):
		neighbors = A[i].indices
		n_neigh = len(neighbors)
		n_neighbors[i] = n_neigh

		if n_neigh == 0:
			n_unique_celltypes[i] = np.nan
			continue

		neighbor_types = celltypes[neighbors]
		n_unique = len(np.unique(neighbor_types))
		n_unique_celltypes[i] = n_unique

	d = pd.DataFrame([n_neighbors, n_unique_celltypes], index=["NN", "AbsoluteHetero"]).T
	d["N_CT"] = len(np.unique(celltypes))
	return d


def process_dataset(dataset_info):
	dataset, fs, job = dataset_info
	print(f"Starting processing for {dataset} with feature space {fs}...", flush=True)

	adata = sc.read_h5ad(f"Datasets/{dataset}/FeatureSpaces/{fs}/CM/{dataset}_{fs}_Dropout.h5ad")

	X = adata.X.toarray()
	theta = estimate_theta_smoothed(X)
	nb_consts = precompute_nb_constants(theta)
	rob_rows = []
	# hetero_rows = []

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

			# ================= AE =================
			model_name = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/AE/{dataset}_{fs}_AE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> AE", flush=True)

				data = torch.tensor(X, dtype=torch.float32)
				model = mod.BaseAE(input_dim=adata.shape[1],hidden_dim=int(adata.shape[1]**0.5),latent_dim=int(adata.shape[1]**(1/3)))
				model.load_state_dict(torch.load(model_name, map_location="cpu"))

				X_hat = model.decode(model.encode(data)).detach().numpy()

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
				rob_rows.append([dataset, fs, "AE", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

				latent_space = model.encode(data).detach().numpy()
				d = computehetero(latent_space, adata)
				d["Dataset"] = dataset
				d["FS"] = fs
				d["Method"] = "AE"
				d["Dropout"] = int(dp)
				d["Run"] = int(run)
				hetero_rows.append(d)

			else:
				print(f"{model_name} IS MISSING")

			# ================= VAE =================
			model_name = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/VAE/{dataset}_{fs}_VAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> VAE", flush=True)

				data = torch.tensor(X, dtype=torch.float32)
				model = mod.VAutoencoder(ae_kwargs={"input_dim": adata.shape[1],"hidden_dim": int(adata.shape[1]**0.5),"latent_dim": int(adata.shape[1]**(1/3))})
				model.load_state_dict(torch.load(model_name, map_location="cpu"))

				X_hat = model.decode(model.encode(data)[0]).detach().numpy()

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
				rob_rows.append([dataset, fs, "VAE", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

				latent_space = model.encode(data)[0].detach().numpy()
				d = computehetero(latent_space, adata)
				d["Dataset"] = dataset
				d["FS"] = fs
				d["Method"] = "VAE"
				d["Dropout"] = int(dp)
				d["Run"] = int(run)
				hetero_rows.append(d)

			else:
				print(f"{model_name} IS MISSING")

			# ================= TAE =================
			model_name = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/TAE/{dataset}_{fs}_TAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> TAE", flush=True)

				data = torch.tensor(X, dtype=torch.float32)
				model = mod.TopologicallyRegularizedAutoencoder(ae_kwargs={"input_dim": adata.shape[1],"hidden_dim": int(adata.shape[1]**0.5),"latent_dim": int(adata.shape[1]**(1/3))})
				model.load_state_dict(torch.load(model_name, map_location="cpu"))

				X_hat = model.decode(model.encode(data)).detach().numpy()

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
				rob_rows.append([dataset, fs, "TAE", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

				latent_space = model.encode(data).detach().numpy()
				d = computehetero(latent_space, adata)
				d["Dataset"] = dataset
				d["FS"] = fs
				d["Method"] = "TAE"
				d["Dropout"] = int(dp)
				d["Run"] = int(run)
				hetero_rows.append(d)

			else:
				print(f"{model_name} IS MISSING")

			# ================= GRAE =================
			model_name = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/GRAE/{dataset}_{fs}_GRAE_{dp}_{run}.pth"
			if os.path.isfile(model_name):
				print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> GRAE", flush=True)

				model = GRAE(n_components=int(adata.shape[1]**(1/3)))
				model.load(model_name)
				data_grae = grae.data.base_dataset.BaseDataset(X, np.ones(X.shape[0]), "none", 0.85, 42, np.ones(X.shape[0]))

				X_hat = model.inverse_transform(model.transform(data_grae))

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
				rob_rows.append([dataset, fs, "GRAE", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

				latent_space = model.transform(data_grae)
				d = computehetero(latent_space, adata)
				d["Dataset"] = dataset
				d["FS"] = fs
				d["Method"] = "GRAE"
				d["Dropout"] = int(dp)
				d["Run"] = int(run)
				hetero_rows.append(d)

			else:
				print(f"{model_name} IS MISSING")

			# ================= scVI / PeakVI =================
			if fs == "GEX":
				path = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/scVI/{dataset}_{fs}_scVI_{dp}_{run}"
				model_name = f"{path}/model.pt"
				if os.path.isfile(model_name):
					print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> scVI", flush=True)

					scvi.model.LinearSCVI.setup_anndata(adata)
					model = scvi.model.LinearSCVI(adata)
					model.load(path, adata=adata)

					X_hat = model.get_normalized_expression(n_samples=1).values

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
					rob_rows.append([dataset, fs, "scVI", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

					latent_space = model.get_latent_representation()
					d = computehetero(latent_space, adata)
					d["Dataset"] = dataset
					d["FS"] = fs
					d["Method"] = "scVI"
					d["Dropout"] = int(dp)
					d["Run"] = int(run)
					hetero_rows.append(d)

				else:
					print(f"{model_name} IS MISSING")
			else:
				path = f"Datasets/{dataset}/FeatureSpaces/{fs}/Dropout/PeakVI/{dataset}_{fs}_PeakVI_{dp}_{run}"
				model_name = f"{path}/model.pt"
				if os.path.isfile(model_name):
					print(f"{dataset} - Run {run}/10 and dropout {dp}", time.strftime("%a, %d %b %Y %H:%M:%S"), "AE is --> PeakVI", flush=True)

					scvi.model.PEAKVI.setup_anndata(adata)
					model = scvi.model.PEAKVI(adata)
					model.load(path, adata=adata)

					X_hat = model.get_accessibility_estimates().values

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
					rob_rows.append([dataset, fs, "PeakVI", int(dp),mse, mse1obs, mse1imp, spear, NBloss, int(run)])

					latent_space = model.get_latent_representation()
					d = computehetero(latent_space, adata)
					d["Dataset"] = dataset
					d["FS"] = fs
					d["Method"] = "PeakVI"
					d["Dropout"] = int(dp)
					d["Run"] = int(run)
					hetero_rows.append(d)

				else:
					print(f"{model_name} IS MISSING")

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

	with ctx.Pool(processes=n_workers) as pool:
		results = pool.map(process_dataset, dataset_info_list)

	rob = pd.concat(results[0], ignore_index=True)
	rob.to_csv("Tables/Robustness.tsv.gz", sep="\t", compression="gzip", index=False)

	hetero = pd.concat(results[1], ignore_index=True)
	hetero.to_csv("Tables/Heterogeneity.tsv.gz", sep="\t", compression="gzip", index=False)


	elapsed_time = time.time() - start_time
	print(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
	main()
