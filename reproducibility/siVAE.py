import numpy as np
import scanpy as sc
import pandas as pd

import sys
import os

from pathlib import Path

def apply_siVAE(M, y=[], folder, name):

	## System
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'  # no debugging from TF
	os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

	import logging
	logging.getLogger('tensorflow').disabled = True
	logging.getLogger().setLevel(logging.INFO)

	## Tensorflow
	import tensorflow as tf
	tf.get_logger().setLevel('INFO')
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	## Scanpy
	import scanpy as sc
	#### Set up tf config
	gpu_device = '0'
	os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = True
	config.allow_soft_placement = True
	config.intra_op_parallelism_threads = 5
	config.inter_op_parallelism_threads = 5
	from siVAE.data.data_handler import data2handler
	from siVAE.run_model import run_VAE


	adata=sc.AnnData(M.toarray(), obs=pd.DataFrame(y))
	adata.obs.columns=["Labels"]
	datah_sample, datah_feature, plot_args = data2handler(adata)

	#### Setup the train/test/validation split
	k_split=0.85
	datah_sample.create_split_index_list(k_split=k_split,random_seed=0)
	#### Training Parameters
	iter          = 3
	mb_size       = 0.2
	l2_scale      = 1e-3
	keep_prob     = 1
	learning_rate = 1e-4
	early_stop    = True
	decay_rate    = 0.9

	#### Model parameters
	# Architecture should be a string with a specific format
	# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
	architecture = '1024-512-128-LE-128-512-1024-0-3'
	decoder_activation = 'NA'
	zv_recon_scale = 0.1
	LE_dim = int(adata.shape[1]**(1/3))
	datah_sample.create_dataset(kfold_idx=0)

	#### Set parameters

	graph_args = {'LE_dim'       : LE_dim,
		          'architecture' : architecture,
		          'config'       : config,
		          'iter'         : iter,
		          'mb_size'      : mb_size,
		          'l2_scale'     : l2_scale,
		          'tensorboard'  : True,
		          'batch_norm'   : False,
		          'keep_prob'    : keep_prob,
		          'log_frequency': 50,
		          'learning_rate': learning_rate,
		          "early_stopping"   : early_stop,
		          "validation_split" : 0,
		          "decay_rate"       : decay_rate,
		          "decay_steps"      : 1000,
		          'var_dependency'   : True,
		          'activation_fun'   : tf.nn.relu,
		          'activation_fun_decoder': tf.nn.relu,
		          'output_distribution': 'normal',
		          'beta'               : 1,
		          'l2_scale_final'     : 5e-3,
		          'log_variational'    : False,
		          'beta_warmup'        : 1000,
		          'max_patience_count' : 100}

	logdir=folder
	graph_args['logdir_tf'] = logdir
	os.makedirs(logdir,exist_ok=True)

	siVAE_output = run_VAE(graph_args_sample=graph_args,
		                    LE_method='siVAE',
		                    datah_sample=datah_sample,
		                    datah_feature=datah_feature)

	siVAE_output.save(filename=os.path.join(folder,f'{name}_result.pickle'))

	lat_space=siVAE_output["model"]["latent_embedding"]["sample"]
	x_hat=siVAE_output["model"]["reconstruction"][1]
	return lat_space, x_hat


dataset = sys.argv[1]
featurespace = sys.argv[1]

names = ["Script", "Dataset", "Features space"]
for n,arg in zip(names,sys.argv):
	print(n,arg, flush=True)

path = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/Dropout"
matrix = f"Datasets/{dataset}/FeatureSpaces/{featurespace}/CM/{dataset}_{featurespace}_Dropout.h5ad"
adata = sc.read_h5ad(matrix)
adata = adata[adata.obs[adata.obs["CellType"].isin(list(adata.obs.value_counts("CellType")[adata.obs.value_counts("CellType")>50].index))].index]
y = np.array(adata.obs.target).astype(int)
Path(f"{path}/siVAE").mkdir(parents=True, exist_ok=True)
for run in range(0, 10):
	for d in np.linspace(0, 50, 6).astype(int):

		M = adata.layers[f"X_{str(d)}_{str(run)}"].copy()

		print(run, d, adata.X.getnnz()/(adata.shape[0]*adata.shape[1]), M.getnnz()/(adata.shape[0]*adata.shape[1]), flush=True)

		Path(f"{path}/siVAE").mkdir(parents=True, exist_ok=True)

		out = apply_siVAE(M=M, y=y, folder=f"{path}/siVAE"  name=f"{dataset}_{featurespace}_siVAE_{str(d)}_{str(run)}")
		pd.DataFrame(out[0]).to_csv("{path}/siVAE/{dataset}_{featurespace}_siVAE_{str(d)}_{str(run)}_latent.tsv.gz", sep="\t", compression="gzip")
		pd.DataFrame(out[1]).to_csv("{path}/siVAE/{dataset}_{featurespace}_siVAE_{str(d)}_{str(run)}_x_hat.tsv.gz", sep="\t", compression="gzip")
		del out
