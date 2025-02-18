import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

import sys
import random
import time
import os
import sklearn
import json
import scipy

import torch
import torch_geometric

import grae
from grae.models import GRAE

from . import ML_utils as mlu
from . import Utils as ut
from . import HPO as hpo

from pathlib import Path

torch.manual_seed(np.random.randint(0,10000))
device = 'cpu'

def GeometricalEmbedding(M, y, epochs=300, patience=20, train_size=0.85, model_name="SeagallGRAE"):
	'''
	Embedding of a feature matrix preserving geometry. See https://github.com/KevinMoonLab/GRAE for more infos 

 	Parameters
    ----------

	M : N * F matrix with N cells and F features
	
	y : array containing the class of each cell

	epochs : number of epoch to train the GRAE for, default = 300

	patience : early stopping threshold, default = 20

	train_size : fraction of dataset to use for training the GRAE, default = 0.85

	model_name : name to use to save the model, default =  SeagallGRAE


	Output
	------
	
	Embedded matrix (N x latent space's dimension) and decoded matrix (N x F)

	'''
	
	M = scipy.sparse.csr_matrix(M, dtype="float32").toarray()
	m = GRAE(epochs=epochs, patience=patience, n_components=int(np.around(M.shape[1]**(1/3), decimals=0)))
	temp=grae.data.base_dataset.BaseDataset(M, y, "none", train_size, 42, y)
	m.fit(temp)
	m.save(f"{model_name}.pth")
	return m.transform(temp), scipy.sparse.csr_matrix(m.inverse_transform(m.transform(temp)), dtype="float32")

def embbedding_and_graph(adata, label=None, layer="X", epochs=300, patience=20, train_size=0.85, model_name="SeagallGRAE"):

	'''
	Function to contruct the k-NN graph of the cell in GRAE's latent space

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	label : target label, important for the train-val-split of cells accounting for label unbalance, default = None

	layer : layer to embed, default = "X"

	epochs : number of epoch to train the GRAE for, default = 300

	patience : early stopping threshold, default = 20

	train_size : fraction of dataset to use for training the GRAE, default = 0.85

	model_name : name to use to save the model, default =  SeagallGRAE


	Output
	------
	
	AnnData object with the graph in .obsp and the representation in .obsm, decoded matrix in .layer

	'''
	adata.var_names_make_unique()

	if label is not None:
		adata=adata[adata.obs[label].dropna().index]
		mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(adata.obs[label])))])
		inv_map = {v: k for k, v in mymap.items()}
		adata.uns["map"]=mymap
		adata.uns["inv_map"]=inv_map
		adata.obs["target"]=[mymap[x] for x in adata.obs[label]]
		adata.obs["target"]=adata.obs["target"].astype(int)
		y=np.array(adata.obs["target"]).astype(int)
	else:
		y=np.ones(shape=(adata.shape[0],))	

	if layer == "X":
		M=adata.X.copy()
	else:
		M=adata.layers[layer].copy()

	Z = GeometricalEmbedding(M, y=y, epochs=epochs, patience=patience, train_size=train_size, model_name=model_name)
	ad_ret=sc.AnnData(Z[0])
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")

	adata.obsp[f"GRAE_graph"], adata.obsm["GRAE_latent_space"], adata.layers[f"GRAE_decoded_matrix"]  = scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), Z[0], Z[1]


def classify_and_explain(adata, label, path, hypopt=1, n_feat=50):

	'''
	Function to extract the relevant features

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	label : target label

	path : path where to save the results

	hypopt : fraction of cells to use to run HPO, default 1 (all the cells) 0 for not run it

	n_feat : number of to extract, default = 50. Anyway it will be saved a file with the importance of each feature for each cell


	Output
	------
	
	AnnData object with updates infos about the classification

	'''

	path=f"{path}/Seagal_{label}"
	Path(path).mkdir(parents=True, exist_ok=True)

	adata=adata[adata.obs[label].dropna().index]
	adata.var_names_make_unique()
	mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(adata.obs[label])))])
	inv_map = {v: k for k, v in mymap.items()}
	adata.uns["map"]=mymap
	adata.uns["inv_map"]=inv_map
	adata.obs["target"]=[mymap[x] for x in adata.obs[label]]
		
	if hypopt > 0:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Looking for HPO file", flush=True)
		xai_path = f"{path}/Seagal_{label}_HPO"

		if os.path.isfile(f"{xai_path}.json") == False:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"No HPO .json found --> Running HPO using {hypopt} of the cells", flush=True)
			mydata = mlu.create_pyg_dataset(adata, label, hypopt)
			mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.2, num_test=0)(mydata)
			study = hpo.run_HPO_GAT(mydata, xai_path)
			
			with open(f"{xai_path}.json", "w") as f:
				json.dump(study.best_params, f)
			best_params = study.best_params
		else:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "HPO .json found", flush=True)
			best_params = json.load(open(f"{xai_path}.json", "r"))
			for key, value in best_params.items():
				print(f"Best value for {key} is {value}", flush=True)	
		
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Creating dataset", flush=True)
		mydata = mlu.create_pyg_dataset(adata, label)
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

	else:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Creating dataset, no HPO", flush=True)
		xai_path = f"{path}/Seagal_{label}"
		mydata = mlu.create_pyg_dataset(adata, label)
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

	class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(mydata.y), y=mydata.y.numpy())
	criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Training model", flush=True)
	model, history = mlu.GAT_train_node_classifier(model, mydata, optimizer_model, criterion, f"{xai_path}_Model.pth", epochs=500, patience=50)

	with open(f"{xai_path}_Model_Progress.json", "w") as f:
		json.dump(history, f)
	del history


	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Metrics for model's performances", flush=True)
	model.eval()
	pred = model(mydata.x, mydata.edge_index).argmax(dim=1)

	adata.obs["Seagal_set"]="--"
	adata.obs["Seagal_prediction"]=[adata.uns["inv_map"][str(num)] for num in list(pred.cpu().detach().numpy())]
	adata.obs.loc[mydata.train_mask.cpu().detach().numpy(),"Seagal_set"]="Train"
	adata.obs.loc[mydata.val_mask.cpu().detach().numpy(),"Seagal_set"]="Validation"
	adata.obs.loc[mydata.test_mask.cpu().detach().numpy(),"Seagal_set"]="Test"
	adata.obs.to_csv(f"{xai_path}_Predictions.tsv.gz", sep="\t", compression="gzip")


	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "XAI features extraction", flush=True)
	explainer = torch_geometric.explain.Explainer(
				model=model,
				algorithm=torch_geometric.explain.GNNExplainer(epochs=200),
				explanation_type='model',
				node_mask_type='attributes',
				edge_mask_type='object',
				model_config=dict(
					mode='multiclass_classification',
					task_level='node',
					return_type='probs',),)
				   
	explanation = explainer(x=mydata.x, edge_index=mydata.edge_index)

	a=pd.DataFrame(explanation.node_mask, index=adata.obs.index, columns=adata.var.index)
	df_feat=pd.DataFrame()
	df_imp=pd.DataFrame()
	for ct in sorted(list(set(adata.obs[label]))):
		b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
		df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False).to_numpy())[0]], axis=1)
		df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False).index)], axis=1)
	df_feat.columns=sorted(list(set(adata.obs[label])))
	df_imp.columns=sorted(list(set(adata.obs[label])))

	a.to_csv(f"{xai_path}_FeatImpCM.tsv.gz", sep="\t", compression="gzip")
	df_feat.to_csv(f"{xai_path}_Features.tsv.gz", sep="\t", compression="gzip")
	df_imp.to_csv(f"{xai_path}_FeaturesImportance.tsv.gz", sep="\t", compression="gzip")
	del df_imp, a


	jc=pd.DataFrame(index=df_feat.columns, columns=df_feat.columns)
	for column in jc.columns:
		for col in jc.columns:
			if len(df_feat[column].dropna())==0 or len(df_feat[col].dropna())==0:
				print("Problem with either {column} or {col}")
			else:	
				fsi=df_feat[column].dropna()[:int(n_feat)]
				fsj=df_feat[col].dropna()[:int(n_feat)]
				jc.at[column, col]=len(ut.intersection([fsi, fsj]))/len(ut.flat_list([fsi, fsj]))
	jc.to_csv(f"{xai_path}_Top{str(n_feat)}Features_Jaccard.tsv.gz", sep="\t", compression="gzip")
