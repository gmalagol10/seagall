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

def geometrical_embedding(M, y=None, epochs=300, patience=20, path="SEAGALL", model_name="mymodel", overwrite=False):
	'''
	Embedding of a feature matrix preserving geometry. See https://github.com/KevinMoonLab/GRAE for more infos 

 	Parameters
    ----------

	M : N * F matrix with N cells and F features
	
	y : array containing the class of each cell, default = None

	epochs : number of epoch to train the GRAE for, default = 300

	patience : early stopping threshold, default = 20

	path : folder where to save the model, default =  SEAGALL

	model_name : name of the model, default = mymodel


	Output
	------
	
	Embedded matrix (N x latent space's dimension) and decoded matrix (N x F)

	'''
	if os.path.isfile(f"{path}/SEAGALL_{model_name}_GRAE.pth") == True and override == False:
		m = GRAE(n_components=int(np.around(M.shape[1]**(1/3), decimals=0)))
		dataset = grae.data.base_dataset.BaseDataset(M, y=y, split='none', split_ratio=1, random_state=42, labels=y)
		return m.transform(dataset), scipy.sparse.csr_matrix(m.inverse_transform(m.transform(dataset)), dtype="float32")


	Path(path).mkdir(parents=True, exist_ok=True)

	if y is None:
		y=np.ones(shape=(M.shape[0],))	

	M = scipy.sparse.csr_matrix(M, dtype="float32").toarray()
	m = GRAE(epochs=epochs, patience=patience, n_components=int(np.around(M.shape[1]**(1/3), decimals=0)))

	dataset = grae.data.base_dataset.BaseDataset(M, y=y, split='none', split_ratio=1, random_state=42, labels=y)
	train_dataset, val_dataset = dataset.validation_split(ratio=0.15)
	
	m.fit(train_dataset)
	m.save(f"{path}/SEAGALL_{model_name}_GRAE.pth")
	return m.transform(dataset), scipy.sparse.csr_matrix(m.inverse_transform(m.transform(dataset)), dtype="float32")

def geometrical_graph(adata, label=None, layer="X", epochs=300, patience=20, path="SEAGALL", model_name="mymodel", overwrite=False):

	'''
	Function to contruct the k-NN graph of the cell in GRAE's latent space

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	label : target label, important for the train-val split of cells accounting for label unbalance, default = None

	layer : layer to embed, default = "X"

	epochs : number of epoch to train the GRAE for, default = 300

	patience : early stopping threshold, default = 20


	path : folder where to save the model, default =  SEAGALL

	model_name : name of the model, default = mymodel

	Output
	------
	
	AnnData object with the graph in .obsp, the latent space in .obsm, decoded matrix in .layers

	'''
	adata.var_names_make_unique()

	if label is not None:
		adata.obs[label].astype(str).replace("nan","unknown")
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

	Z = geometrical_embedding(M=M, y=y, epochs=epochs, patience=patience, path=path, model_name=model_name, overwrite=overwrite)
	ad_ret=sc.AnnData(Z[0])
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")

	adata.obsp[f"GRAE_graph"], adata.obsm["GRAE_latent_space"], adata.layers[f"GRAE_decoded_matrix"]  = scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), Z[0], Z[1]


def classify_and_explain(adata, label, hypopt=1, n_feat=50, path="SEAGALL", model_name="mymodel"):

	'''
	Function to extract the relevant features

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	label : target label

	hypopt : fraction of cells to use to run HPO, default 1 (all the cells) 0 for not run it

	n_feat : number of to extract, default = 50. However, in adata.var will be saved the importance of each feature for each ground truth label

	path : folder where to save the model, default =  SEAGALL

	model_name : name of the model, default = mymodel

	Output
	------
	
	AnnData object with importance of each feature for each class in .var and predictions infos in .obs
	The classifier is saved in {path}/SEAGALL_{model_name}_{label}

	'''

	Path(path).mkdir(parents=True, exist_ok=True)
	path=f"{path}/SEAGALL_{model_name}_{label}"

	adata.obs[label]=np.array(adata.obs[label].astype(str).replace("nan","unknown"))
	adata.var_names_make_unique()
	mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(adata.obs[label])))])
	inv_map = {v: k for k, v in mymap.items()}
	adata.uns["map"]=mymap
	adata.uns["inv_map"]=inv_map
	adata.obs["target"]=[mymap[x] for x in adata.obs[label]]
		
	if hypopt > 0:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Looking for HPO file", flush=True)
		hpo_path = f"{path}_HPO"

		if os.path.isfile(f"{hpo_path}.json") == False:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"No HPO .json found --> Running HPO using {int(100*hypopt)}% of the cells", flush=True)
			mydata = mlu.create_pyg_dataset(adata, label, hypopt)
			mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.2, num_test=0)(mydata)
			study = hpo.run_HPO_GAT(mydata, hpo_path)
			
			with open(f"{hpo_path}.json", "w") as f:
				json.dump(study.best_params, f)
			best_params = study.best_params
		else:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "HPO .json found", flush=True)
			best_params = json.load(open(f"{hpo_path}.json", "r"))
			for key, value in best_params.items():
				print(f"Best value for {key} is {value}", flush=True)	
		
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Creating dataset", flush=True)
		mydata = mlu.create_pyg_dataset(adata, label)
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

	else:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Creating dataset, no HPO", flush=True)
		mydata = mlu.create_pyg_dataset(adata, label)
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

	class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(mydata.y), y=mydata.y.numpy())
	criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Training model", flush=True)
	model, history = mlu.GAT_train_node_classifier(model, mydata, optimizer_model, criterion, f"{path}.pth", epochs=300, patience=20)

	with open(f"{path}_GAT_Progress.json", "w") as f:
		json.dump(history, f)
	del history

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Metrics for model's performances", flush=True)
	model.eval()
	pred = model(mydata.x, mydata.edge_index).argmax(dim=1)

	adata.obs["SEAGALL_set"] = "--"
	adata.obs["SEAGALL_prediction"] = [adata.uns["inv_map"][str(num)] for num in list(pred.cpu().detach().numpy())]
	adata.obs.loc[mydata.train_mask.cpu().detach().numpy(),"SEAGALL_set"] = "Train"
	adata.obs.loc[mydata.val_mask.cpu().detach().numpy(),"SEAGALL_set"] = "Validation"
	adata.obs.loc[mydata.test_mask.cpu().detach().numpy(),"SEAGALL_set"] = "Test"

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

	feat_imp_matrix = pd.DataFrame(explanation.node_mask, index=adata.obs.index, columns=adata.var.index)
	for gt in sorted(set(adata.obs[label])):
		imps = np.array(feat_imp_matrix.loc[adata[adata.obs[label]==gt].obs.index].mean(axis=0))
		adata.var[f"Importance_for_{gt}"] = imps

	gts = sorted(set(adata.obs[label]))
	jc = pd.DataFrame(index=gts, columns=gts)
	for gti in gts:
		for gtj in gts[gts.index(gt)+1:]:
			fsi = adata[adata.obs[label]==gti].var[f"Importance_for_{gti}"].sort_values()[::-1][:int(n_feat)].index
			fsj = adata[adata.obs[label]==gtj].var[f"Importance_for_{gtj}"].sort_values()[::-1][:int(n_feat)].index
			jc.at[gti, gtj] = len(ut.intersection([fsi, fsj]))/len(ut.flat_list([fsi, fsj]))
	jc.to_csv(f"{path}_Top{str(n_feat)}Features_Jaccard.tsv.gz", sep="\t", compression="gzip")
