import numpy as np
import pandas as pd

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
from . import Models as mod
from . import HPO as hpo

from pathlib import Path

torch.manual_seed(np.random.randint(0,10000))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GeometricalEmbedding(M, y=None, epochs=300):
	'''
	Embedding of a feature matrix preserving geometry. See https://github.com/KevinMoonLab/GRAE for more infos 

 	Parameters
    ----------

	M : N * F matrix with N cells and F features
	
	y : target label, important for the train-val-split of cells accounting for label unbalance

	epochs : number of epoch to train the GRAE for, default = 300


	Output
	------
	
	Embedded matrix (N x latent space's dimension) and decoded matrix (N x F)

	'''

	if y != None:
		y=np.array(y).astype(int)
	else:
		y=np.ones(shape=(M.shape[0],))		
	
	M=scipy.sparse.csr_matrix(M, dtype="float32").todense()
	m = GRAE(epochs=300, patience=20, n_components=int(M.shape[1]**(1/3)))
	temp=grae.data.base_dataset.BaseDataset(M, y, "none", 0.85, 42, y)
	m.fit(temp)
	return scipy.sparse.csr_matrix(m.transform(temp), dtype="float32"), scipy.sparse.csr_matrix(m.inverse_transform(m.transform(temp)), dtype="float32")

def embbedding_and_graph(adata, y=None, layer="X", model_name="Pappo", params=None):

	'''
	Function to contruct the k-NN graph of the cell in GRAE's latent space

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	y : target label, important for the train-val-split of cells accounting for label unbalance, default = None

	layer : layer to embed, default = "X"

	epochs : number of epoch to train the GRAE for, default = 300


	Output
	------
	
	AnnData object with the graph in .obsp and the reprentation in .obsm, decoded matrix in .layer

	'''
	
	if layer == "X":
		M=adata.X.copy()
	else:
		M=adata.layers[layer].copy()
	
	Z = GeomtricalEmbedding(M, y=y)
	ad_ret=sc.AnnData(scipy.sparse.csr_matrix(Z[0], dtype="float32"))
	del Z
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")

	adata.obsp[f"{representantion}_kNN"], adata.obsm[f"{representantion}"], adata.layer[f"X_{representantion}"],  = scipy.sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32"), scipy.sparse.csr_matrix(ad_ret.X, dtype="float32"), Z[1]


def classify_and_explain(adata, label, path, hypopt=True, n_feat=50)

	'''
	Function to extract the relevant features

 	Parameters
    ----------

	adata : count matrix of class AnnData
	
	label : target label

	path : path where to save the results

	hypopt : whether to apply hyperparameter optimization to the GAT classifier, default = True

	n_feat : number of to extract, default = 50. Anyway it will be saved a file with the importance of each feature for each cell


	Output
	------
	
	AnnData object with updates infos about the classification

	'''

	Path(f"{path}/Seagal_{label}").mkdir(parents=True, exist_ok=True)
		
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Creating dataset", flush=True)					
	adata = sc.read(matrix)

	mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(adata.obs[label])))])
	inv_map = {v: k for k, v in mymap.items()}
	adata.uns["map"]=mymap
	adata.uns["inv_map"]=inv_map
	adata.obs["target"]=[mymap[x] for x in adata.obs[label]]

	edges = pd.DataFrame(adata.obsp["GRAE_graph"].todense()).rename_axis('Source')\
		.reset_index()\
		.melt('Source', value_name='Weight', var_name='Target')\
		.query('Source != Target')\
		.reset_index(drop=True)
	edges = edges[edges["Weight"]!=0]

	mydata = torch_geometric.data.Data(x=torch.tensor(scipy.sparse.csr_matrix(adata.X, dtype="float32").todense()), 
						 	 edge_index=torch.tensor(edges[["Source","Target"]].astype(int).to_numpy().T),
						 	# edge_weight=torch.tensor(edges["Weight"].astype("float32").to_numpy()),
						 	 y=torch.from_numpy(adata.obs["target"].to_numpy().astype(int)).type(torch.LongTensor))
	mydata.num_features = mydata.x.shape[1]
	mydata.num_classes = len(set(np.array(mydata.y)))

	if hypopt != False:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Starting HPO", flush=True)	
		xai_path = f"{path}/Seagal_{label}_HPO"

		if os.path.isfile(f"{xai_path}.json") == False:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "No HPO .json found --> Running HPO", flush=True)	
			mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.2, num_test=0)(mydata)
			study = hpo.run_HPO_GAT(mydata, xai_path)
			
			with open(f"{xai_path}.json", "w") as f:
				json.dump(study.best_params, f)	
			best_params = study.best_params
		else:
			print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "HPO .json found", flush=True)	
			best_params = json.load(open(f"{xai_path}.json", "r"))
		
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

	else:
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Skipping HPO", flush=True)	
		xai_path = f"{path}/Seagal_{label}"
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
		model = mlu.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes).to(device)
		optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

	class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(mydata.y), y=mydata.y.numpy())
	criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Training model", flush=True)					
	model, history = mlu.GAT_train_node_classifier(model, mydata, optimizer_model, criterion, f"{xai_path}_Model.pth"", epochs=500, patience=50)

	with open(f"{xai_path}_Model_Progress.json", "w") as f:
		json.dump(history, f)
	del history


	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Metrics for model's performances", flush=True)				
	model.eval()
	pred = model(mydata.x, mydata.edge_index).argmax(dim=1)

	adata.obs["Seagal_set"]="--"
	adata.obs["Seagal_prediction"]=[adata.uns["inv_map"][str(num)] for num in list(pred.cpu().detach().numpy())]
	adata.obs.loc[mydata.train_mask.cpu().detach().numpy(),"Seagal_set"]="Train"
	adata.obs.loc[mydata.val_mask.cpu().detach().numpy(),"Seagal_set"]="Validation"
	adata.obs.loc[mydata.test_mask.cpu().detach().numpy(),"Seagal_set"]="Test"
	adata.obs.to_csv(f"{xai_path}_Predictions.tsv.gz", sep="\t", compression="gzip")


	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI features extraction", flush=True)
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
		print(ct)
		b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
		df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False).to_numpy())[0]], axis=1)
		df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False).index)], axis=1)
	df_feat.columns=sorted(list(set(adata.obs[label])))
	df_imp.columns=sorted(list(set(adata.obs[label])))

	a.to_csv(f"{xai_path}_FeatImpCM.tsv.gz", sep="\t", compression="gzip")
	df_feat.to_csv(f"{xai_path}_Features.tsv.gz", sep="\t", compression="gzip")
	df_imp.to_csv(f"{xai_path}_FeaturesImportance.tsv.gz", sep="\t", compression="gzip")
	del df_imp, a


	a=pd.DataFrame(explanation.node_mask, index=adata.obs.index, columns=adata.var.index)
	df_feat=pd.DataFrame()
	df_imp=pd.DataFrame()
	for ct in sorted(list(set(adata.obs[label]))):
		print(ct)
		b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
		df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False)[:int(n_feat)].to_numpy())[0]], axis=1)
		df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False)[:int(n_feat)].index)], axis=1)
	df_feat.columns=sorted(list(set(adata.obs[label])))
	df_imp.columns=sorted(list(set(adata.obs[label])))

	a.to_csv(f"{xai_path}_Top{str(n_feat)}FeatImpCM.tsv.gz", sep="\t", compression="gzip")
	df_feat.to_csv(f"{xai_path}_Top{str(n_feat)}Features.tsv.gz", sep="\t", compression="gzip")
	df_imp.to_csv(f"{xai_path}_Top{str(n_feat)}FeaturesImportance.tsv.gz", sep="\t", compression="gzip")
	del df_imp, a


	jc=pd.DataFrame(index=df_feat.columns, columns=df_feat.columns)
	for column in jc.columns:
		for col in jc.columns:
			if len(df_feat[column].dropna())==0 or len(df_feat[col].dropna())==0:
				print("Problem with either {column} or {col}")
			else:	
				jc.at[column, col]=len(ut.intersection([df_feat[column].dropna(), df_feat[col].dropna()]))/len(ut.flat_list([df_feat[column].dropna(), df_feat[col].dropna()]))
	jc.to_csv(f"{xai_path}_Top{str(n_feat)}Features_Jaccard.tsv.gz", sep="\t", compression="gzip")
