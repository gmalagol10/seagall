import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import episcanpy as epi
import anndata as ad
import networkx as nx

import sklearn
import time
import os
import scipy
import json
import sys

import optuna
import torch
import torch_geometric

from pathlib import Path

import ML_utils as mlu
import Utils as ut
import Models as mod
import HPO as hpo

print(f"MS Script started at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

matrix = sys.argv[1]
path = sys.argv[2]
name = sys.argv[3]
rep = sys.argv[4]
gnn = sys.argv[5]

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']	
n_feat = 50

names = ["Script","Matrix", "Path","Name", "Representation", "GNN"]
for n,arg in zip(names,sys.argv):
	print(n,arg, flush=True)
	

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Loading {matrix}", flush=True)					 	
adata = epi.read_h5ad(matrix)

Path(f"{path}/BestParams").mkdir(parents=True, exist_ok=True)	

hpo_name=f"{path}/BestParams/{name}_best"

if os.path.isfile(f"{hpo_name}.json") == False:

	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"HPO", flush=True)		
	epi.pp.neighbors(adata, use_rep=f"X_{rep}", method="umap")
	edges = pd.DataFrame(adata.obsp["connectivities"].todense()).rename_axis('Source').reset_index().melt('Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)
	edges = edges[edges["Weight"]!=0]
	mydata = torch_geometric.data.Data(x=torch.tensor(scipy.sparse.csr_matrix(adata.X, dtype="float32").todense()), 
		             edge_index=torch.tensor(edges[["Source","Target"]].astype(int).to_numpy().T),
		             edge_weight=torch.tensor(edges["Weight"].astype("float32").to_numpy()),
		             y=torch.from_numpy(adata.obs["target"].to_numpy().astype(int)).type(torch.LongTensor))
	mydata.num_features = mydata.x.shape[1]
	mydata.num_classes = len(set(np.array(mydata.y)))
	mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.2)(mydata)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if gnn=="GCN":
		study = hpo.run_HPO_GCN(mydata, hpo_name)
	elif gnn=="GAT":
		study = hpo.run_HPO_GAT(mydata, hpo_name)

	with open(f"{hpo_name}.json", "w") as f:
		json.dump(study.best_params, f)	
	best_params = study.best_params
else:
	best_params = json.load(open(f"{hpo_name}.json", "r"))


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Runs", flush=True)					 		
for run in range(0, 50):

	print("\n\n", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Starting run {run}/50 with {rep}, {gnn} and k-NN", flush=True)
	
	final_path=f"{path}/Run{run}"
	if os.path.isfile(f"{final_path}/{name}_XAITop{str(n_feat)}Features_Jaccard.tsv.gz") == False: 
		Path(final_path).mkdir(parents=True, exist_ok=True)	
			
		epi.pp.neighbors(adata, use_rep=f"X_{rep}", method="umap")
		edges = pd.DataFrame(adata.obsp["connectivities"].todense()).rename_axis('Source').reset_index().melt('Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)
		edges = edges[edges["Weight"]!=0]

		mydata=torch_geometric.data.Data(x=torch.tensor(scipy.sparse.csr_matrix(adata.X, dtype="float32").todense()), 
						 edge_index=torch.tensor(edges[["Source","Target"]].astype(int).to_numpy().T),
						 edge_weight=torch.tensor(edges["Weight"].astype("float32").to_numpy()),
						 y=torch.from_numpy(adata.obs["target"].to_numpy().astype(int)).type(torch.LongTensor))
		mydata.num_features=mydata.x.shape[1]
		mydata.num_classes=len(set(np.array(mydata.y)))
		mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.2)(mydata)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			
		if gnn == "GCN":
			model = mod.GCN(n_feats=mydata.num_features, n_classes=mydata.num_classes, hidden_dim=best_params["hidden_dim"]).to(device)
		elif gnn == "GAT":
			model = mod.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
						
		optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
		class_weights=sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(mydata.y), y=mydata.y.numpy())
		criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")
				
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Training {gnn} with {rep} and k-NN", flush=True)					 	
		model, data = mlu.GNN_train_node_classifier(model, mydata, optimizer_model, criterion, f"{final_path}/{name}_Model.pth", gnn)

		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Explaining {gnn} with {rep} and k-NN", flush=True)					 			   
		explainer = torch_geometric.explain.Explainer(
					model=model,
					algorithm=torch_geometric.explain.GNNExplainer(epochs=300),
				explanation_type='model',
				node_mask_type='attributes',
				edge_mask_type='object',
				model_config=dict(
					mode='multiclass_classification',
					task_level='node',
					return_type='probs',))
		
		if gnn == "GCN":
			explanation=explainer(x=mydata.x, edge_index=mydata.edge_index, edge_weight=mydata.edge_weight)
		elif gnn == "GAT":
			explanation=explainer(x=mydata.x, edge_index=mydata.edge_index)

		
		print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Saving results for {gnn} with {rep} and k-NN", flush=True)					 			

		with open(f"{final_path}/{name}_Model_Progress.json", "w") as f:
			json.dump(data, f)
		del data
		
		model.eval()
		if gnn == "GCN":
			pred = model(mydata.x, mydata.edge_index, mydata.edge_weight).argmax(dim=1)
		elif gnn == "GAT":
			pred = model(mydata.x, mydata.edge_index).argmax(dim=1)
			
		adata.obs["GNN_set"]="--"
		adata.obs["GNN_prediction"]=[adata.uns["inv_map"][str(num)] for num in list(pred.cpu().detach().numpy())]
		adata.obs.loc[mydata.train_mask.cpu().detach().numpy(),"GNN_set"]="Train"
		adata.obs.loc[mydata.val_mask.cpu().detach().numpy(),"GNN_set"]="Validation"
		adata.obs.loc[mydata.test_mask.cpu().detach().numpy(),"GNN_set"]="Test"
		adata.obs.to_csv(f"{final_path}/{name}_Predictions.tsv.gz", sep="\t", compression="gzip")
							
		a=pd.DataFrame(explanation.node_mask, index=adata.obs.index, columns=adata.var.index)
		df_feat=pd.DataFrame()
		df_imp=pd.DataFrame()
		for ct in sorted(list(set(adata.obs["target"]))):
			b=pd.DataFrame(a.loc[adata.obs[adata.obs["target"]==ct].index])
			df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False)[:int(n_feat)].index)], axis=1)
			df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False)[:int(n_feat)])[0]], axis=1)
		df_feat.columns=sorted(list(set(adata.obs["target"])))
		df_imp.columns=sorted(list(set(adata.obs["target"])))

		a.to_csv(f"{final_path}/{name}_XAIFeatImpCM.tsv.gz", sep="\t", compression="gzip")
		df_feat.to_csv(f"{final_path}/{name}_XAITop{str(n_feat)}Features.tsv.gz", sep="\t", compression="gzip")
		df_imp.to_csv(f"{final_path}/{name}_XAITop{str(n_feat)}FeaturesProbs.tsv.gz", sep="\t", compression="gzip")

		try:
			unf=torch_geometric.explain.unfaithfulness(explainer, explanation)
			fid=torch_geometric.explain.fidelity(explainer, explanation)
			chr_sc=torch_geometric.explain.characterization_score(fid[0], fid[1])
			met=pd.DataFrame(data=[unf, fid[0], fid[1], chr_sc], index=["unfaithfulness","Fidelity_pos","Fidelity_neg","characterization_score"]).T
			met.to_csv(f"{final_path}/{name}_XAITop{str(n_feat)}Features_ExplainerMetrics.tsv.gz", sep="\t", compression="gzip")
		except:
			continue
			
		jc=pd.DataFrame(index=df_feat.columns, columns=df_feat.columns)
		for column in jc.columns:
			for col in jc.columns:
				if len(df_feat[column].dropna())==0 or len(df_feat[col].dropna())==0:
					print("Problem with either {column} or {col}")
				else:	
					jc.at[column, col]=len(ut.intersection([df_feat[column].dropna(), df_feat[col].dropna()]))/len(ut.flat_list([df_feat[column].dropna(), df_feat[col].dropna()]))
		jc.to_csv(f"{final_path}/{name}_XAITop{str(n_feat)}Features_Jaccard.tsv.gz", sep="\t", compression="gzip")
	else:
		print("\n\n", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run {run}/50 with {rep}, {gnn} and k-NN is already DONE", flush=True)	
