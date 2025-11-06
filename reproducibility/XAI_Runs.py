import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import pandas as pd

import sys
import time
import scipy
import json

import torch
import torch_geometric

import Utils as ut
import Models as mod

from pathlib import Path

torch.manual_seed(np.random.randint(0,10000))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")
print( time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI script started", flush=True)

path = sys.argv[1]
name = sys.argv[2]
matrix = sys.argv[3]
label = sys.argv[4]
graph = sys.argv[5]

names = ["Script -->", "Path -->", "Name -->", "Matrix -->", "Label -->","Graph -->", "HPO -->"]
for n,arg in zip(names,sys.argv):
	print(n, arg, flush=True)

Path(f"{path}/{label}").mkdir(parents=True, exist_ok=True)
Path(f"{path}/DE").mkdir(parents=True, exist_ok=True)
de_path = f"{path}/DE/{name}"

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Creating dataset", flush=True)
adata = sc.read(matrix)
adata=adata[adata.obs[label].dropna().index]

mymap = dict([(y,str(x)) for x,y in enumerate(sorted(set(adata.obs[label])))])
inv_map = {v: k for k, v in mymap.items()}
adata.uns["map"]=mymap
adata.uns["inv_map"]=inv_map
adata.obs["target"]=[mymap[x] for x in adata.obs[label]]

edges = pd.DataFrame(adata.obsp[str(graph)].todense()).rename_axis('Source')\
	.reset_index()\
	.melt('Source', value_name='Weight', var_name='Target')\
	.query('Source != Target')\
	.reset_index(drop=True)
edges = edges[edges["Weight"]!=0]

mydata = torch_geometric.data.Data(x=torch.tensor(scipy.sparse.csr_matrix(adata.X, dtype="float32").todense()), 
									edge_index=torch.tensor(edges[["Source","Target"]].astype(int).to_numpy().T),
									edge_weight=torch.tensor(edges["Weight"].astype("float32").to_numpy()),
									y=torch.from_numpy(adata.obs["target"].to_numpy().astype(int)).type(torch.LongTensor))
mydata.num_features = mydata.x.shape[1]
mydata.num_classes = len(set(np.array(mydata.y)))

xai_path = f"{path}/{name}_HPO"
best_params = json.load(open(f"{xai_path}.json", "r"))
model = mod.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
model.load_state_dict(torch.load(f"{xai_path}_Model.pth"))

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run XAI experiments", flush=True)
dfs=[]
n_runs=50
columns=sorted(list(set(adata.obs[label])))

for n in range(n_runs):
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"	--> Run {n}/{n_runs}", flush=True)
	torch.manual_seed(np.random.randint(0,10000))
	explainer = torch_geometric.explain.Explainer(
			model=model,
			algorithm=torch_geometric.explain.GNNExplainer(epochs=300),
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
	for ct in columns:
		b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
		df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False).to_numpy())[0]], axis=1)
		df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False).index)], axis=1)
	df_feat.columns=columns
	df_imp.columns=columns

	Path(f"{path}/Runs/Run{n}/").mkdir(parents=True, exist_ok=True)
	a.to_csv(f"{path}/Runs/Run{n}/{name}_HPO_XAIFeatImpCM.tsv.gz", sep="\t", compression="gzip")
	df_feat.to_csv(f"{path}/Runs/Run{n}/{name}_HPO_XAIFeatures.tsv.gz", sep="\t", compression="gzip")
	df_imp.to_csv(f"{path}/Runs/Run{n}/{name}_HPO_XAIFeaturesImportance.tsv.gz", sep="\t", compression="gzip")
	dfs.append(df_feat)

dfs=[pd.read_csv(f"{path}/Runs/Run{n}/{name}_HPO_XAIFeatures.tsv.gz", sep="\t", index_col=0) for n in range(n_runs)]

expls={}
print(list(dfs[0].columns))
for col in dfs[0].columns:
	expls[col]=[list(df[col]) for df in dfs]

thres=[10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 500, 750, 1000, 1500, 2000]
for th in thres:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI Stability at {th}", flush=True)
	for key in expls.keys():
		t=pd.DataFrame(expls[key]).T[:th].T
		table=pd.DataFrame(index=range(0, n_runs), columns=range(0, n_runs))
		for n in range(0, n_runs):
			for k in range(n, n_runs):
				table.at[n,k]=len(ut.intersection([t.iloc[n],t.iloc[k]]))
				table.at[k,n]=table.at[n,k]
		Path(f"{path}/Runs/CTS/{key}").mkdir(parents=True, exist_ok=True)
		table.to_csv(f"{path}/Runs/CTS/{key}/{name}_{key}_{th}_Stability.tsv.gz", sep="\t", compression="gzip")

for th in thres:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI Specificity at {th}", flush=True)
	for key in list(expls.keys()):
		a=list(expls.keys())
		a.remove(key)
		df_spec=pd.DataFrame(index=range(0, n_runs), columns=[key])
		for n in range(0, n_runs):
			df_spec.at[n, key]=np.mean([len(ut.intersection([dfs[n][key][:th], dfs[n][ct][:th]])) for ct in a])
		df_spec.to_csv(f"{path}/Runs/CTS/{key}/{name}_{key}_{th}_Specificity.tsv.gz", sep="\t", compression="gzip")
