import warnings
warnings.filterwarnings("ignore")

import numpy as np
import episcanpy as epi
import scanpy as sc
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
from captum.attr import IntegratedGradients

import ML_utils as mlu
import Utils as ut
import Models as mod
import HPO as hpo

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
hypopt = sys.argv[6]

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

if hypopt == "True":
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Starting HPO", flush=True)	
	xai_path = f"{path}/{name}_HPO"
	xai_label_path = f"{path}/{label}/{name}_HPO"

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
	model = mod.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(device)
	optimizer_model = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

else:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Skipping HPO", flush=True)	
	xai_path = f"{path}/{name}"
	xai_label_path = f"{path}/{label}/{name}"
	mydata = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(mydata)
	model = mod.GAT(n_feats=mydata.num_features, n_classes=mydata.num_classes).to(device)
	optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
print("XAI path", xai_path, flush=True)

class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(mydata.y), y=mydata.y.numpy())
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "Training model", flush=True)					
model, history = mlu.GNN_train_node_classifier(model, mydata, optimizer_model, criterion, f"{xai_path}_Model.pth", "GAT", epochs=300, patience=30)

with open(f"{xai_path}_Model_Progress.json", "w") as f:
	json.dump(history, f)
del history


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Metrics for model's performances", flush=True)				
model.eval()
pred = model(mydata.x, mydata.edge_index).argmax(dim=1)

adata.obs["GNN_set"]="--"
adata.obs["GNN_prediction"]=[adata.uns["inv_map"][str(num)] for num in list(pred.cpu().detach().numpy())]
adata.obs.loc[mydata.train_mask.cpu().detach().numpy(),"GNN_set"]="Train"
adata.obs.loc[mydata.val_mask.cpu().detach().numpy(),"GNN_set"]="Validation"
adata.obs.loc[mydata.test_mask.cpu().detach().numpy(),"GNN_set"]="Test"
adata.obs.to_csv(f"{xai_path}_Predictions.tsv.gz", sep="\t", compression="gzip")


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI features extraction", flush=True)
n_feat=50
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
for ct in sorted(list(set(adata.obs[label]))):
    print(ct)
    b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
    df_imp=pd.concat([df_imp, pd.DataFrame(b.mean().sort_values(ascending=False).to_numpy())[0]], axis=1)
    df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False).index)], axis=1)
df_feat.columns=sorted(list(set(adata.obs[label])))
df_imp.columns=sorted(list(set(adata.obs[label])))

a.to_csv(f"{xai_path}_XAIFeatImpCM.tsv.gz", sep="\t", compression="gzip")
df_feat.to_csv(f"{xai_path}_XAIFeatures.tsv.gz", sep="\t", compression="gzip")
df_imp.to_csv(f"{xai_path}_XAIFeaturesImportance.tsv.gz", sep="\t", compression="gzip")
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

a.to_csv(f"{xai_path}_XAITop{str(n_feat)}FeatImpCM.tsv.gz", sep="\t", compression="gzip")
df_feat.to_csv(f"{xai_path}_XAITop{str(n_feat)}Features.tsv.gz", sep="\t", compression="gzip")
df_imp.to_csv(f"{xai_path}_XAITop{str(n_feat)}FeaturesImportance.tsv.gz", sep="\t", compression="gzip")
del df_imp, a


jc=pd.DataFrame(index=df_feat.columns, columns=df_feat.columns)
for column in jc.columns:
	for col in jc.columns:
		if len(df_feat[column].dropna())==0 or len(df_feat[col].dropna())==0:
			print("Problem with either {column} or {col}")
		else:	
			jc.at[column, col]=len(ut.intersection([df_feat[column].dropna(), df_feat[col].dropna()]))/len(ut.flat_list([df_feat[column].dropna(), df_feat[col].dropna()]))
jc.to_csv(f"{xai_path}_XAITop{str(n_feat)}Features_Jaccard.tsv.gz", sep="\t", compression="gzip")	   	


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Run XAI experiments", flush=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']   
dfs=[]
n_runs=50
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
	for ct in sorted(list(set(adata.obs[label]))):
		b=pd.DataFrame(a.loc[adata.obs[adata.obs[label]==ct].index])
		df_feat=pd.concat([df_feat, pd.DataFrame(b.mean().sort_values(ascending=False)[:int(n_feat)].index)], axis=1)
	df_feat.columns=sorted(list(set(adata.obs[label])))
	dfs.append(df_feat)   	

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI Stability", flush=True)
expls={}
print(list(dfs[0].columns))
for col in dfs[0].columns:
	expls[col]=[list(df[col]) for df in dfs]

for key in expls.keys():
	t=pd.DataFrame(expls[key])
	table=pd.DataFrame(index=range(0, n_runs), columns=range(0, n_runs))
	for n in range(0, n_runs):
		for k in range(n, n_runs):
			table.at[n,k]=len(ut.intersection([t.iloc[n],t.iloc[k]]))
			table.at[k,n]=table.at[n,k]
	table.to_csv(f"{xai_label_path}_{key}_Stability.tsv.gz", sep="\t", compression="gzip")
del table, t, df_feat


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"XAI Specificity", flush=True)

for key in list(expls.keys()):
	a=list(expls.keys())
	a.remove(key)
	df_spec=pd.DataFrame(index=range(0, n_runs), columns=[key])
	for n in range(0, n_runs):
		df_spec.at[n, key]=np.mean([len(ut.intersection([dfs[n][key], dfs[n][ct]])) for ct in a])
	df_spec.to_csv(f"{xai_label_path}_{key}_Specificity.tsv.gz", sep="\t", compression="gzip")
del dfs, expls, df_spec


print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"DEA {n_feat}", flush=True)
adata.uns["log1p"]["base"]=None
epi.tl.rank_features(adata, label, use_raw=False, n_features=len(adata.var))
fts=pd.DataFrame(adata.uns["rank_features_groups"]["names"])
lfc=pd.DataFrame(adata.uns["rank_features_groups"]["logfoldchanges"])
pvs=pd.DataFrame(adata.uns["rank_features_groups"]["pvals_adj"])
fts=fts[(pvs<0.01) & (lfc > 0.5)]
lfc=lfc[(pvs<0.01) & (lfc > 0.5)]
pvs=pvs[(pvs<0.01) & (lfc > 0.5)]

fts.to_csv(f"{de_path}_DEFeatures.tsv.gz", sep="\t", compression="gzip")
lfc.to_csv(f"{de_path}_DEFeaturesLFC.tsv.gz", sep="\t", compression="gzip")
pvs.to_csv(f"{de_path}_DEFeaturesPvsAdj.tsv.gz", sep="\t", compression="gzip")

adata.uns["log1p"]["base"]=None
epi.tl.rank_features(adata, label, use_raw=False, n_features=int(n_feat))
fts=pd.DataFrame(adata.uns["rank_features_groups"]["names"])
lfc=pd.DataFrame(adata.uns["rank_features_groups"]["logfoldchanges"])
pvs=pd.DataFrame(adata.uns["rank_features_groups"]["pvals_adj"])
fts=fts[(pvs<0.01) & (lfc > 0.5)]
lfc=lfc[(pvs<0.01) & (lfc > 0.5)]
pvs=pvs[(pvs<0.01) & (lfc > 0.5)]

fts.to_csv(f"{de_path}_DETop{str(n_feat)}Features.tsv.gz", sep="\t", compression="gzip")
lfc.to_csv(f"{de_path}_DETop{str(n_feat)}FeaturesLFC.tsv.gz", sep="\t", compression="gzip")
pvs.to_csv(f"{de_path}_DETop{str(n_feat)}FeaturesPvsAdj.tsv.gz", sep="\t", compression="gzip")


##### NN
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Classify cell with NN", flush=True)					
model=mod.NN(n_feats=adata.shape[1], n_classes=len(set(adata.obs.target)))
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(np.array(adata.obs.target)), y=np.array(adata.obs.target))
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")
output = mlu.NN_train_classifier(model, adata, optimizer_model, criterion, f"{xai_path}_NN_Model.pth")

print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Explaining NN", flush=True)					
model = output[0]
baseline = torch.zeros(adata.shape[0], adata.shape[1])
inp=torch.FloatTensor(scipy.sparse.csr_matrix(adata.X, dtype="float32").toarray())
ig = IntegratedGradients(output[0])
attributions, delta = ig.attribute(inp, baseline, target=torch.LongTensor(adata.obs.target.astype(int)), return_convergence_delta=True, internal_batch_size=32)

df_feat=pd.DataFrame(data=attributions, columns=adata.var.index, index=adata.obs.index).T
df=pd.DataFrame()
for ct in set(adata.obs.target):
    cells=adata[adata.obs.target==ct].obs.index
    feats=list(df_feat[cells].mean(axis=1).sort_values()[::-1].index)
    to_append=pd.DataFrame(feats, columns=[adata.uns["inv_map"][ct]])
    df=pd.concat([df, to_append], axis=1)

df.to_csv(f"{xai_path}_NN_XAIFeatures.tsv.gz", sep="\t", compression="gzip")
