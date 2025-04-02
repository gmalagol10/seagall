import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

import scipy
import random
import sklearn

import torch
import torch_geometric
device = 'cpu'

from . import Utils as ut

class GAT(torch.nn.Module):

	'''
	Class to apply Graph Attention Network based on https://arxiv.org/abs/2105.14491 paper
	See https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GATv2Conv.html for the details 

 	Parameters
	----------

	n_classes : number of classes in the dataset
	
	dim_h : dimension of the hidden layer

	heads : number of multi-head attention


	Output
	------
	
	Return the transformed input vector (x)

	'''

	def __init__(self, n_feats, n_classes, dim_h=64, heads=8, dropout=0.5):
		super().__init__()
		self.gat1 = torch_geometric.nn.GATv2Conv(n_feats, dim_h, heads=heads, dropout=dropout)
		self.gat2 = torch_geometric.nn.GATv2Conv(dim_h*heads, n_classes, dropout=dropout, heads=1)

	def forward(self, x, edge_index):
		x = self.gat1(x, edge_index)
		x = torch.nn.functional.relu(x)
		x = self.gat2(x, edge_index)
		
		return x


def split_train_val(X, y, train_size=0.85, val_size=0.15, train_batch_size=128, val_batch_size=64):

	'''
	Function to split the dataset in train and validation set with batches

 	Parameters
	----------

	X : N * F matrix with N cells and F features
	
	y : target label, important for the train-val-split of cells accounting for label unbalance

	train_size : fraction of elements to put in the train set

	val_size : fraction of elements to put in the validation set

	train_batch_size: number of elements to store in each batch of the training set

	val_batch_size: number of elements to store in each batch of the validation set


	Output
	------
	
	Return two data loader of class torch.utils.data.DataLoader, for training and validation set
	
	'''

	X=scipy.sparse.csr_matrix(X, dtype="float32").todense()		 
	y=np.array(y).astype(int)

	X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=train_size, test_size=val_size, random_state=42, stratify=y)
	y_train=y_train.astype(int)	
	y_val=y_val.astype(int)	

	class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))	
	train_dataloader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=train_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	 

	class_sample_count = np.array([len(np.where(y_val==t)[0]) for t in np.unique(y_val)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_val])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

	valDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
	val_dataloader = torch.utils.data.DataLoader(dataset = valDataset, batch_size=val_batch_size, shuffle=False, num_workers=1, sampler = sampler)

	return train_dataloader, val_dataloader

def split_train_val_test(X, y, train_size=0.7, val_size=0.1, test_size=0.2, train_batch_size=128, valtest_batch_size=64):

	'''
	Function to split the dataset in train, validation and test set with batches for the training and the validation

 	Parameters
	----------

	X : N * F matrix with N cells and F features
	
	y : target label, important for the train-val-split of cells accounting for label unbalance

	train_size : fraction of elements to put in the train set

	val_size : fraction of elements to put in the validation set

	test_size : fraction of elements to put in the test set

	train_batch_size: number of elements to store in each batch of the training set

	val_batch_size: number of elements to store in each batch of the validation set


	Output
	------
	
	Return two data loader of class torch.utils.data.DataLoader, for training and validation set
	
	'''

	if type(X) == scipy.sparse._csr.csr_matrix:
		X=X.todense()
	y=np.array(y).astype(int)

	X_tv, X_test, y_tv, y_test = sklearn.model_selection.train_test_split(X, y, train_size=train_size+val_size, random_state=42, stratify=y)
	y_tv=y_tv.astype(int)
	y_test=y_test.astype(int)

	class_sample_count = np.array([len(np.where(y_test==t)[0]) for t in np.unique(y_test)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_test])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	testDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
	test_dataloader = torch.utils.data.DataLoader(dataset = testDataset, batch_size=valtest_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	
	
	X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_tv, y_tv, train_size=train_size/(train_size + val_size), random_state=42, stratify=y_tv)
	y_train=y_train.astype(int)
	y_val=y_val.astype(int)

	class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
	
	trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
	train_dataloader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=train_batch_size, shuffle=False, num_workers=1, sampler = sampler)
	 

	class_sample_count = np.array([len(np.where(y_val==t)[0]) for t in np.unique(y_val)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in y_val])
	samples_weight = torch.from_numpy(samples_weight)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

	valDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
	val_dataloader = torch.utils.data.DataLoader(dataset = valDataset, batch_size=valtest_batch_size, shuffle=False, num_workers=1, sampler = sampler)

	return train_dataloader, val_dataloader, test_dataloader

def create_pyg_dataset(adata, label, size=1):
	print(f"Creating pyg dataset based on AnnData object with target label {label} and GRAE's graph. Using {int(size*100)}% of the cells", flush=True)
	if size < 1 and size > 0:
		cells=[]
		for l in set(adata.obs[label].dropna()):
			sub_ad=adata[adata.obs[label]==l]
			sc.pp.subsample(sub_ad, fraction=size)
			cells.append(sub_ad.obs.index)
			del sub_ad
		cells=ut.flat_list(cells)
		ad=adata[cells]
	elif size == 0:
		raise ValueError(f"Size can't be {size}")
	else:
		ad=adata.copy()
	
	edges = pd.DataFrame(ad.obsp["GRAE_graph"].toarray()).rename_axis('Source').reset_index().melt('Source', value_name='Weight', var_name='Target')\
		.query('Source != Target').reset_index(drop=True)
	edges = edges[edges["Weight"]!=0]
	
	mydata = torch_geometric.data.Data(x=torch.tensor(scipy.sparse.csr_matrix(ad.X, dtype="float32").toarray()), 
							 edge_index=torch.tensor(edges[["Source","Target"]].astype(int).to_numpy().T),
							 y=torch.from_numpy(ad.obs["target"].to_numpy().astype(int)).type(torch.LongTensor))
	mydata.num_features = mydata.x.shape[1]
	mydata.num_classes = len(set(np.array(mydata.y)))
	del ad
	return mydata
	
# ++++++++++++++++++++++++++++++ TRAIN GATs

def GAT_1_step_training(model, train_loader, optimizer, criterion):
		
	'''
	Apply one iteration of GAT's training

 	Parameters
	----------

	model : object of class torch.nn.Module
	
	train_loader : object of class torch.utils.data.DataLoader storing the elements of the training set

	optimizer : object from the model torch.optim 

	criterion : loss function to apply (e.g. torch.nn.CrossEntropyLoss)

	Output
	------
	
	Return loss and F1 score computed on the whole training set batch by batch

	'''

	device = 'cpu'
	train_loss=0
	train_f1=0
	for batch in train_loader:
		optimizer.zero_grad()
		batch = batch.to(device)
		out = model(batch.x, batch.edge_index)[:batch.batch_size].to(device)

		# NOTE Only consider predictions and labels of seed nodes:
		y = batch.y[:batch.batch_size].to(device)
		loss_batch = criterion(out, y)
		loss_batch.backward()
		optimizer.step()
		train_loss+=loss_batch.item()
		train_f1+=sklearn.metrics.precision_recall_fscore_support(y.detach().numpy(), out.argmax(dim=1).detach().numpy(), average="macro")[2]
	
	return train_loss/len(train_loader), train_f1/len(train_loader)

def GAT_validation(model, val_loader, optimizer, criterion):

	'''
	Apply one iteration of GAT's validation

 	Parameters
	----------

	model : object of class torch.nn.Module
	
	val_loader : object of class torch.utils.data.DataLoader storing the elements of the validation set

	optimizer : object from the model torch.optim 

	criterion : loss function to apply (e.g. torch.nn.CrossEntropyLoss)

	Output
	------
	
	Return loss and F1 score computed on the whole validation set batch by batch
	
	'''
		
	device = 'cpu'
	val_f1=0
	val_loss=0
	model.eval()
	with torch.no_grad(): 
		for batch in val_loader:
			batch = batch.to(device)
			out = model(batch.x, batch.edge_index)[:batch.batch_size].to(device)
		
			y = batch.y[:batch.batch_size].to(device)
			loss_batch = criterion(out, y)
			val_loss += loss_batch.item()
			val_f1 += sklearn.metrics.precision_recall_fscore_support(y.detach().numpy(), out.argmax(dim=1).detach().numpy(), average="macro")[2]
	
	return val_loss/len(val_loader), val_f1/len(val_loader)
	
def GAT_train_node_classifier(model, data, optimizer, criterion, model_name, epochs=250, patience=30):

	'''
	Complete funciton to train the GAT

 	Parameters
	----------

	model : object of class torch.nn.Module
	
	data : object of class torch_geometric.data.Data storing thw whole count matrix. See EmbeddExplain.classify_and_explain to see how the data object is built

	optimizer : object from the model torch.optim 

	criterion : loss function to apply (e.g. torch.nn.CrossEntropyLoss)

	model_name : name to use to save the model during the training

	epochs : maximum number ot epoch to train the model for

	patience : maximum number of epochs to wait when the loss function doesn't decrease

	Output
	------
	
	Return trained model and historical data about the trainig and validation

	'''

	best_val_f1 = -1
	best_epoch = -1
	history={}
	history["TrainLoss"]=[]
	history["TrainF1"]=[]
	history["ValLoss"]=[]
	history["ValF1"]=[]
	   
	train_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[3,2], batch_size=128, directed=False, shuffle=True)
	val_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[3,2], batch_size=64, directed=False, shuffle=True)
	
	for epoch in range(1, epochs + 1):
		### Training
		train_loss, train_f1 = GAT_1_step_training(model, train_loader, optimizer, criterion)
		history["TrainLoss"].append(np.around(train_loss, decimals=5))
		history["TrainF1"].append(np.around(train_f1, decimals=5))

		### Validation
		val_loss, val_f1 = GAT_validation(model, val_loader, optimizer, criterion)
		history["ValLoss"].append(np.around(val_loss, decimals=5))
		history["ValF1"].append(np.around(val_f1, decimals=5))

		### Early stopping
		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			best_epoch = epoch
			torch.save(model.state_dict(), model_name)
			print(f"GAT checkpoint! Best epoch {best_epoch} | Best val loss {val_loss:.3f} | Best val F1 {best_val_f1:.3f}", flush=True)
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(model_name))
			val_loss, best_val_f1 = GAT_validation(model, val_loader, optimizer, criterion)
			print(f"GAT early stopped at epoch {epoch} with best epoch {best_epoch} | Best val loss: {val_loss:.3f} | Best val F1: {best_val_f1:.3f}", flush=True)
			return model, history  
	
	model.load_state_dict(torch.load(model_name))
	return model, history

def specificty (adata, label, n_feat=50):
	gts = sorted(set(adata.obs[label]))
	spec = pd.DataFrame(index=gts, columns=gts)
	for gti in gts:
		for gtj in gts[gts.index(gti)+1:]:
		    fsi = adata[adata.obs[label]==gti].var[f"SEAGALL_Importance_for_{gti}"].sort_values()[::-1][:int(n_feat)].index
		    fsj = adata[adata.obs[label]==gtj].var[f"SEAGALL_Importance_for_{gtj}"].sort_values()[::-1][:int(n_feat)].index
		    spec.at[gti, gtj] = len(ut.intersection([fsi, fsj]))/len(ut.flat_list([fsi, fsj]))
	return spec
