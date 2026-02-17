import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx

import time
import scipy

import sklearn

from pathlib import Path
from functools import partial

import Utils as ut
import Models as mod
import ML_utils as mlu

import optuna
import torch
import torch_geometric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ++++++++++++++++++++++++++++++ GNN
def HPO_TrainModel_GNN(model, data, model_name, trial, param, GNN, epochs=300):
	if not (GNN == "GCN" or GNN == "GAT"):
		raise ValueError(f'Parameter GNN cannot be {GNN}')
		
	class_weights=sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(data.y), y=data.y.numpy())
	criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
	
	patience = 30
	best_val_f1w = -1
	best_epoch = -1

	train_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[3,2], batch_size=128, directed=False, shuffle=True)
	val_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[3,2], batch_size=64, directed=False, shuffle=True)
	
	for epoch in range(1, epochs + 1):
		### Training
		train_loss, train_f1w = mlu.GNN_1_step_training(model, train_loader, optimizer, criterion, GNN)

		### Validation
		val_loss, val_f1w = mlu.GNN_validation(model, val_loader, optimizer, criterion, GNN)

		### Early stopping
		if val_f1w > best_val_f1w:
			best_val_f1w = val_f1w
			best_epoch = epoch
			torch.save(model.state_dict(), f"{model_name}.pth")
			print(f"Checkpoint! Best epoch {best_epoch} | Best val loss {val_loss:.3f} | Best val F1W {best_val_f1w:.3f}", flush=True)
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(f"{model_name}.pth"))
			val_loss, best_val_f1w = mlu.GNN_validation(model, val_loader, optimizer, criterion, GNN)
			print(f"Early stopped at epoch {epoch} with best epoch {best_epoch} | Best val loss: {val_loss:.3f} | Best val F1W: {best_val_f1w:.3f}", flush=True)
			trial.report(best_val_f1w, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			return best_val_f1w

		trial.report(val_f1w, epoch)
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	model.load_state_dict(torch.load(f"{model_name}.pth"))
	val_loss, best_val_f1w = mlu.GNN_validation(model, val_loader, optimizer, criterion, GNN)
	
	trial.report(best_val_f1w, epoch)
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()

	return best_val_f1w

def build_GCN(trial, data):
	hidden_dim = trial.suggest_int('hidden_dim', low=32, high=256, step=32)
	model = mod.GCN(n_feats=data.num_features, n_classes=data.num_classes, hidden_dim=hidden_dim).to(device)
	return model

def objective_GCN(trial, data, model, model_name):
	params = {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), 'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)}
	model = build_GCN(trial, data)
	f1w = HPO_TrainModel_GNN(model=model, data=data, model_name=model_name, trial=trial, param=params, GNN="GCN")
	
	return f1w

def run_HPO_GCN(data, model_name):
	model=partial(build_GCN, data = data)
	obejctive=partial(objective_GCN, data = data, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), 
								study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=25, n_jobs=1, gc_after_trial=True)
	
	return study

def build_GAT(trial, data):
	dim_h = trial.suggest_int('dim_h', low=32, high=256, step=32)
	heads = trial.suggest_int('heads', low=4, high=12, step=2)
	model = mod.GAT(n_feats=data.num_features, n_classes=data.num_classes, dim_h=dim_h, heads=heads).to(device)
	
	return model

def objective_GAT(trial, data, model, model_name):
	params = {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), 'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)}
	model = build_GAT(trial, data)
	f1w = HPO_TrainModel_GNN(model=model, data=data, model_name=model_name, trial=trial, param=params, GNN="GAT")
	
	return f1w

def run_HPO_GAT(data, model_name):
	model=partial(build_GAT, data = data)
	obejctive=partial(objective_GAT, data = data, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), 
							study_name=model_name, storage=storage_name, load_if_exists=False)
	study.optimize(obejctive, n_trials=25, n_jobs=1, gc_after_trial=True)
	
	return study




# ++++++++++++++++++++++++++++++ TAE
def HPO_train_TAE(model, M, model_name, trial, param, y=[], epochs=300):
	if len(y) == 0:
		y=np.ones(shape=(M.shape[0],))
	y=np.array(y).astype(int)
	train_dataloader, val_dataloader = mlu.split_train_val(M, y)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
	
	best_val_loss = 10e10
	best_epoch = -1
	patience = 30
	torch.save(model.state_dict(), f"{model_name}.pth")
	
	for epoch in range(0, epochs):
		val_losses=[]
		for X in train_dataloader:
			X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
			train_loss = model(X)
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

		for X in val_dataloader:
			X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
			val_losses.append(model(X).item())
		val_loss=np.mean(val_losses)
		
		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"TAE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(model.state_dict(), f"{model_name}.pth")
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(f"{model_name}.pth"))
			val_losses=[]
			for X in val_dataloader:
				X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
				val_losses.append(model(X).item())
			best_val_loss=np.mean(val_losses)
			print(f"TAE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			trial.report(best_val_loss, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			return best_val_loss
		
		trial.report(val_loss, epoch)

		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	model.load_state_dict(torch.load(f"{model_name}.pth"))
	val_losses=[]
	for X,y in val_dataloader:
		X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
		val_losses.append(model(X).item())
	best_val_loss=np.mean(val_losses)
	trial.report(best_val_loss, epoch)
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()
	return best_val_loss

def build_TAE(trial, M):

	ae_kwargs={}
	ae_kwargs["input_dim"]=M.shape[1]
	ae_kwargs["hidden_dim"]=int(M.shape[1]**(1/2))
	ae_kwargs["latent_dim"]=int(M.shape[1]**(1/3))
	ae_kwargs["dp"]=trial.suggest_uniform('dp', low=0.1, high=0.7)

	lam = trial.suggest_int('lam', low=1, high=30, step=3)
	p = trial.suggest_int('p', low=1, high=5, step=1)

	model = mod.TopologicallyRegularizedAutoencoder(ae_kwargs=ae_kwargs, lam=lam, p=p)
	
	return model

def objective_TAE(trial, M, y, model, model_name):
	params = {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), 'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)}
	model = build_TAE(trial, M)
	val_loss = HPO_train_TAE(model=model, M=M, y=y, model_name=model_name, trial=trial, param=params)
	
	return val_loss

def run_HPO_TAE(M, y, model_name):
	model=partial(build_TAE, M = M)
	obejctive=partial(objective_TAE, M = M, y=y, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),
								study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=25, n_jobs=1, gc_after_trial=True)
	return study

# ++++++++++++++++++++++++++++++ VAE

def HPO_train_VAE(model, M, model_name, trial, param, y=[], epochs=300):
	if len(y) == 0:
		dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
		del M
		train_set, val_set = torch.utils.data.random_split(dataset, [0.85, 0.15])
		train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
		val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
	else:
		train_dataloader, val_dataloader = mlu.split_train_val(M, y)
    
	optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
	loss_fn=torch.nn.MSELoss()
	
	best_val_loss = 10e10
	best_epoch = -1
	patience = 30
	
	annealer = mod.KLAnnealer()
	global_step = 0

	for epoch in range(0, epochs):
		val_losses=[]
		for X in train_dataloader:
			global_step += 1
			beta = annealer.step()
			X_hat, mu, log_var = model(X)
			reconstruction_loss = loss_fn(X, X_hat)
			kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
			train_loss = reconstruction_loss + beta * kl_loss
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

		for X in val_dataloader:
			X_hat, mu, log_var = model(X)
			reconstruction_loss = loss_fn(X, X_hat)
			kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
			beta = annealer.beta(global_step)   # use current beta
			tot_loss = reconstruction_loss + beta * kl_loss
			val_losses.append(tot_loss.item())
		val_loss=np.mean(val_losses)

		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"VAE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(model.state_dict(), f"{model_name}.pth")
		
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(f"{model_name}.pth"))
			val_losses=[]
			for X in val_dataloader:
				X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
				X_hat, mu, log_var = model(X)
				reconstruction_loss = loss_fn(X, X_hat)
				kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
				beta = annealer.beta(global_step)   # use current beta
				tot_loss = reconstruction_loss + beta * kl_loss
				val_losses.append(tot_loss.item())
			best_val_loss=np.mean(val_losses)
			print(f"VAE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			trial.report(best_val_loss, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			
			return best_val_loss
		
		trial.report(val_loss, epoch)

		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	model.load_state_dict(torch.load(f"{model_name}.pth"))
	val_losses=[]
	for X,y in val_dataloader:
		X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
		X_hat, mu, log_var = model(X)
		reconstruction_loss = loss_fn(X, X_hat)
		kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim = 1), dim = 0)
		beta = annealer.beta(global_step)   # use current beta
		tot_loss = reconstruction_loss + beta * kl_loss
		val_losses.append(tot_loss.item())
	best_val_loss=np.mean(val_losses)
	trial.report(best_val_loss, epoch)
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()
		
	return best_val_loss

def build_VAE(trial, M):
	dp = trial.suggest_uniform('dp', low=0.1, high=0.8)
	ae_kwargs = {"input_dim" : M.shape[1], 
				 "hidden_dim" : int(M.shape[1]**(1/2)), 
				 "latent_dim"  : int(M.shape[1]**(1/3)),
				 "dp" : dp}

	AE_model = mod.VAutoencoder(ae_kwargs=ae_kwargs)
	
	return AE_model

def objective_VAE(trial, M, y, model, model_name):
	params = {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), 
			  'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)}
	model = build_VAE(trial, M)
	val_loss = HPO_train_VAE(model=model, M=M, model_name=model_name, trial=trial, param=params)

	return val_loss

def run_HPO_VAE(M, y, model_name):
	model = partial(build_VAE, M=M)
	obejctive = partial(objective_VAE, M=M, y=y, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),
								study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=25, n_jobs=1, gc_after_trial=True)
	return study


# ++++++++++++++++++++++++++++++ AE

def HPO_train_AE(model, M, model_name, trial, param, y=[], epochs=300):
	if len(y) == 0:
		dataset = torch.tensor(scipy.sparse.csr_matrix(M, dtype="float32").todense())
		del M
		train_set, val_set = torch.utils.data.random_split(dataset, [0.85, 0.15])
		train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
		val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
	else:
		train_dataloader, val_dataloader = mlu.split_train_val(M, y)
    
	optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
	loss_fn=torch.nn.MSELoss()
	
	best_val_loss = 10e10
	best_epoch = -1
	patience = 30
	
	for epoch in range(0, epochs):
		val_losses=[]
		for X in train_dataloader:
			X_hat = model(X)
			reconstruction_loss = loss_fn(X, X_hat)
			optimizer.zero_grad()
			reconstruction_loss.backward()
			optimizer.step()

		for X in val_dataloader:
			X_hat = model(X)
			reconstruction_loss = loss_fn(X, X_hat)
			val_losses.append(reconstruction_loss.item())
		val_loss=np.mean(val_losses)

		if np.around(val_loss/best_val_loss, decimals=2) < 0.95:
			best_val_loss = val_loss
			best_epoch = epoch
			print(f"AE checkpoint! Best epoch {best_epoch} | Best loss {best_val_loss:.7f}", flush=True)
			torch.save(model.state_dict(), f"{model_name}.pth")
		
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(f"{model_name}.pth"))
			val_losses=[]
			for X in val_dataloader:
				X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
				X_hat = model(X)
				reconstruction_loss = loss_fn(X, X_hat)
				val_losses.append(reconstruction_loss.item())
			best_val_loss=np.mean(val_losses)
			print(f"AE early stopped at epoch {epoch} with best epoch {best_epoch} | Best loss: {best_val_loss:.7f}", flush=True)
			trial.report(best_val_loss, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			
			return best_val_loss
		
		trial.report(val_loss, epoch)

		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	model.load_state_dict(torch.load(f"{model_name}.pth"))
	val_losses=[]
	for X,y in val_dataloader:
		X=torch.tensor(scipy.sparse.csr_matrix(X[0], dtype="float32").todense()).to(device)
		X_hat = model(X)
		reconstruction_loss = loss_fn(X, X_hat)
		val_losses.append(reconstruction_loss.item())
	best_val_loss=np.mean(val_losses)
	trial.report(best_val_loss, epoch)
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()
		
	return best_val_loss

def build_AE(trial, M):
	dp = trial.suggest_uniform('dp', low=0.1, high=0.8)
	AE_model = mod.BaseAE(input_dim=M.shape[1], hidden_dim=int(M.shape[1]**(1/2)), latent_dim=int(M.shape[1]**(1/3)), dp=dp)
	
	return AE_model

def objective_AE(trial, M, y, model, model_name):
	params = {'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), 
			  'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)}
	model = build_AE(trial, M)
	val_loss = HPO_train_AE(model=model, M=M, model_name=model_name, trial=trial, param=params)

	return val_loss

def run_HPO_AE(M, y, model_name):
	model = partial(build_AE, M=M)
	obejctive = partial(objective_AE, M=M, y=y, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),
								study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=25, n_jobs=1, gc_after_trial=True)
	return study
