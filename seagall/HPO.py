import warnings
warnings.filterwarnings("ignore")

import numpy as np

import sklearn

from functools import partial

from . import ML_utils as mlu
from . import Models as mod

import optuna
import torch
import torch_geometric

from .base_dataset import DEVICE, logger
# ++++++++++++++++++++++++++++++ GAT
def HPO_TrainModel_GAT(model, data, model_name, trial, param):

	'''
	Operative function to apply HPO

	Parameters
	----------
	model : object of class torch.nn.Module

	data : object of class torch_geometric.data.Data storing thw whole count matrix. See EmbeddExplain.classify_and_explain to see how the data object is built

	model_name : name to use to save the model during the training

	trial : object of class optuna.trial.Trial to evaluate the objective function	

	param :  dictinary containg the hyperparameters to study and their range of plausible values

	Output
	------
	
	Retruns F1 score

	'''	
		
	class_weights=sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data.y), y=data.y.numpy())
	criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float), reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
	
	epochs=100

	train_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[3,2], batch_size=128, directed=False, shuffle=True)
	val_loader = torch_geometric.loader.NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[3,2], batch_size=64, directed=False, shuffle=True)
	
	for epoch in range(1, epochs + 1):
		### Training
		train_loss, train_f1 = mlu.GAT_1_step_training(model, train_loader, optimizer, criterion)

		### Validation
		val_loss, val_f1 = mlu.GAT_validation(model, val_loader, criterion)

		trial.report(val_f1, epoch)
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	return val_f1

def build_GAT(trial, data):

	'''
	Define model to which apply HPO

	Parameters
	----------

	trial : object of class optuna.trial.Trial to evaluate the objective function

	data : object of class torch_geometric.data.Data storing thw whole count matrix. See EmbeddExplain.classify_and_explain to see how the data object is built


	Output
	------
	
	Retruns object of class torch.nn.Module

	'''	

	dim_h = trial.suggest_int('dim_h', low=32, high=512, step=32)
	heads = trial.suggest_int('heads', low=4, high=20, step=2)
	dropout = trial.suggest_float('dropout', low=0.1, high=0.7, step=0.1)
	model = mod.GAT(n_feats=data.num_features, n_classes=data.num_classes, dim_h=dim_h, heads=heads, dropout=dropout).to(DEVICE)
	
	return model

def objective_GAT(trial, data, model, model_name):

	'''
	Define objective function to maximise. For a classification problem is the weighted-F1

	Parameters
	----------

	trial : object of class optuna.trial.Trial to evaluate the objective function

	data : object of class torch_geometric.data.Data storing thw whole count matrix. See EmbeddExplain.classify_and_explain to see how the data object is built

	model : object of class torch.nn.Module

	model_name : name to use to save the model during the training
	
	Output
	------
	
	Retruns F1 score

	'''	

	params = {'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True), 'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)}
	model = build_GAT(trial, data)
	f1 = HPO_TrainModel_GAT(model=model, data=data, model_name=model_name, trial=trial, param=params)
	
	return f1

def run_HPO_GAT(data, model_name):

	'''
	Apply HPO to GAT model

 	Parameters
	----------

	data : object of class torch_geometric.data.Data storing thw whole count matrix. See Ml_utils.create_pyg_dataset for the building of the data object
	model_name : name to use to save the model during the training

	Output
	------
	
	Returns object of class optuna.study.Study see https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
	
	'''

	model=partial(build_GAT, data=data)
	obejctive=partial(objective_GAT, data = data, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), 
							study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=30, n_jobs=2, gc_after_trial=True)

	for key, value in study.best_trial.params.items():
		logger.info(f"Best value for {key} is {value}")	

	return study



