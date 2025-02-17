import warnings
warnings.filterwarnings("ignore")

import numpy as np

import sklearn

from functools import partial

from . import ML_utils as mlu

import optuna
import torch
import torch_geometric

device = 'cpu'
# ++++++++++++++++++++++++++++++ GAT
def HPO_TrainModel_GAT(model, data, model_name, trial, param, epochs=250):

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
	
	Retruns F1W score

	'''	
		
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
		train_loss, train_f1w = mlu.GAT_1_step_training(model, train_loader, optimizer, criterion)

		### Validation
		val_loss, val_f1w = mlu.GAT_validation(model, val_loader, optimizer, criterion)

		### Early stopping
		if val_f1w > best_val_f1w:
			best_val_f1w = val_f1w
			best_epoch = epoch
			torch.save(model.state_dict(), f"{model_name}.pth")
			print(f"Checkpoint! Best epoch {best_epoch} | Best val loss {val_loss:.3f} | Best val F1W {best_val_f1w:.3f}", flush=True)
		elif epoch - best_epoch > patience:
			model.load_state_dict(torch.load(f"{model_name}.pth"))
			val_loss, best_val_f1w = mlu.GAT_validation(model, val_loader, optimizer, criterion)
			print(f"Early stopped at epoch {epoch} with best epoch {best_epoch} | Best val loss: {val_loss:.3f} | Best val F1W: {best_val_f1w:.3f}", flush=True)
			trial.report(best_val_f1w, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			return best_val_f1w

		trial.report(val_f1w, epoch)
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()
	
	model.load_state_dict(torch.load(f"{model_name}.pth"))
	val_loss, best_val_f1w = mlu.GAT_validation(model, val_loader, optimizer, criterion)
	
	trial.report(best_val_f1w, epoch)
	if trial.should_prune():
		raise optuna.exceptions.TrialPruned()

	return best_val_f1w

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

	dim_h = trial.suggest_int('dim_h', low=32, high=256, step=32)
	heads = trial.suggest_int('heads', low=4, high=12, step=2)
	model = mlu.GAT(n_feats=data.num_features, n_classes=data.num_classes, dim_h=dim_h, heads=heads).to(device)
	
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
	
	Retruns F1W score

	'''	
	params = {'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True), 'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)}
	model = build_GAT(trial, data)
	f1w = HPO_TrainModel_GAT(model=model, data=data, model_name=model_name, trial=trial, param=params)
	
	return f1w

def run_HPO_GAT(data, model_name):

	'''
	Apply HPO to GAT model

 	Parameters
    ----------

	data : object of class torch_geometric.data.Data storing thw whole count matrix. See EmbeddExplain.classify_and_explain to see how the data object is built
	
	model_name : name to use to save the model during the training

	Output
	------
	
	Returns object of class optuna.study.Study see https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
	
	'''
	model=partial(build_GAT, data = data)
	obejctive=partial(objective_GAT, data = data, model=model, model_name=model_name)
	storage_name = "sqlite:///{}.db".format(model_name)
	study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), 
							study_name=model_name, storage=storage_name, load_if_exists=True)
	study.optimize(obejctive, n_trials=100, n_jobs=1, gc_after_trial=True)
	
	return study
