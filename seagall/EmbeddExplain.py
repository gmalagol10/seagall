# Local module imports (as specified)
from . import ML_utils as mlu
from . import Models as mod
from . import Utils as ut
from . import HPO as hpo
from .Models import GRAE
from .base_dataset import DEVICE, BaseDataset, logger

# Standard library imports
import os
import gc
import json
import time
from pathlib import Path
from typing import Optional

# Numerical and scientific computing
import numpy as np
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight

# PyTorch and PyTorch Geometric
import torch
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer

# Scanpy for AnnData support
import scanpy as sc

# Device setup
torch.manual_seed(np.random.randint(0,10000))
from .base_dataset import DEVICE

def geometrical_embedding(
	M: np.ndarray, 
	y: Optional[np.ndarray] = None, 
	epochs: int = 300,
	patience: int = 30, 
	path: str = "SEAGALL", 
	model_name: str = "mymodel", 
	overwrite: bool = False) -> tuple:
	"""
	Compute a geometry-preserving embedding of a feature matrix using GRAE.

	This function uses the Geometric Regularized AutoEncoder (GRAE) to embed 
	a feature matrix while preserving its intrinsic geometric structure.
	For details, refer to: https://github.com/KevinMoonLab/GRAE

	Parameters
	----------
	M : array-like, shape (N, F)
		Input feature matrix with N samples (cells) and F features.
	
	y : array-like, optional, default=None
		Array containing class labels for each sample. If None, a dummy 
		array of ones is used.
	
	epochs : int, optional, default=200
		Number of training epochs for the GRAE model.
	
	patience : int, optional, default=20
		Number of epochs with no improvement after which training will be stopped.
	
	path : str, optional, default="SEAGALL"
		Directory where the trained model will be saved.
	
	model_name : str, optional, default="mymodel"
		Name under which the model will be saved or loaded.
	
	overwrite : bool, optional, default=False
		If True, retrains and overwrites the existing model. If False and a 
		saved model is found, it will be loaded and reused.

	Returns
	-------
	embedding : array-like, shape (N, latent_dim)
		Embedded representation of the input matrix in a lower-dimensional latent space.
	
	reconstruction : sparse matrix, shape (N, F)
		Reconstructed version of the input matrix obtained by decoding the embedding.
	"""
	# Ensure output directory exists
	Path(path).mkdir(parents=True, exist_ok=True)

	# Default label handling
	if y is None:
		y = np.ones(M.shape[0])

	# Memory management
	torch.cuda.empty_cache()
	gc.collect()

	# Ensure dense format for GRAE
	M = sparse.csr_matrix(M, dtype=np.float32).toarray()

	# Prepare dataset
	dataset = BaseDataset(M, y=y, split='none', split_ratio=1, random_state=42, labels=y)
	train_dataset, val_dataset, val_mask = dataset.validation_split(ratio=0.15)

	# Initialize model
	latent_dim = int(np.round(M.shape[1] ** (1/3)))
	model = GRAE(epochs=epochs, patience=patience, latent_dim=latent_dim, write_path=path, data_val=val_dataset)

	model_path = f"{path}/SEAGALL_{model_name}_GRAE.pth"
	logger.info("Fitting GRAE")
	model.fit(train_dataset)
	model.save(model_path)

	transformed = model.transform(dataset)
	reconstructed = sparse.csr_matrix(model.inverse_transform(transformed), dtype=np.float32)

	return transformed, reconstructed

def geometrical_graph(
	adata: sc.AnnData,
	target_label: Optional[str] = None,
	layer: str = "X",
	epochs: int = 300,
	patience: int = 30,
	path: str = "SEAGALL",
	model_name: str = "mymodel",
	overwrite: bool = False) -> None:
	"""
	Construct a k-NN graph of cells in the latent space of the GRAE model.

	This function takes the input `AnnData` object, computes the GRAE embedding, 
	constructs a k-NN graph based on the latent representation, and stores the result
	in the `.obsp`, `.obsm`, and `.layers` attributes of the `AnnData` object.

	Parameters
	----------
	adata : sc.AnnData
		Annotated data matrix with cells as rows and features as columns.
		
	target_label : str, optional (default: None)
		The label used for stratified splitting in the GRAE model. If provided, 
		the cells will be split based on this label for training and validation, 
		helping to account for class imbalance.
		
	layer : str (default: "X")
		The layer to embed (e.g., "X" for raw data or any custom layer name).
		
	epochs : int (default: 300)
		Number of epochs to train the GRAE model.
		
	patience : int (default: 30)
		Early stopping threshold based on validation performance.
		
	path : str (default: "SEAGALL")
		Directory where the GRAE model will be saved.
		
	model_name : str (default: "mymodel")
		The name of the GRAE model.
		
	overwrite : bool (default: False)
		Whether to overwrite an existing model if the model file already exists.

	Returns
	-------
	None
		The `AnnData` object is modified in-place with:
		- `.obsp['GRAE_graph']`: Sparse matrix representing the k-NN graph.
		- `.obsm['GRAE_latent_space']`: The latent space obtained from the GRAE model.
		- `.layers['GRAE_decoded_matrix']`: The decoded matrix of the GRAE model.

	Notes
	-----
	This function uses the `geometrical_embedding` function to compute the latent 
	representation. Ensure that the embedding function and model are compatible with 
	GPU acceleration for large datasets.
	"""
	# Make sure variable names are unique in adata
	adata.var_names_make_unique()

	# Handle target label if provided for stratified training-validation split
	if target_label:
		ut.process_target_label(adata, target_label)

	# Get the data matrix (raw or from a specified layer)
	M = adata.X.copy() if layer == "X" else adata.layers.get(layer, None)
	if M is None:
		raise ValueError(f"Layer '{layer}' not found in the AnnData object.")
	# Compute the geometrical embedding using the GRAE model
	Z = geometrical_embedding(M=M, y=adata.obs["target"].values, epochs=epochs, 
							  patience=patience, path=path, model_name=model_name, overwrite=overwrite)
	
	# Create a new AnnData object for the transformed data
	ad_ret = sc.AnnData(Z[0])

	# Compute k-NN graph based on the latent space using UMAP
	sc.pp.neighbors(ad_ret, use_rep="X", method="umap")

	# Store the results in the original AnnData object
	adata.obsp[f"GRAE_graph"] = sparse.csr_matrix(ad_ret.obsp["connectivities"], dtype="float32")
	adata.obsm["GRAE_latent_space"] = Z[0]
#	adata.layers[f"GRAE_decoded_matrix"] = Z[1]




def explain(adata, target_label: str, hypopt: float = 0, n_feat: int = 50, path: str = "SEAGALL", model_name: str = "mymodel"):
	"""
	Trains a GAT classifier on an AnnData object, optionally performs HPO, and explains predictions using GNNExplainer.

	Parameters
	----------
	adata : AnnData
		Annotated data matrix.
	target_label : str
		Column name in `adata.obs` used as the label for classification.
	hypopt : float, default=1
		Fraction of cells to use for hyperparameter optimization (0 to skip).
	n_feat : int, default=50
		Number of top features to extract per class.
	path : str, default="SEAGALL"
		Folder to save the model and output files.
	model_name : str, default="mymodel"
		Name to use in saved model filenames.

	Returns
	-------
	None
		Modifies `adata` in-place. Saves trained model and feature importances.
	"""


	# Path setup
	save_path = Path(path)
	save_path.mkdir(parents=True, exist_ok=True)
	model_path = save_path / f"SEAGALL_{model_name}_{target_label}"

	# Clean and map target labels
	ut.process_target_label(adata, target_label)

	# Dataset creation for HPO
	if float(hypopt) > 0:
		logger.info(f"{time.strftime('%c')} Checking for HPO file")
		hpo_path = f"{model_path}_HPO"
		if not os.path.isfile(f"{hpo_path}.json"):
			logger.info(f"{time.strftime('%c')} No HPO found. Running with {int(100*hypopt)}% of cells")
			data = mlu.create_pyg_dataset(adata, target_label, hypopt)
			data = torch_geometric.transforms.RandomNodeSplit(num_val=0.2, num_test=0)(data)
			study = hpo.run_HPO_GAT(data, hpo_path)
			with open(f"{hpo_path}.json", "w") as f:
				json.dump(study.best_params, f)
			best_params = study.best_params
		else:
			logger.info(f"{time.strftime('%c')} Loading existing HPO results")
			with open(f"{hpo_path}.json", "r") as f:
				best_params = json.load(f)
			for k, v in best_params.items():
				logger.info(f"Best {k}: {v}")
	else:
		best_params = None

	# Full training dataset
	logger.info(f"{time.strftime('%c')} Creating dataset")
	data = mlu.create_pyg_dataset(adata, target_label)
	data = torch_geometric.transforms.RandomNodeSplit(num_val=0.15, num_test=0.15)(data)

	# Model initialization
	if best_params:
		model = mod.GAT(n_feats=data.num_features, n_classes=data.num_classes, dim_h=best_params["dim_h"], heads=best_params["heads"]).to(DEVICE)
		optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
	else:
		model = mod.GAT(n_feats=data.num_features, n_classes=data.num_classes).to(DEVICE)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

	# Loss setup
	class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(data.y), y=data.y.numpy())
	class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)
	criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

	# Train the model
	logger.info(f"{time.strftime('%c')} Training model")
	model, history = mlu.GAT_train_node_classifier(model, data, optimizer, criterion, f"{model_path}.pth", epochs=300, patience=20)

	with open(f"{model_path}_GAT_Progress.json", "w") as f:
		json.dump(history, f)

	# Make predictions
	logger.info(f"{time.strftime('%c')} Predicting and annotating")
	model.eval()
	predictions = model(data.x, data.edge_index).argmax(dim=1).cpu().detach().numpy()
	adata.obs["SEAGALL_prediction"] = [adata.uns["inv_map"][str(p)] for p in predictions]

	# Set membership
	adata.obs["SEAGALL_set"] = "--"
	assert len(adata) == len(data.y), "Mismatch between AnnData and training data"
	adata.obs.loc[data.train_mask.cpu().numpy(), "SEAGALL_set"] = "Train"
	adata.obs.loc[data.val_mask.cpu().numpy(), "SEAGALL_set"] = "Validation"
	adata.obs.loc[data.test_mask.cpu().numpy(), "SEAGALL_set"] = "Test"

	# Explain the model
	logger.info(f"{time.strftime('%c')} Running GNNExplainer")
	explainer = Explainer(model=model,
						  algorithm=GNNExplainer(epochs=300),
						  explanation_type='model',
						  node_mask_type='attributes',
						  edge_mask_type='object',
						  model_config=dict(mode='multiclass_classification', task_level='node', return_type='probs'))
	explanation = explainer(x=data.x, edge_index=data.edge_index)
	importance_matrix = explanation.node_mask.cpu().detach().numpy()
	adata.layers["SEAGALL_Importance"] = importance_matrix.astype(np.float32)

	# Class-specific feature importance
	for label in sorted(set(adata.obs[target_label])):
		idx = adata.obs[target_label] == label
		avg_importance = np.array(adata[idx].layers["SEAGALL_Importance"].mean(axis=0)).reshape(-1)
		adata.var[f"SEAGALL_Importance_for_{label}"] = avg_importance

	# Feature specificity
#	adata.uns[f"SEAGALL_Top_{n_feat}_Specificty"] = mlu.specificty(adata, target_label, n_feat).values.astype(float)

	logger.info(f"{time.strftime('%c')} Pipeline complete")

