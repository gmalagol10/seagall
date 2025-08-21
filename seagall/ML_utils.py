import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from . import Utils as ut

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from .base_dataset import DEVICE


class GAT(torch.nn.Module):
	"""
	Graph Attention Network (GAT) implementation based on https://arxiv.org/abs/2105.14491

	Parameters
	----------
	n_feats : int
		Number of input features per node.
	n_classes : int
		Number of target classes.
	dim_h : int, optional
		Dimension of the hidden layer. Default is 64.
	heads : int, optional
		Number of attention heads in the first GAT layer. Default is 8.
	dropout : float, optional
		Dropout rate to use in the attention layers. Default is 0.5.

	Methods
	-------
	forward(x, edge_index)
		Forward pass applying two GATv2Conv layers with ReLU activation in between.

	"""

	def __init__(self, n_feats: int, n_classes: int, dim_h: int = 64, heads: int = 8, dropout: float = 0.5):
		super().__init__()
		self.gat1 = GATv2Conv(n_feats, dim_h, heads=heads, dropout=dropout)
		self.gat2 = GATv2Conv(dim_h * heads, n_classes, heads=1, dropout=dropout)

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
		x = self.gat1(x, edge_index)
		x = torch.relu(x)
		x = self.gat2(x, edge_index)
		return x


def _create_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
	"""
	Create a WeightedRandomSampler to handle class imbalance.

	Parameters
	----------
	y : np.ndarray
		Array of target labels.

	Returns
	-------
	WeightedRandomSampler
	"""
	if len(y) == 0:
		raise ValueError("Labels array is empty.")
	classes, counts = np.unique(y, return_counts=True)
	if np.any(counts == 0):
		raise ValueError("Some classes have zero samples.")
	weights = 1.0 / counts
	samples_weight = np.array([weights[classes.tolist().index(label)] for label in y])
	samples_weight_tensor = torch.DoubleTensor(samples_weight)
	sampler = WeightedRandomSampler(samples_weight_tensor, num_samples=len(samples_weight_tensor), replacement=True)
	return sampler


def _prepare_dataloader(X: np.ndarray | scipy.sparse.csr_matrix, y: np.ndarray, batch_size: int, sampler: WeightedRandomSampler) -> DataLoader:
	"""
	Helper function to create a DataLoader from features, labels and sampler.

	Parameters
	----------
	X : np.ndarray or scipy.sparse.csr_matrix
		Features matrix.
	y : np.ndarray
		Target labels.
	batch_size : int
		Batch size for DataLoader.
	sampler : WeightedRandomSampler
		Sampler to handle class imbalance.

	Returns
	-------
	DataLoader
	"""
	if scipy.sparse.issparse(X):
		X = X.todense()
	tensor_X = torch.FloatTensor(np.array(X, dtype=np.float32))
	tensor_y = torch.LongTensor(y)
	dataset = TensorDataset(tensor_X, tensor_y)
	dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=1)
	return dataloader


def split_train_val(X: np.ndarray | scipy.sparse.csr_matrix, y: np.ndarray, train_size: float = 0.85, val_size: float = 0.15,
					train_batch_size: int = 128, val_batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
	"""
	Split dataset into train and validation sets with balanced weighted sampling.

	Parameters
	----------
	X : np.ndarray or scipy.sparse.csr_matrix
		Feature matrix (N_samples x N_features).
	y : np.ndarray
		Target labels.
	train_size : float, optional
		Fraction of data for training. Default is 0.85.
	val_size : float, optional
		Fraction of data for validation. Default is 0.15.
	train_batch_size : int, optional
		Batch size for training DataLoader. Default is 128.
	val_batch_size : int, optional
		Batch size for validation DataLoader. Default is 64.

	Returns
	-------
	train_dataloader : DataLoader
		DataLoader for the training set.
	val_dataloader : DataLoader
		DataLoader for the validation set.

	Raises
	------
	ValueError
		If sizes don't sum to 1 or labels are empty.
	"""
	if not (0 < train_size < 1) or not (0 < val_size < 1):
		raise ValueError("train_size and val_size must be between 0 and 1")
	if not np.isclose(train_size + val_size, 1.0):
		raise ValueError("train_size and val_size must sum to 1")
	if len(y) == 0:
		raise ValueError("Target labels 'y' cannot be empty.")

	X = scipy.sparse.csr_matrix(X, dtype="float32").todense() if scipy.sparse.issparse(X) else np.array(X, dtype="float32")
	y = np.array(y).astype(int)

	X_train, X_val, y_train, y_val = train_test_split(
		X, y, train_size=train_size, test_size=val_size, random_state=42, stratify=y)

	train_sampler = _create_weighted_sampler(y_train)
	val_sampler = _create_weighted_sampler(y_val)

	train_dataloader = _prepare_dataloader(X_train, y_train, train_batch_size, train_sampler)
	val_dataloader = _prepare_dataloader(X_val, y_val, val_batch_size, val_sampler)

	return train_dataloader, val_dataloader


def split_train_val_test(X: np.ndarray | scipy.sparse.csr_matrix, y: np.ndarray, train_size: float = 0.7, val_size: float = 0.1,
						test_size: float = 0.2, train_batch_size: int = 128, valtest_batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""
	Split dataset into train, validation, and test sets with balanced weighted sampling.

	Parameters
	----------
	X : np.ndarray or scipy.sparse.csr_matrix
		Feature matrix (N_samples x N_features).
	y : np.ndarray
		Target labels.
	train_size : float, optional
		Fraction of data for training. Default is 0.7.
	val_size : float, optional
		Fraction of data for validation. Default is 0.1.
	test_size : float, optional
		Fraction of data for testing. Default is 0.2.
	train_batch_size : int, optional
		Batch size for training DataLoader. Default is 128.
	valtest_batch_size : int, optional
		Batch size for validation and test DataLoader. Default is 64.

	Returns
	-------
	train_dataloader : DataLoader
		DataLoader for the training set.
	val_dataloader : DataLoader
		DataLoader for the validation set.
	test_dataloader : DataLoader
		DataLoader for the test set.

	Raises
	------
	ValueError
		If sizes don't sum to 1 or labels are empty.
	"""
	if not np.isclose(train_size + val_size + test_size, 1.0):
		raise ValueError("train_size, val_size and test_size must sum to 1")
	if len(y) == 0:
		raise ValueError("Target labels 'y' cannot be empty.")

	if scipy.sparse.issparse(X):
		X = X.todense()
	y = np.array(y).astype(int)

	# Split off test set first
	X_tv, X_test, y_tv, y_test = train_test_split(
		X, y, train_size=train_size + val_size, random_state=42, stratify=y
	)
	test_sampler = _create_weighted_sampler(y_test)
	test_dataloader = _prepare_dataloader(X_test, y_test, valtest_batch_size, test_sampler)

	# Split train and validation
	train_fraction = train_size / (train_size + val_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_tv, y_tv, train_size=train_fraction, random_state=42, stratify=y_tv
	)
	train_sampler = _create_weighted_sampler(y_train)
	val_sampler = _create_weighted_sampler(y_val)

	train_dataloader = _prepare_dataloader(X_train, y_train, train_batch_size, train_sampler)
	val_dataloader = _prepare_dataloader(X_val, y_val, valtest_batch_size, val_sampler)

	return train_dataloader, val_dataloader, test_dataloader


def create_pyg_dataset(adata: sc.AnnData, label: str, size: float = 1.0) -> torch_geometric.data.Data:
	"""
	Create a PyG (PyTorch Geometric) dataset from an AnnData object using GRAE's graph.

	Parameters
	----------
	adata : scanpy.AnnData
		Annotated data matrix.
	label : str
		The column name in adata.obs used as target labels.
	size : float, optional
		Fraction of cells to use per label. Must be in (0, 1] (default is 1).

	Returns
	-------
	torch_geometric.data.Data
		A PyG data object with features x, edge_index and labels y.

	Raises
	------
	ValueError
		If size is zero or out of bounds.
	"""
	if not (0 < size <= 1):
		raise ValueError(f"Size must be between 0 (exclusive) and 1 (inclusive), got {size}")

	logger.info(f"Creating pyg dataset based on AnnData object with target label '{label}' and GRAE's graph. Using {int(size*100)}% of the cells")

	if size < 1:
		cells = []
		for label_val in set(adata.obs[label].dropna()):
			sub_ad = adata[adata.obs[label] == label_val].copy()
			sc.pp.subsample(sub_ad, fraction=size)
			cells.append(sub_ad.obs.index.tolist())
			del sub_ad
		cells = ut.flat_list(cells)
		ad = adata[cells].copy()
	else:
		ad = adata.copy()

	# Extract edges from GRAE graph adjacency matrix
	adj = ad.obsp.get("GRAE_graph")
	if adj is None:
		raise KeyError("GRAE_graph adjacency matrix not found in adata.obsp")

	edges_df = pd.DataFrame(adj.toarray())\
		.rename_axis("Source")\
		.reset_index()\
		.melt("Source", value_name="Weight", var_name="Target")
	edges_df = edges_df.query("Source != Target and Weight != 0").reset_index(drop=True)

	x_tensor = torch.tensor(scipy.sparse.csr_matrix(ad.X, dtype="float32").toarray(), dtype=torch.float32)
	edge_index_tensor = torch.tensor(edges_df[["Source", "Target"]].astype(int).to_numpy().T, dtype=torch.long)
	y_tensor = torch.from_numpy(ad.obs["target"].to_numpy().astype(int)).long()

	mydata = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)
	mydata.num_features = mydata.x.shape[1]
	mydata.num_classes = len(set(mydata.y.numpy()))

	del ad  # free memory
	return mydata


def GAT_1_step_training(model: torch.nn.Module, train_loader: DataLoader,
						optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> tuple[float, float]:
	"""
	Perform one training epoch on GAT model.

	Parameters
	----------
	model : torch.nn.Module
		GAT model.
	train_loader : DataLoader
		DataLoader for training data.
	optimizer : torch.optim.Optimizer
		Optimizer for model parameters.
	criterion : torch.nn.Module
		Loss function (e.g., CrossEntropyLoss).

	Returns
	-------
	average_loss : float
		Average loss over all batches.
	average_f1 : float
		Average macro F1-score over all batches.
	"""
	model.train()
	train_loss = 0.0
	train_f1 = 0.0

	for batch in train_loader:
		optimizer.zero_grad()
		batch = batch.to(DEVICE)
		out = model(batch.x, batch.edge_index)[: batch.batch_size]

		y_true = batch.y[: batch.batch_size]
		loss = criterion(out, y_true)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		preds = out.argmax(dim=1).cpu().numpy()
		labels = y_true.cpu().numpy()
		f1 = precision_recall_fscore_support(labels, preds, average="macro")[2]
		train_f1 += f1

	return train_loss / len(train_loader), train_f1 / len(train_loader)


def GAT_validation(model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module) -> tuple[float, float]:
	"""
	Perform one validation epoch on GAT model.

	Parameters
	----------
	model : torch.nn.Module
		GAT model.
	val_loader : DataLoader
		DataLoader for validation data.
	criterion : torch.nn.Module
		Loss function (e.g., CrossEntropyLoss).

	Returns
	-------
	average_loss : float
		Average loss over all batches.
	average_f1 : float
		Average macro F1-score over all batches.
	"""
	model.eval()
	val_loss = 0.0
	val_f1 = 0.0

	with torch.no_grad():
		for batch in val_loader:
			batch = batch.to(DEVICE)
			out = model(batch.x, batch.edge_index)[: batch.batch_size]

			y_true = batch.y[: batch.batch_size]
			loss = criterion(out, y_true)
			val_loss += loss.item()

			preds = out.argmax(dim=1).cpu().numpy()
			labels = y_true.cpu().numpy()
			f1 = precision_recall_fscore_support(labels, preds, average="macro")[2]
			val_f1 += f1

	return val_loss / len(val_loader), val_f1 / len(val_loader)


def GAT_train_node_classifier(model: torch.nn.Module, data: torch_geometric.data.Data, optimizer: torch.optim.Optimizer,
							  criterion: torch.nn.Module, model_name: str, epochs: int = 300, patience: int = 30) -> tuple[torch.nn.Module, dict[str, list[float]]]:
	"""
	Train a GAT model node classifier with early stopping.

	Parameters
	----------
	model : torch.nn.Module
		GAT model.
	data : torch_geometric.data.Data
		PyG data object containing node features, edge_index and masks.
	optimizer : torch.optim.Optimizer
		Optimizer for training.
	criterion : torch.nn.Module
		Loss function (e.g., CrossEntropyLoss).
	model_name : str
		Path to save the best model weights.
	epochs : int, optional
		Maximum number of training epochs. Default is 250.
	patience : int, optional
		Number of epochs to wait for improvement before early stopping. Default is 30.

	Returns
	-------
	model : torch.nn.Module
		Trained GAT model with best weights loaded.
	history : dict
		Dictionary containing loss and F1 score history for training and validation.
	"""
	best_val_f1 = -np.inf
	best_epoch = -1

	history = {"TrainLoss": [], "TrainF1": [], "ValLoss": [], "ValF1": []}

	train_loader = NeighborLoader(data,
								  num_neighbors=[15, 10],
								  input_nodes=data.train_mask,
								  batch_size=128,
								  directed=False,
								  shuffle=True)

	val_loader = NeighborLoader(data,
								num_neighbors=[15,10],
								input_nodes=data.val_mask,
								batch_size=64,
								directed=False,
								shuffle=True)

	for epoch in range(1, epochs + 1):
		train_loss, train_f1 = GAT_1_step_training(model, train_loader, optimizer, criterion)
		val_loss, val_f1 = GAT_validation(model, val_loader, criterion)

		history["TrainLoss"].append(train_loss)
		history["TrainF1"].append(train_f1)
		history["ValLoss"].append(val_loss)
		history["ValF1"].append(val_f1)

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			best_epoch = epoch
			torch.save(model.state_dict(), model_name)
			logger.info(f"Epoch {epoch:03d}: New best val F1 = {val_f1:.4f}, model saved.")
		if epoch - best_epoch >= patience:
			logger.info(f"Early stopping at epoch {epoch:03d}. Best val F1 = {best_val_f1:.4f}")
			break

	model.load_state_dict(torch.load(model_name))
	return model, history

