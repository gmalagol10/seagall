"""PHATE, AE and GRAE model classes with sklearn inspired interface."""
import os


import torch
import torch.nn as nn
import numpy as np
import scipy

from . import base_model
from .  import manifold_tools
from .  import torch_modules
import torch

from torch.autograd import grad as torch_grad
from torch_geometric.nn import GATv2Conv

from .base_dataset import DEVICE, logger
from .  import base_dataset

from pathlib import Path


# Hyperparameters defaults
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 0
EPOCHS = 300
HIDDEN_DIMS = {"hidden_dim" : 800, "hidden_dim_2" : 400, "hidden_dim_1" : 200}  # Default fully-connected dimensions

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

class AE(base_model.BaseModel):
	"""Vanilla Autoencoder model.

	Trained with Adam and MSE Loss.
	Model will infer from the data whether to use a fully FC or convolutional + FC architecture.
	"""

	def __init__(self, *,
				 lr=LR,
				 epochs=EPOCHS,
				 batch_size=BATCH_SIZE,
				 weight_decay=WEIGHT_DECAY,
				 random_state=base_model.SEED,
				 latent_dim=2,
				 hidden_dims=HIDDEN_DIMS,
				 conv_dims=[],
				 conv_fc_dims=[],
				 train_mask=None,
				 patience=30,
				 data_val=None,
				 comet_exp=None,
				 write_path=None):
		"""Init. Arguments specify the architecture of the encoder. Decoder will use the reversed architecture.

		Args:
			lr(float): Learning rate.
			epochs(int): Number of epochs for model training.
			batch_size(int): Mini-batch size.
			weight_decay(float): L2 penalty.
			random_state(int): To seed parameters and training routine for reproducible results.
			latent_dim(int): Bottleneck dimension.
			hidden_dims(dict[int]): Number and size of fully connected layers for encoder. Do not specify the input
			layer or the bottleneck layer, since they are inferred from the data or from the latent_dim
			argument respectively. Decoder will use the same dimensions in reverse order. This argument is only used if
			provided samples are flat vectors.
			conv_dims(List[int]): Specify the number of convolutional layers. The int values specify the number of
			channels for each layer. This argument is only used if provided samples are images (i.e. 3D tensors)
			conv_fc_dims(List[int]): Number and size of fully connected layers following the conv_dims convolutionnal
			layer. No need to specify the bottleneck layer. This argument is only used if provided samples
			are images (i.e. 3D tensors)
			patience(int): Epochs with no validation MSE improvement before early stopping.
			data_val(BaseDataset): Split to validate MSE on for early stopping.
			comet_exp(Experiment): Comet experiment to log results.
			write_path(str): Where to write temp files.
		"""
		self.random_state = random_state
		self.latent_dim = latent_dim
		self.hidden_dims = tuple(hidden_dims.values())
		self.fitted = False  # If model was fitted
		self.torch_module = None  # Will be initialized to the appropriate torch module when fit method is called
		self.optimizer = None  # Will be initialized to the appropriate optimizer when fit method is called
		self.lr = lr
		self.train_mask = train_mask
		self.epochs = epochs
		self.batch_size = batch_size
		self.weight_decay = weight_decay
		self.criterion = nn.MSELoss(reduction='mean')
		self.conv_dims = conv_dims
		self.conv_fc_dims = conv_fc_dims
		self.comet_exp = comet_exp
		self.data_shape = None  # Shape of input data

		# Early stopping attributes
		self.data_val = data_val
		self.val_loader = None
		self.patience = patience
		self.current_loss_min = np.inf
		self.early_stopping_count = 0
		if write_path is not None:
			self.write_path = write_path
			Path(self.write_path).mkdir(parents=True, exist_ok=True)

	def init_torch_module(self, data_shape, vae=False, sigmoid=False):
		"""Infer autoencoder architecture (MLP or Convolutional + MLP) from data shape.

		Initialize torch module.

		Args:
			data_shape(tuple[int]): Shape of one sample.
			vae(bool): Make this architecture a VAE.
			sigmoid(bool): Apply sigmoid to decoder output.

		"""
		# Infer input size from data. Initialize torch module and optimizer
		if len(data_shape) == 1:
			# Samples are flat vectors. MLP case
			input_size = data_shape[0]
			self.torch_module = torch_modules.AutoencoderModule(input_dim=input_size,
												  hidden_dims=self.hidden_dims,
												  z_dim=self.latent_dim,
												  sigmoid=sigmoid)
		else:
			raise Exception(f'Invalid channel number. X has {len(data_shape)}')

		self.torch_module.to(base_model.DEVICE)

	def fit(self, x):
		"""Fit model to data.

		Args:
			x(BaseDataset): Dataset to fit.

		"""

		# Reproducibility
		torch.manual_seed(self.random_state)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		# Save data shape
		self.data_shape = x[0][0].shape

		# Fetch appropriate torch module
		if self.torch_module is None:
			self.init_torch_module(self.data_shape)

		# Optimizer
		self.optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		# Train AE
		# Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
		self.torch_module.train()

		self.loader = self.get_loader(x)

		if self.data_val is not None:
			self.val_loader = self.get_loader(self.data_val)

		# Get first metrics
		self.log_metrics(0)

		for epoch in range(1, self.epochs + 1):
			if epoch % 10 == 0:
				logger.info(f'Epoch {epoch}...')
			for batch in self.loader:
				self.optimizer.zero_grad()
				self.train_body(batch)
				self.optimizer.step()

			self.log_metrics(epoch)
			self.end_epoch(epoch)

			# Early stopping
			if self.early_stopping_count == self.patience:
				logger.info(f"Model has been early stopped at epoch {epoch}")
				break

		# Load checkpoint if it exists
		checkpoint_path = os.path.join(self.write_path, 'checkpoint.pt')

		if os.path.exists(checkpoint_path):
			self.load(checkpoint_path)
			os.remove(checkpoint_path)

	def get_loader(self, x):
		"""Fetch data loader.

		Args:
			x(BaseDataset): Data to be wrapped in loader.

		Returns:
			torch.utils.data.DataLoader: Torch DataLoader for mini-batch training.

		"""
		return torch.utils.data.DataLoader(x, batch_size=self.batch_size, shuffle=False)

	def train_body(self, batch):
		"""Called in main training loop to update torch_module parameters.

		Args:
			batch(tuple[torch.Tensor]): Training batch.

		"""
		data, _, idx = batch  # No need for labels. Training is unsupervised
		data = data.to(base_model.DEVICE)

		x_hat, z = self.torch_module(data)  # Forward pass
		self.compute_loss(data, x_hat.to(base_model.DEVICE), z.to(base_model.DEVICE), idx)

	def compute_loss(self, x, x_hat, z, idx):
		"""Apply loss to update parameters following a forward pass.

		Args:
			x(torch.Tensor): Input batch.
			x_hat(torch.Tensor): Reconstructed batch (decoder output).
			z(torch.Tensor): Batch embedding (encoder output).
			idx(torch.Tensor): Indices of samples in batch.

		"""
		loss = self.criterion(x_hat, x)
		loss.backward()

	def end_epoch(self, epoch):
		"""Method called at the end of every training epoch.

		Args:
			epoch(int): Current epoch.

		"""
		pass

	def eval_MSE(self, loader):
		"""Compute MSE on data.

		Args:
			loader(DataLoader): Dataset loader.

		Returns:
			float: MSE.

		"""
		# Compute MSE over dataset in loader
		self.torch_module.eval()
		sum_loss = 0

		for batch in loader:
			data, _, idx = batch  # No need for labels. Training is unsupervised
			data = data.to(base_model.DEVICE)

			x_hat, z = self.torch_module(data)  # Forward pass
			sum_loss += data.shape[0] * self.criterion(data, x_hat).item()

		self.torch_module.train()

		return sum_loss / len(loader.dataset)  # Return average per observation

	def log_metrics(self, epoch):
		"""Log metrics.

		Args:
			epoch(int): Current epoch.

		"""
		self.log_metrics_train(epoch)
		self.log_metrics_val(epoch)

	def log_metrics_val(self, epoch):
		"""Compute validation metrics, log them to comet if need be and update early stopping attributes.

		Args:
			epoch(int):  Current epoch.
		"""
		# Validation loss
		if self.val_loader is not None:
			val_mse = self.eval_MSE(self.val_loader)
			if np.around(val_mse/self.current_loss_min, decimals=2) < 0.975:
				# If new min, update attributes and checkpoint model
				self.current_loss_min = val_mse
				self.early_stopping_count = 0
				self.save(os.path.join(self.write_path, 'checkpoint.pt'))
			else:
				self.early_stopping_count += 1

	def log_metrics_train(self, epoch):
		"""Log train metrics, log them to comet if need be and update early stopping attributes.

		Args:
			epoch(int):  Current epoch.
		"""
		# Train loss

		train_mse = self.eval_MSE(self.loader)

	def transform(self, x):
		"""Transform data.

		Args:
			x(BaseDataset): Dataset to transform.
		Returns:
			ndarray: Embedding of x.

		"""
		self.torch_module.eval()
		loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size, shuffle=False)
		z = [self.torch_module.encoder(batch.to(base_model.DEVICE)).cpu().detach().numpy() for batch, _, _ in loader]
		return np.concatenate(z)

	def inverse_transform(self, x):
		"""Take coordinates in the embedding space and invert them to the data space.

		Args:
			x(ndarray): Points in the embedded space with samples on the first axis.
		Returns:
			ndarray: Inverse (reconstruction) of x.

		"""
		self.torch_module.eval()
		x = base_dataset.FromNumpyDataset(x)
		loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
											 shuffle=False)
		x_hat = [self.torch_module.decoder(batch.to(base_model.DEVICE)).cpu().detach().numpy()
				 for batch in loader]

		return np.concatenate(x_hat)

	def save(self, path):
		"""Save state dict.

		Args:
			path(str): File path.

		"""
		state = self.torch_module.state_dict()
		state['data_shape'] = self.data_shape
		torch.save(state, path)

	def load(self, path):
		"""Load state dict.

		Args:
			path(str): File path.

		"""
		state = torch.load(path)
		data_shape = state.pop('data_shape')

		if self.torch_module is None:
			self.init_torch_module(data_shape)

		self.torch_module.load_state_dict(state)


class GRAE(AE):
	"""Standard GRAE class.

	AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by a manifold
	learning algorithm.
	"""

	def __init__(self, *, embedder=manifold_tools.PHATE, embedder_params={}, target_embedding = None, lam=100, relax=False, **kwargs):
		"""Init.

		Args:
			embedder(BaseModel): Manifold learning class constructor.
			embedder_params(dict): Parameters to pass to embedder.
			lam(float): Regularization factor.
			relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
			**kwargs: All other arguments with keys are passed to the AE parent class.
		"""
		super().__init__(**kwargs)
		self.lam = lam
		self.lam_original = lam  # Needed to compute the lambda relaxation
		self.relax = relax
		
		if target_embedding is not None:
			self.target_embedding = torch.from_numpy(scipy.stats.zscore(target_embedding)).float().to(base_model.DEVICE)  # To store the target embedding as computed by embedder
		else:
			self.target_embedding = None
			self.embedder = embedder(random_state=self.random_state, n_components=self.latent_dim, **embedder_params)  # To compute target embedding.

	#	if embedder_params is None and self.target_embedding is None:
	#		embedder_params = {"knn" : 5, t : "auto", "gamma" : 1, verbose : 1, "n_jobs" : -1}
			

	def fit(self, x):
		"""Fit model to data.

		Args:
			x(BaseDataset): Dataset to fit.

		"""
		if self.target_embedding is not None:
			logger.info('Geometrical embedding was passed, no need to to fit manifold learning method...')
		else:
			logger.info('Fitting manifold learning method...')
			emb = scipy.stats.zscore(self.embedder.fit_transform(x))  # Normalize embedding
			emb = (emb - emb.min())/(emb.max()-emb.min())
			self.target_embedding = torch.from_numpy(emb).float().to(base_model.DEVICE)

		logger.info('Fitting encoder & decoder...')
		super().fit(x)

	def compute_loss(self, x, x_hat, z, idx):
		"""Compute torch-compatible geometric loss.

		Args:
			x(torch.Tensor): Input batch.
			x_hat(torch.Tensor): Reconstructed batch (decoder output).
			z(torch.Tensor): Batch embedding (encoder output).
			idx(torch.Tensor): Indices of samples in batch.

		"""
		if self.lam > 0:
			loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.target_embedding[idx])
		else:
			loss = self.criterion(x, x_hat)

		loss.backward()
