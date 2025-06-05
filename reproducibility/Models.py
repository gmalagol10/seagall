import warnings
warnings.filterwarnings("ignore")

import torch
import torch_geometric

from topology import TopologicalSignatureDistance
import submodules
from topology import AutoencoderModel

from pathlib import Path
from tqdm import tqdm

class GCN(torch.nn.Module):
	def __init__(self, n_feats, n_classes, hidden_dim=128):
		super(GCN, self).__init__()
		self.conv1 = torch_geometric.nn.GCNConv(n_feats, hidden_dim)
		self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, n_classes)

	def forward(self, x, edge_index, edge_weight, dp1=0.5, dp2=0.5):
		x = torch.nn.functional.dropout(x, p=dp1, training=self.training)
		x = self.conv1(x, edge_index, edge_weight)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.dropout(x, p=dp2, training=self.training)
		x = self.conv2(x, edge_index, edge_weight)
		
		return x


class GAT(torch.nn.Module):
	def __init__(self, n_feats, n_classes, dim_h=64, heads=8):
		super().__init__()
		self.gat1 = torch_geometric.nn.GATv2Conv(n_feats, dim_h, heads=heads)
		self.gat2 = torch_geometric.nn.GATv2Conv(dim_h*heads, n_classes, heads=1)

	def forward(self, x, edge_index, dp1=0.5, dp2=0.5):
		x = torch.nn.functional.dropout(x, p=dp1, training=self.training)
		x = self.gat1(x, edge_index)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.dropout(x, p=dp2, training=self.training)
		x = self.gat2(x, edge_index)
		
		return x

class NN(torch.nn.Module):
	def __init__(self, n_feats, n_classes, hidden_dim=256):
		super().__init__()
		self.l1 = torch.nn.Linear(n_feats, hidden_dim)
		self.l2 = torch.nn.Linear(hidden_dim, n_classes)

	def forward(self, x, dp1=0.5, dp2=0.5):
		x = torch.nn.functional.dropout(x, p=dp1, training=self.training)
		x = self.l1(x)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.dropout(x, p=dp2, training=self.training)
		x = self.l2(x)
		
		return x
				
class TopologicallyRegularizedAutoencoder(AutoencoderModel):
	def __init__(self, lam=1., p=2, autoencoder_model='BaseAE', ae_kwargs=None, toposig_kwargs=None):
		"""
			Args:
			lam: Regularization strength
			ae_kwargs: Kewords to pass to `LinearAutoencoder` class
			toposig_kwargs: Keywords to pass to `TopologicalSignature` class
		"""
		super().__init__()
		self.lam = lam
		self.p = p
		ae_kwargs = ae_kwargs if ae_kwargs else {}
		toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
		self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
		self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)
		self.latent_norm = torch.nn.Parameter(data=torch.ones(1), requires_grad=True)

	@staticmethod
	def _compute_distance_matrix(self, x):
		x_flat = x.view(x.size(0), -1)
		distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=self.p)
		return distances

	def forward(self, x):

		z = self.autoencoder.encode(x)

		x_distances = self._compute_distance_matrix(self, x)

		dimensions = x.size()
		batch_size = dimensions[0]
		
		x_distances = x_distances / x_distances.max()

		latent_distances = self._compute_distance_matrix(self, z)
		latent_distances = latent_distances / self.latent_norm

		rec_loss = self.autoencoder(x)

		topo_loss, topo_loss_components = self.topo_sig(x_distances, latent_distances)
		topo_loss = topo_loss / float(batch_size) 

		loss = rec_loss + self.lam * topo_loss

		return loss

	def encode(self, x):
		return self.autoencoder.encode(x)

	def decode(self, z):
		return self.autoencoder.decode(z)
	
	
class VAutoencoder(torch.nn.Module):
	def __init__(self, autoencoder_model='BaseAE', ae_kwargs=None):
		super().__init__()

		ae_kwargs = ae_kwargs if ae_kwargs else {}
		self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)
					
		self.fc_mu = torch.nn.Linear(self.autoencoder.latent_dim, self.autoencoder.latent_dim)
		self.fc_log_var = torch.nn.Linear(self.autoencoder.latent_dim, self.autoencoder.latent_dim)

	def sample(self, mu, log_var):
		"""Sample from distribution"""
		epsilon = torch.randn_like(log_var)
		return mu + torch.exp(log_var / 2) * epsilon

	def encode(self, x):
		"""Embed data in latent space."""
		x = self.autoencoder.encoder(x)
		mu = self.fc_mu(x)
		log_var = self.fc_log_var(x)
		z = self.sample(mu, log_var)
		return [z, mu, log_var]

	def decode(self, z):
		"""Decode data from latent space."""
		return self.autoencoder.decoder(z)

	def forward(self, x):
		"""Embeds and reconstructs data, returning a loss."""
		z, mu, log_var = self.encode(x)
		x_hat = self.autoencoder.decode(z)
		return [x_hat, mu, log_var]

