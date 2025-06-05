"""Submodules used by models."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from topology import AutoencoderModel


# Hush the linter: Warning W0221 corresponds to a mismatch between parent class
# method signature and the child class
# pylint: disable=W0221
class BaseAE(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim=256, latent_dim=32, act_fn=torch.nn.LeakyReLU, act_fn_out=None, dp=0.5):
		"""Create new autoencoder with pre-defined latent dimension."""
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
																																																																															
		self.act_fn = act_fn
		self.act_fn_out = act_fn_out

		self.encoder = torch.nn.Sequential(torch.nn.Dropout(dp), torch.nn.Linear(self.input_dim, self.hidden_dim), self.act_fn(), 
										   torch.nn.Dropout(dp), torch.nn.Linear(self.hidden_dim, self.latent_dim))

		if act_fn_out:
				self.decoder = torch.nn.Sequential(torch.nn.Dropout(dp), torch.nn.Linear(self.latent_dim, self.hidden_dim), self.act_fn(),
												   torch.nn.Dropout(dp), torch.nn.Linear(self.hidden_dim, self.input_dim), self.act_fn_out())
		else:
				self.decoder = torch.nn.Sequential(torch.nn.Dropout(dp), torch.nn.Linear(self.latent_dim, self.hidden_dim), self.act_fn(), 
												   torch.nn.Dropout(dp), torch.nn.Linear(self.hidden_dim, self.input_dim))

		self.loss_fn = torch.nn.MSELoss()

	def encode(self, x):
		"""Embed data in latent space."""
		return self.encoder(x)

	def decode(self, z):
		"""Decode data from latent space."""
		return self.decoder(z)

	def forward(self, x):
		"""Embeds and reconstructs data, returning a loss."""
		z = self.encode(x)
		x_hat = self.decode(z)

		# The loss can of course be changed. If this is your first time
		# working with autoencoders, a good exercise would be to 'grok'
		# the meaning of different losses.
		reconstruction_error = self.loss_fn(x, x_hat)
		return reconstruction_error		
