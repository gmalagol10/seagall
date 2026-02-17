# module adapted from https://github.com/KevinMoonLab/GRAE.git

"""Torch modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
	"""FC layer with Relu activation."""

	def __init__(self, in_dim, out_dim):
		"""Init.

		Args:
			in_dim(int): Input dimension.
			out_dim(int): Output dimension
		"""
		super().__init__()
		self.linear = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		"""Forward pass.

		Args:
			x(torch.Tensor): Input data.

		Returns:
			torch.Tensor: Activations.

		"""
		return F.relu(self.linear(x))


class MLP(nn.Sequential):
	"""Sequence of FC layers with Relu activations.

	No activation on last layer, unless sigmoid is requested."""

	def __init__(self, dim_list, sigmoid=False):
		"""Init.

		Args:
			dim_list(List[int]): List of dimensions. Ex: [200, 100, 50] will create two layers (200x100 followed by
			100x50).
		"""
		# Activations on all layers except last one
		modules = [LinearBlock(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
		modules.append(nn.Linear(dim_list[-2], dim_list[-1]))

		if sigmoid:
			modules.append(nn.Sigmoid())

		super().__init__(*modules)



class AutoencoderModule(nn.Module):
	"""Vanilla Autoencoder torch module"""

	def __init__(self, input_dim, hidden_dims, z_dim, sigmoid=False):
		"""Init.

		Args:
			input_dim(int): Dimension of the input data.
			hidden_dims(List[int]): List of hidden dimensions. Do not include dimensions of the input layer and the
			bottleneck. See MLP for example.
			z_dim(int): Bottleneck dimension.

			sigmoid(bool): Apply sigmoid to the output.
		"""
		super().__init__()

		full_list = [input_dim] + list(hidden_dims) + [z_dim]

		self.encoder = MLP(dim_list=full_list)

		full_list.reverse()  # Use reversed architecture for decoder
		full_list[0] = z_dim

		self.decoder = MLP(dim_list=full_list, sigmoid=sigmoid)

	def forward(self, x):
		"""Forward pass.

		Args:
			x(torch.Tensor): Input data.

		Returns:
			tuple:
				torch.Tensor: Reconstructions
				torch.Tensor: Embedding (latent space coordinates)

		"""
		z = self.encoder(x)

		output = self.decoder(z)

		return output, z
