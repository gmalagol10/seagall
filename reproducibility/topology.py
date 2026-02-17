'''
Methods for calculating lower-dimensional persistent homology.
'''

import numpy as np
import abc
import typing
import torch.nn as nn

class View(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape

	def forward(self, x):
		return x.view(*self.shape)

class Print(nn.Module):
	def __init__(self, name):
		self.name = name
		super().__init__()

	def forward(self, x):
		print(self.name, x.size())
		return x

class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
	"""Abstract base class for autoencoders."""
	# pylint: disable=W0221
	@abc.abstractmethod
	def forward(self, x) -> typing.Tuple[float, typing.Dict[str, float]]:
		"""Compute loss for model.
		Args:
			x: Tensor with data

		Returns:
			Tuple[loss, dict(loss_component_name -> loss_component)]
		"""

	@abc.abstractmethod
	def encode(self, x):
		"""Compute latent representation."""

	@abc.abstractmethod
	def decode(self, z):
		"""Compute reconstruction."""


class TopologicalSignatureDistance(nn.Module):
	"""Topological signature."""

	def __init__(self, sort_selected=False, use_cycles=False,
				 match_edges=None):
		"""Topological signature computation.

		Args:
			p: Order of norm used for distance computation
			use_cycles: Flag to indicate whether cycles should be used
				or not.
		"""
		super().__init__()
		self.use_cycles = use_cycles

		self.match_edges = match_edges

		# if use_cycles:
		#	 use_aleph = True
		# else:
		#	 if not sort_selected and match_edges is None:
		#		 use_aleph = True
		#	 else:
		#		 use_aleph = False

		# if use_aleph:
		#	 print('Using aleph to compute signatures')
		##self.signature_calculator = AlephPersistenHomologyCalculation(
		##	compute_cycles=use_cycles, sort_selected=sort_selected)
		# else:
		print('Using python to compute signatures')
		self.signature_calculator = PersistentHomologyCalculation()

	def _get_pairings(self, distances):
		pairs_0, pairs_1 = self.signature_calculator(
			distances.detach().cpu().numpy())

		return pairs_0, pairs_1

	def _select_distances_from_pairs(self, distance_matrix, pairs):
		# Split 0th order and 1st order features (edges and cycles)
		pairs_0, pairs_1 = pairs
		selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

		if self.use_cycles:
			edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
			edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
			edge_differences = edges_2 - edges_1

			selected_distances = torch.cat(
				(selected_distances, edge_differences))

		return selected_distances

	@staticmethod
	def sig_error(signature1, signature2):
		"""Compute distance between two topological signatures."""
		return ((signature1 - signature2)**2).sum(dim=-1)

	@staticmethod
	def _count_matching_pairs(pairs1, pairs2):
		def to_set(array):
			return set(tuple(elements) for elements in array)
		return float(len(to_set(pairs1).intersection(to_set(pairs2))))

	@staticmethod
	def _get_nonzero_cycles(pairs):
		all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
		return np.sum(np.logical_not(all_indices_equal))

	# pylint: disable=W0221
	def forward(self, distances1, distances2):
		"""Return topological distance of two pairwise distance matrices.

		Args:
			distances1: Distance matrix in space 1
			distances2: Distance matrix in space 2

		Returns:
			distance, dict(additional outputs)
		"""
		pairs1 = self._get_pairings(distances1)
		pairs2 = self._get_pairings(distances2)

		distance_components = {
			'metrics.matched_pairs_0D': self._count_matching_pairs(
				pairs1[0], pairs2[0])
		}
		# Also count matched cycles if present
		if self.use_cycles:
			distance_components['metrics.matched_pairs_1D'] = \
				self._count_matching_pairs(pairs1[1], pairs2[1])
			nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
			nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
			distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
			distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

		if self.match_edges is None:
			sig1 = self._select_distances_from_pairs(distances1, pairs1)
			sig2 = self._select_distances_from_pairs(distances2, pairs2)
			distance = self.sig_error(sig1, sig2)

		elif self.match_edges == 'symmetric':
			sig1 = self._select_distances_from_pairs(distances1, pairs1)
			sig2 = self._select_distances_from_pairs(distances2, pairs2)
			# Selected pairs of 1 on distances of 2 and vice versa
			sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
			sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

			distance1_2 = self.sig_error(sig1, sig1_2)
			distance2_1 = self.sig_error(sig2, sig2_1)

			distance_components['metrics.distance1-2'] = distance1_2
			distance_components['metrics.distance2-1'] = distance2_1

			distance = distance1_2 + distance2_1

		elif self.match_edges == 'random':
			# Create random selection in oder to verify if what we are seeing
			# is the topological constraint or an implicit latent space prior
			# for compactness
			n_instances = len(pairs1[0])
			pairs1 = torch.cat([
				torch.randperm(n_instances)[:, None],
				torch.randperm(n_instances)[:, None]
			], dim=1)
			pairs2 = torch.cat([
				torch.randperm(n_instances)[:, None],
				torch.randperm(n_instances)[:, None]
			], dim=1)

			sig1_1 = self._select_distances_from_pairs(
				distances1, (pairs1, None))
			sig1_2 = self._select_distances_from_pairs(
				distances2, (pairs1, None))

			sig2_2 = self._select_distances_from_pairs(
				distances2, (pairs2, None))
			sig2_1 = self._select_distances_from_pairs(
				distances1, (pairs2, None))

			distance1_2 = self.sig_error(sig1_1, sig1_2)
			distance2_1 = self.sig_error(sig2_1, sig2_2)
			distance_components['metrics.distance1-2'] = distance1_2
			distance_components['metrics.distance2-1'] = distance2_1

			distance = distance1_2 + distance2_1

		return distance, distance_components


class UnionFind:
	'''
	An implementation of a Union--Find class. The class performs path
	compression by default. It uses integers for storing one disjoint
	set, assuming that vertices are zero-indexed.
	'''

	def __init__(self, n_vertices):
		'''
		Initializes an empty Union--Find data structure for a given
		number of vertices.
		'''

		self._parent = np.arange(n_vertices, dtype=int)

	def find(self, u):
		'''
		Finds and returns the parent of u with respect to the hierarchy.
		'''

		if self._parent[u] == u:
			return u
		else:
			# Perform path collapse operation
			self._parent[u] = self.find(self._parent[u])
			return self._parent[u]

	def merge(self, u, v):
		'''
		Merges vertex u into the component of vertex v. Note the
		asymmetry of this operation.
		'''

		if u != v:
			self._parent[self.find(u)] = self.find(v)

	def roots(self):
		'''
		Generator expression for returning roots, i.e. components that
		are their own parents.
		'''

		for vertex, parent in enumerate(self._parent):
			if vertex == parent:
				yield vertex


class PersistentHomologyCalculation:
	def __call__(self, matrix):

		n_vertices = matrix.shape[0]
		uf = UnionFind(n_vertices)

		triu_indices = np.triu_indices_from(matrix)
		edge_weights = matrix[triu_indices]
		edge_indices = np.argsort(edge_weights, kind='stable')

		# 1st dimension: 'source' vertex index of edge
		# 2nd dimension: 'target' vertex index of edge
		persistence_pairs = []

		for edge_index, edge_weight in \
				zip(edge_indices, edge_weights[edge_indices]):

			u = triu_indices[0][edge_index]
			v = triu_indices[1][edge_index]

			younger_component = uf.find(u)
			older_component = uf.find(v)

			# Not an edge of the MST, so skip it
			if younger_component == older_component:
				continue
			elif younger_component > older_component:
				uf.merge(v, u)
			else:
				uf.merge(u, v)

			if u < v:
				persistence_pairs.append((u, v))
			else:
				persistence_pairs.append((v, u))

		# Return empty cycles component
		return np.array(persistence_pairs), np.array([])


class AlephPersistenHomologyCalculation():
	def __init__(self, compute_cycles, sort_selected):
		"""Calculate persistent homology using aleph.

		Args:
			compute_cycles: Whether to compute cycles
			sort_selected: Whether to sort the selected pairs using the
				distance matrix (such that they are in the order of the
				filteration)
		"""
		self.compute_cycles = compute_cycles
		self.sort_selected = sort_selected

	def __call__(self, distance_matrix):
		"""Do PH calculation.

		Args:
			distance_matrix: numpy array of distances

		Returns: tuple(edge_featues, cycle_features)
		"""
		import aleph
		if self.compute_cycles:
			pairs_0, pairs_1 = aleph.vietoris_rips_from_matrix_2d(
				distance_matrix)
			pairs_0 = np.array(pairs_0)
			pairs_1 = np.array(pairs_1)
		else:
			pairs_0 = aleph.vietoris_rips_from_matrix_1d(
				distance_matrix)
			pairs_0 = np.array(pairs_0)
			# Return empty cycles component
			pairs_1 = np.array([])

		if self.sort_selected:
			selected_distances = \
				distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]
			indices_0 = np.argsort(selected_distances)
			pairs_0 = pairs_0[indices_0]
			if self.compute_cycles:
				cycle_creation_times = \
					distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
				cycle_destruction_times = \
					distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
				cycle_persistences = \
					cycle_destruction_times - cycle_creation_times
				# First sort by destruction time and then by persistence of the
				# create cycles in order to recover original filtration order.
				indices_1 = np.lexsort(
					(cycle_destruction_times, cycle_persistences))
				pairs_1 = pairs_1[indices_1]

		return pairs_0, pairs_1
