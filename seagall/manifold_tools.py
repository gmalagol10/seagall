# module adapted from https://github.com/KevinMoonLab/GRAE.git

"""PHATE and procrustes tools."""
import numpy as np
import phate
from sklearn.decomposition import PCA as SKPCA
from sklearn.pipeline import make_pipeline

from .  import base_model
from .procrustes import procrustes

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


PROC_THRESHOLD = 20000
PROC_BATCH_SIZE = 5000
PROC_LM = 1000


def fit_transform_procrustes(x, fit_transform_call, procrustes_batch_size, procrustes_lm):
	"""Fit model and transform data for larger datasets.

	If dataset has more than self.proc_threshold samples, then compute PHATE over
	mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
	which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.
	This last step is required since PHATE can lead to embeddings with different rotations or reflections
	depending on the batch.

	Args:
		x(BaseDataset): Dataset to fit and transform.
		fit_transform_call(callback): fit & transform method of an sklearn-style estimator.
		procrustes_batch_size(int): Batch size of procrustes approach.
		procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
		transform.

	Returns:
		ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

	"""
	lm_points = x[:procrustes_lm, :]  # Reference points included in all batches
	initial_embedding = fit_transform_call(lm_points)
	result = [initial_embedding]
	remaining_x = x[procrustes_lm:, :]
	while len(remaining_x) != 0:
		if len(remaining_x) >= procrustes_batch_size:
			new_points = remaining_x[:procrustes_batch_size, :]
			remaining_x = np.delete(remaining_x,
									np.arange(procrustes_batch_size),
									axis=0)
		else:
			new_points = remaining_x
			remaining_x = np.delete(remaining_x,
									np.arange(len(remaining_x)),
									axis=0)

		subsetx = np.vstack((lm_points, new_points))
		subset_embedding = fit_transform_call(subsetx)

		d, Z, tform = procrustes(initial_embedding,
								 subset_embedding[:procrustes_lm, :])

		subset_embedding_transformed = np.dot(
			subset_embedding[procrustes_lm:, :],
			tform['rotation']) + tform['translation']

		result.append(subset_embedding_transformed)
	return np.vstack(result)


class PHATE(phate.PHATE, base_model.BaseModel):
	"""Wrapper for PHATE to work with torch datasets.

	Also add procrustes transform when dealing with large datasets for improved scalability.
	"""

	def __init__(self, proc_threshold=PROC_THRESHOLD, procrustes_batch_size=PROC_BATCH_SIZE,
				 procrustes_lm=PROC_LM, **kwargs):
		"""Init.

		Args:
			proc_threshold(int): Threshold beyond which PHATE is computed over mini-batches of the data and batches are
			realigned with procrustes. Otherwise, vanilla PHATE is used.
			procrustes_batch_size(int): Batch size of procrustes approach.
			procrustes_lm (int): Number of anchor points present in all batches. Used as a reference for the procrustes
			transform.
			**kwargs: Any remaining keyword arguments are passed to the PHATE model.
		"""
		self.proc_threshold = proc_threshold
		self.procrustes_batch_size = procrustes_batch_size
		self.procrustes_lm = procrustes_lm
		self.comet_exp = None
		super().__init__(**kwargs)

	def fit_transform(self, x):
		"""Fit model and transform data.

		Overrides PHATE fit_transform method on datasets larger than self.proc_threshold to compute PHATE over
		mini-batches with procrustes realignment.

		Args:
			x(BaseDataset): Dataset to fit and transform.

		Returns:
			ndarray: Embedding of x.

		"""
		x, _ = x.numpy()

		if x.shape[0] < self.proc_threshold:
			result = super().fit_transform(x)
		else:
			logger.info('Fitting procrustes...')
			result = self.fit_transform_procrustes(x)
		return result

	def fit_transform_procrustes(self, x):
		"""Fit model and transform data for larger datasets.

		If dataset has more than self.proc_threshold samples, then compute PHATE over
		mini-batches. In each batch, add self.procrustes_lm samples (which are the same for all batches),
		which can be used to compute a  procrustes transform to roughly align all batches in a coherent manner.
		This last step is required since PHATE can lead to embeddings with different rotations or reflections
		depending on the batch.

		Args:
			x(BaseDataset): Dataset to fit and transform.

		Returns:
			ndarray: Embedding of x, which is the union of all batches aligned with procrustes.

		"""
		return fit_transform_procrustes(x, super().fit_transform, self.procrustes_batch_size, self.procrustes_lm)
