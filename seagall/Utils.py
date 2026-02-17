import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from typing import List, Any

warnings.filterwarnings("ignore")


def intersection(lists: List[List[Any]]) -> np.ndarray:
	"""
	Returns the intersection of all sublists.
	
	Parameters
	----------
	lists : List[List[Any]]
		A list of lists whose intersection is to be found.
	
	Returns
	-------
	np.ndarray
		Array containing elements common to all sublists.
	"""
	if not lists:
		return np.array([])
	return np.array(list(set.intersection(*map(set, lists))))


def flat_list(nested_lists: List[List[Any]]) -> np.ndarray:
	"""
	Flattens a list of lists and removes duplicates.
	
	Parameters
	----------
	nested_lists : List[List[Any]]
		A list of lists to be flattened.
	
	Returns
	-------
	np.ndarray
		A flattened array with unique elements.
	"""
	return np.array(list(set(item for sublist in nested_lists for item in sublist)))


def most_common(items: List[Any]) -> Any:
	"""
	Returns the most common element in a list.
	
	Parameters
	----------
	items : List[Any]
		List of elements.
	
	Returns
	-------
	Any
		The element that appears most frequently.
	"""
	if not items:
		return None
	items = list(items)
	return max(set(items), key=items.count)


def process_target_label(adata: sc.AnnData, target_label: str) -> None:
	"""
	Processes the target label in an AnnData object for stratified splitting.
	
	Parameters
	----------
	adata : sc.AnnData
		The AnnData object containing cell metadata.
		
	target_label : str
		Column name in `adata.obs` containing categorical target labels.
	
	Returns
	-------
	None
		Modifies `adata.obs` in place by creating an integer-mapped `target` column.
		Also adds forward and inverse label mappings in `adata.uns`.
	"""
	# Replace NaNs, ensure string type, remove dangerous chars
	adata.obs[target_label] = adata.obs[target_label].astype(str)
	adata.obs[target_label] = adata.obs[target_label].str.replace("/","_")
	adata.obs[target_label] = adata.obs[target_label].fillna("unknown").astype(str)

	# Create sorted list of unique labels and mappings
	unique_labels = sorted(adata.obs[target_label].unique())
	label_map = {label: str(i) for i, label in enumerate(unique_labels)}
	inv_map = {str(i): label for label, i in label_map.items()}

	# Store mappings in adata.uns
	adata.uns["map"] = label_map
	adata.uns["inv_map"] = inv_map

	# Create integer-mapped target column
	adata.obs["target"] = adata.obs[target_label].map(label_map).astype(int)

