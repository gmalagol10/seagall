import warnings
warnings.filterwarnings("ignore")

import episcanpy as epi
import scanpy as sc
import numpy as np
import scipy

from . import EmbeddExplain as ee

def create_count_matrix(fragments_file : str, valid_bcs : list, features_space : str, features_file=None, gtf_file=None, source=None, meta=None):

	'''
	Function to create a sc-ATACseq count matrix. It's a wrapping around the main function of EpiScanpy.

 	Parameters
    ----------

	fragments_file : path to fragments file
	
	valid_bcs : list of valid barcodes

	features_space : name of the features' space (e.g. Peak, GA, etc)

	features_file : either path to BED file containing genomic coordinate or path to gtf file containing genome annotation

	gtf_file : path to gtf file containing genome annotation

	source : source of the genome information (e.g. HAVANA, BestRefSeq)
	
	meta : object of class pandas.DataFrame storing metadata bout the barcodes. The index must be the barcodes


	Output
	------
	
	AnnData object

	'''

	if not isinstance(fragments_file, str) or not isinstance(valid_bcs, list) or not isinstance(features_space, str):
		print(type(fragments_file), type(valid_bcs), type(features_space), flush=True)
		raise TypeError
	
	if features_space == "Window":
		adata = epi.ct.window_mtx(fragments_file=fragments_file, valid_bcs=valid_bcs, fast=False)
		
	elif features_space != "Window":
		if features_file[-6:]=="gtf.gz" or features_file[-3:]=="gtf":
			if features_space=="GA":
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs, fast=False)
			else:
				data={}
				data["tRNA"]=[30,80] #POL3
				data["rRNA"]=[30,80] #POL1/POL3
				data["protein_coding"]=[5000,2000] #POL2
				data["lncRNA"]=[30,80] #POL3
				data["miRNA"]=[5000,2000] #POl2
				data["telomerase_RNA"]=[5000,2000] #POL2
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs, 
								 upstream=data[features_space][0], downstream=data[features_space][1], 
								 source=source, gene_type=[features_space], fast=False)
								 	
		elif features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed":
			adata = epi.ct.peak_mtx(fragments_file=fragments_file, peak_file=features_file, valid_bcs=valid_bcs, normalized_peak_size=None, fast=False)
			adata.var.index=["_".join([str(adata.var.iloc[i][0]),str(adata.var.iloc[i][1]),str(adata.var.iloc[i][2])]) for i in range(len(adata.var.index))]
	
	adata.var_names_make_unique()
	adata.X=scipy.sparse.csr_matrix(adata.X, dtype="float32")
	adata.var.columns = adata.var.columns.astype(str)
	adata.obs.columns = adata.obs.columns.astype(str)
	adata.var = adata.var.rename(columns={"0": "chr", "1": "start", "2" : "stop"})
	adata.var_names_make_unique(join="_")
	adata = adata[:,adata.var[adata.var['chr'].str.match('chr')].index]
	adata = adata[:,adata.var[~adata.var['chr'].str.match('chrM')].index]
	if features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed" or features_space == "Window":
		a = [a for a in adata.var.index if len(a.split("_"))!=3]
		adata = adata[ :, adata.var.loc[list(set(adata.var.index) - set(a))].index ]
		del a
	
	if meta is not None:
		adata.obs = meta.loc[adata.obs.index]
	
	epi.pp.nucleosome_signal(adata, fragments_file)
	if gtf_file != None:
		epi.pp.tss_enrichment(adata, gtf=gtf_file, fragments=fragments_file)

	return adata
  
def qc_filtering(adata, omic="none"):

	'''
	Function to create a sc-ATACseq count matrix. It's a wrapping around the main function of EpiScanpy.

 	Parameters
    ----------

	adata : raw count matrix to process

	omic : single-cell tehconolgy used to profile cells. Either ATAC or GEX

	Output
	------
	
	AnnData object after QC and filtering

	'''

	adata.var_names_make_unique()
	adata.X==scipy.sparse.csr_matrix(adata.X, dtype="float32")
	epi.pp.qc_stats(adata, verbose=False)
		
	if omic=="GEX":
		mt_gene_mask = np.flatnonzero([gene.startswith(("MT","mt")) for gene in adata.var.index])
		adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']
		sc.pp.scrublet(adata)
		
	min_features = 10**np.quantile(adata.obs["log_n_features"], 0.05)
	max_features = 10**np.quantile(adata.obs["log_n_features"], 0.95)
	min_cells = 10**np.quantile(adata.var["log_n_cells"], 0.05)
	max_cells = 10**np.quantile(adata.var["log_n_cells"], 0.95)

	print("Adata's shape:", adata.shape, flush=True)
	
	epi.pp.set_filter(adata, "n_features", min_threshold=min_features, max_threshold=max_features, verbose=False)
	epi.pp.set_filter(adata, "n_cells", min_threshold=min_cells, max_threshold=max_cells, verbose=False)
	adata = epi.pp.apply_filters(adata, verbose=False)
	if omic=="ATAC":
		try:
			adata=adata[adata.obs.nucleosome_signal < 2]
		except:
			print("Could not filter cells based on nucleosome signal", flush=True)
			pass
		try:
			adata=adata[adata.obs.tss_enrichment_score > 2]
		except:
			print("Could not filter cells based on TSS enrichment", flush=True)
			pass

	print("Adata's shape after cells and features filtering:", adata.shape, flush=True)

	if omic=="GEX":
		max_mt = np.quantile(adata.obs["mt_frac"].dropna(), 0.9)
		adata = adata[adata.obs["mt_frac"]<=max_mt]
		print("Adata's shape after MT filtering:", adata.shape, flush=True)
		
		adata=adata[adata.obs.predicted_doublet==False]
		print("Adata's shape after doublets filtering:", adata.shape, flush=True)
   
	sc.pp.highly_variable_genes(adata, flavor='seurat_v3')
	adata.layers["counts"]	= adata.X.copy()
	sc.pp.normalize_total(adata)
		
	if omic=="GEX":
		min_var = np.quantile(adata.var.variances_norm, 0.9)
	else:
		min_var = np.quantile(adata.var.variances_norm, 0.8)
	adata = adata[:, adata.var.variances_norm>min_var]
	if adata.shape[1] > 30000:
		features=adata.var.sort_values(by="variances_norm")[::-1][:30000].index
		adata=adata[:, features]

	print("Adata's shape after HVG filtering:", adata.shape, flush=True)
	
	if omic=="GEX":	
		sc.pp.log1p(adata)

	adata = adata[:, adata.X.max(axis=0)>0]
	adata = adata[adata.X.max(axis=1)>0]
	print("Adata's final shape:", adata.shape, flush=True)

		
def preprocessing(adata, target_label=None, omic="none", path="SEAGALL", model_name="MySEAGALL"):

	'''
	Filtering and QC

 	Parameters
    ----------

	adata : raw count matrix to process

	target_label : label to take into account when splitting the dataset in train/val/test in case of class unbalance

	omic : single-cell tehconolgy used to profile cells. Either ATAC or GEX

	model_name : name to use to save the model during the training

	Output
	------
	
	AnnData object after QC, filtering and embedding

	'''


	print("QC and filtering", flush=True)

	qc_filtering(adata, omic=omic)
	

	print(f"Embedding with GRAE", flush=True)
	ee.geometrical_graph(adata=adata, target_label=target_label, path=path, model_name=model_name)
			
  
   
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))


def flat_list(l):
	return np.array(list(set([item for sublist in l for item in sublist])))


def most_common(l):
	l=list(l)
	return max(set(l), key=l.count)
