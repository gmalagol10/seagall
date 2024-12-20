import episcanpy as epi
import scanpy as sc
import numpy as np
import scipy

import EmbeddExplain as ee

colors_to_use=[(0.34550725069638827, 0.4203708006658883, 0.9696902293486781),
 (0.9893800026041992, 0.378955911742755, 0.21756841368122667),
 (0.3959642074605608, 0.24823947872676938, 0.4016676539297192),
 (0.9937826924994482, 0.4211527500079969, 0.8812994030921271),
 (0.4140058397372807, 0.9619317608252869, 0.3109026417629064),
 (0.2286247431221609, 0.6437632542888629, 0.4081322805120583),
 (0.25003260615661993, 0.938691496932296, 0.9192515923797947),
 (0.7646511697684856, 0.24254983894398235, 0.7085129830496552),
 (0.3017721221187747, 0.6522700618245787, 0.9844707721904342),
 (0.21782892631529166, 0.2854088109905996, 0.7174819557293214),
 (0.9991896546476063, 0.4844986464266022, 0.5344476773522967),
 (0.8919236560192338, 0.7949117963224906, 0.7730486745511909),
 (0.6428145648345002, 0.31108252586505475, 0.2041098261347507),
 (0.20269999368314223, 0.7748472379028853, 0.6824940025160084),
 (0.9819923318374866, 0.6965816490867496, 0.21263663000131983),
 (0.809020033101876, 0.23056728504993104, 0.9541856467035792),
 (0.9191255237720919, 0.20589563833687236, 0.4656154484972612),
 (0.4066816185888487, 0.7781338620666193, 0.22302116197384453),
 (0.9393586762460103, 0.9614694589148117, 0.22619331869433382),
 (0.29226458048518345, 0.42830335364029093, 0.6916335500700534)]

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
	
	if features_file[-10:]=="narrowPeak":
		epi.pp.nucleosome_signal(adata, fragments_file)
		epi.pp.tss_enrichment(adata, gtf=gtf_file, source=source, fragments=fragments_file)

	return adata
  
def qc_filtering(adata, omic="ATAC"):

	'''
	Function to create a sc-ATACseq count matrix. It's a wrapping around the main function of EpiScanpy.

 	Parameters
    ----------

	adata : raw count matrix to process

	Output
	------
	
	AnnData object after QC and filtering

	'''

	epi.pp.qc_stats(adata, verbose=False)
		
	if omic=="GEX":
		adata.var["MT"] = adata.var.index.str.startswith(("MT","mt"))
		sc.pp.calculate_qc_metrics(adata, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
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
			adata=adata[adata.obs.tss_enrichment_score > 2]
		except:
			pass

	print("Adata's shape after cells and features filtering:", adata.shape, flush=True)

	if omic=="GEX":
		max_mt = np.quantile(adata.obs["pct_counts_MT"].dropna(), 0.9)
		adata = adata[adata.obs["pct_counts_MT"]<=max_mt]
		print("Adata's shape after MT filtering:", adata.shape, flush=True)
		
		adata=adata[adata.obs.predicted_doublet==False]
		print("Adata's shape after doublets filtering:", adata.shape, flush=True)
   
	epi.pp.normalize_total(adata)
		
	sc.pp.highly_variable_genes(adata, flavor='seurat_v3')
	if omic=="GEX":
		min_var = np.quantile(adata.var.variances_norm, 0.9)
	else:
		min_var = np.quantile(adata.var.variances_norm, 0.8)
	adata = adata[:, adata.var.variances_norm>min_var]
	if adata.shape[1] > 30000:
		features=adata.var.sort_values(by="variances_norm")[::-1][:30000].index
		adata=adata[:, features]

	print("Adata's shape after HVG filtering:", adata.shape, flush=True)
	
	epi.pp.log1p(adata)

	adata = adata[:, adata.X.max(axis=0)>0]
	adata = adata[adata.X.max(axis=1)>0]
	print("Adata's shape after max 0 filtering:", adata.shape, flush=True)
	   
	return adata 
 
		
def preprocessing(adata, target_label=None, representantion=None, omic="ATAC", model_name="Pappo"):

	'''
	Function to create a sc-ATACseq count matrix. It's a wrapping around the main function of EpiScanpy.

 	Parameters
    ----------

	adata : raw count matrix to process

	target_label : label to take into account when splitting the dataset in train/val/test for class unbalance

	representation : method to apply for DR, either None or GRAE

	omic : single-cell tehconolgy used to profile cells. Either ATAC or GEX

	model_name : name to use to save the model during the training

	Output
	------
	
	AnnData object after QC, filtering and embedding

	'''


	print("QC and filtering", flush=True)

	adata=qc_filtering(adata, omic=omic)
	
	if representantion != None:	
		if target_label != None:
			adata = adata[adata.obs[target_label].astype(str)!="nan"]
			adata = adata[adata.obs.groupby(target_label).filter(lambda x : len(x)>50)[target_label].index,:]
			mymap = dict([(y, str(x)) for x,y in enumerate(sorted(set(adata.obs[target_label])))])
			inv_map = {v : k for k, v in mymap.items()}
			adata.uns["map"] = mymap
			adata.uns["inv_map"] = inv_map
			adata.obs["target"] = [mymap[x] for x in adata.obs[target_label]]
			y = adata.obs["target"].astype(int).to_numpy()
		else:
			y=[]
	
		print(f"Embedding with {representantion}", flush=True)
		ee.embbedding_and_graph(adata=adata, y=y, representantion=representantion, model_name=f"{model_name}_{representantion}_DR")
			
	return adata

  
   
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))


def flat_list(l):
	return np.array(list(set([item for sublist in l for item in sublist])))


def most_common(l):
	l=list(l)
	return max(set(l), key=l.count)

	
def my_polynom(params, x):
	n = len(params)
	assert n > 0
	x_vec = np.array([np.power(x, i) for i in range(n - 1, -1, -1)])
	return x_vec.T @ np.array(params)


