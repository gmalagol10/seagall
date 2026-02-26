import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as mc
import episcanpy as epi
import scanpy as sc
import pandas as pd
import numpy as np
import colorsys
import sklearn
import anndata
import scipy
import time

import ML_utils as mlu

#colors=distinctipy.get_colors(10, pastel_factor=1, colorblind_type="Deuteranomaly")
colors_to_use_bright=[(0.34550725069638827, 0.4203708006658883, 0.9696902293486781),
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


selected_colors=[(0.4502835075797788, 0.4137972479772782, 0.6667776141888558),
 (0.8289219715510678, 0.41974959068753226, 0.5543085353140962),
 (0.6385027637736282, 0.8403347267917161, 0.5859308575921919)]

colors_to_use_pastel=[(0.5665516830630946, 0.5037138904751852, 0.5075215212500083),
 (0.9949012052279167, 0.7553237006975844, 0.5055095027008268),
 (0.6013710840316495, 0.6764106146805113, 0.9895579386673582),
 (0.998816092724718, 0.7071878014442006, 0.8979090191443285),
 (0.5429870427410571, 0.750719571440085, 0.6124765491924715),
 (0.5084695919264852, 0.5093366344721296, 0.9990436978593071),
 (0.8246194014837736, 0.9846839170837028, 0.5089892119497516),
 (0.5052972476002842, 0.5444186786858283, 0.7812964942323839),
 (0.5124601533942488, 0.9750384643760528, 0.9697621879182152),
 (0.9728598314959997, 0.5013134514042519, 0.8469604310077152)]

	
def create_count_matrix(fragments_file : str, valid_bcs : list, features_space : str, features_file=None, gtf_file=None, source=None, meta=None):

	if not isinstance(fragments_file, str) or not isinstance(valid_bcs, list) or not isinstance(features_space, str):
		print(type(fragments_file), type(valid_bcs), type(features_space), flush=True)
		raise TypeError
	
	if features_space == "Window":
		adata = epi.ct.window_mtx(fragments_file=fragments_file, valid_bcs=valid_bcs, fast=False)
		
	elif features_space != "Window":
		if features_file[-6:]=="gtf.gz" or features_file[-3:]=="gtf":
			if features_space=="GA":
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs)
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
								 source=source, gene_type=[features_space])
								 	
		elif features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed":
			adata = epi.ct.peak_mtx(fragments_file=fragments_file, peak_file=features_file, valid_bcs=valid_bcs)
			adata.var.index=["_".join([str(adata.var.iloc[i][0]),str(adata.var.iloc[i][1]),str(adata.var.iloc[i][2])]) for i in range(len(adata.var.index))]
	
	adata.X=scipy.sparse.csr_matrix(adata.X, dtype="float32")
	adata.var_names_make_unique()
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
	adata.var_names_make_unique()
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
	epi.pp.normalize_total(adata)
	epi.pp.log1p(adata)

	adata = adata[:, adata.X.max(axis=0)>0]
	adata = adata[adata.X.max(axis=1)>0]
	print("Adata's shape after max 0 filtering:", adata.shape, flush=True)
	   
	return adata 
 
		
def preprocessing(adata, target_label=None, representation=None, omic="ATAC", model_name="Pappo"):
	adata.var_names_make_unique()
	print("QC and filtering", flush=True)

	adata=qc_filtering(adata, omic=omic)
	
	print("PCA, UMAP and clustering", flush=True)
	sc.pp.pca(adata)
	sc.pp.neighbors(adata, method="umap")
	sc.tl.umap(adata)
	sc.tl.leiden(adata)
	
	print("Representation", flush=True)
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
	
	if representation != None:
		print(f"Embedding with {representation}", flush=True)
		adata.obsp[f"{representation}_kNN"], adata.obsm[f"X_{representation}"] = mlu.embbedding_and_graph(adata=adata, y=y, 
																					   representation=representation, model_name=f"{model_name}_{representation}_DR")
			
	return adata

	
def upper_tri_masking(A):
	m = A.shape[0]
	r = np.arange(m)
	mask = r[:,None] < r
	return A[mask]	
	
def divide_string(string, parts):
   # Determine the length of each substring
   part_length = len(string) // parts

   # Divide the string into 'parts' number of substrings
   substrings = [string[i:i + part_length] for i in range(0, len(string), part_length)]

   # If there are any leftover characters, add them to the last substring
   if len(substrings) > parts:
		  substrings[-2] += substrings[-1]
		  substrings.pop()

   return substrings
   
   
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


def lighten_color(colors, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    new_colors=[]
    for c in colors:
        c = colorsys.rgb_to_hls(*mc.to_rgb(c)) 
        new_colors.append(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))
    return new_colors


def powlaw(x, a, b) :
    return a * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b
def explaw(x, a, b, c):
    return a * np.exp(-b * x) + c
    
def curve_fit_log(xdata, ydata) :
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = scipy.optimize.curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)
def curve_fit_exp(xdata, ydata):
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = scipy.optimize.curve_fit(explaw, xdata_log, ydata_log, maxfev = 8000)

    ydatafit_log = np.power(10, explaw(xdata_log, *popt_log))

    return (popt_log, pcov_log, ydatafit_log)
