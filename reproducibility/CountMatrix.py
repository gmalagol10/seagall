import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import episcanpy as epi
import anndata as ad

import os
import scipy
import glob
import sys
import time
import sys

import Utils as ut

from pathlib import Path
import time

print(f"CM script started at: ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

path=sys.argv[1]
gtf_file=sys.argv[2]
frag_file=sys.argv[3]
features_space=sys.argv[4]
features_file=sys.argv[5]
metadata=sys.argv[6]
source=sys.argv[7]
rep=sys.argv[8]
target_label=sys.argv[9]

names=["Script -->","Path -->", "GTF -->","Fragments file -->","Features space -->","Features file -->", "Metadata -->", "Source -->", "Representation -->", "Target label -->"]

for nm,arg in zip(names,sys.argv):
	print(nm, arg, flush=True)

folder="/".join(path.split("/")[:-1])
name=path.split("/")[-1]
print(folder, name)
			
Path(f"{folder}/CM").mkdir(parents=True, exist_ok=True)
object_name=f"{folder}/CM/{name}"

if features_space == "GEX":
	omic="GEX"
else:
	omic="ATAC"


if os.path.isfile(f"{object_name}_Raw.h5ad") == False: 
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Building {object_name}_Raw.h5ad", flush=True)
	
	meta=pd.read_csv(metadata, sep=",", index_col=0)
	valid_barcodes=sorted(list(meta.index))
	adata=ut.create_count_matrix(fragments_file=frag_file, valid_bcs=valid_barcodes, features_space=features_space, features_file=features_file, gtf_file=gtf_file, source=source, meta=meta)
	del meta
	
	print(adata, f"\nNon zero elements: {adata.X.count_nonzero()}", flush=True)
	adata.write(f"{object_name}_Raw.h5ad", compression="gzip")

	adata=ut.preprocessing(adata=adata, target_label=target_label, representation=rep, omic=omic,  model_name=f"{object_name}_{rep}_DR")
	
	adata.write(f"{object_name}_Def.h5ad", compression="gzip")
	del adata
else:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"{object_name}_Raw.h5ad already exists, searching Def.h5ad", flush=True)

if os.path.isfile(f"{object_name}_Raw.h5ad") == True and os.path.isfile(f"{object_name}_Def.h5ad") == False:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"Processing {object_name}_Raw.h5ad", flush=True)
	
	adata=epi.read_h5ad(f"{object_name}_Raw.h5ad")
	print(adata, f"\nNon zero elements: {adata.X.count_nonzero()}", flush=True)
	
	adata=ut.preprocessing(adata=adata, target_label=target_label, representation=rep, omic=omic, model_name=f"{object_name}_{rep}_DR")
	
	adata.write(f"{object_name}_Def.h5ad", compression="gzip")
	sys.exit()
else:
	print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), f"{object_name}_Def.h5ad and Def.h5ad already exist", flush=True)
