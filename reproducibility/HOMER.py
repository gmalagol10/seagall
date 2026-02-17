import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# HOMER ++++++++++++++++++++++++
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))

dataset = sys.argv[1]

ds_infos=pd.read_csv("Tables/Datasets_infos.tsv", sep="\t", index_col=0)
ds_infos=ds_infos[ds_infos["FsSs"]!="GEX"]
hs1="../AnnotRef/hs/T2T/chm13v2.0.fa"
ds_infos["Genome"]=[hs1,"hg38",hs1,"hg38","mm10", hs1]
ds_infos=ds_infos[ds_infos["DSs"]==dataset]

featurespace=str(ds_infos["FsSs"].item())
label=str(ds_infos["LBs"].item())
genome=str(ds_infos["Genome"].item())

snp2=f"Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/SnapATAC2/{dataset}_{featurespace}_Features.tsv.gz"
snp2_imp=f"Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/SnapATAC2/{dataset}_{featurespace}_FeaturesPvsAdj.tsv.gz"
xai=f"Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/{dataset}_{featurespace}_GRAE_kNN_{label}_HPO_XAIFeatures.tsv.gz"
xai_imp=f"Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/{dataset}_{featurespace}_GRAE_kNN_{label}_HPO_XAIFeatImpCM.tsv.gz"

snp2=pd.read_csv(snp2, sep="\t", index_col=0)[:500]
snp2_imp=pd.read_csv(snp2_imp, sep="\t", index_col=0)
xai=pd.read_csv(xai, sep="\t", index_col=0)[:500]
xai_imp=pd.read_csv(xai_imp, sep="\t", index_col=0)

snp2_path=f"Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/SnapATAC2/MotifAnalysis/HOMER_SnapATAC2only"
xai_path=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/MotifAnalysis/HOMER_XAIonly"


Path(snp2_path).mkdir(parents=True, exist_ok=True)
Path(xai_path).mkdir(parents=True, exist_ok=True)
inter=intersection([xai.columns, snp2.columns])

for col in inter:

	only_xai=xai_imp[list(set(xai[col]).difference(snp2[col].dropna()))].mean().sort_values()[::-1][:100]
	a=list(set(snp2[col].dropna()).difference(xai[col]))
	b=pd.concat([pd.DataFrame(snp2_imp[col]).rename({col : "Imp"}, axis=1), pd.DataFrame(snp2[col]).rename({col : "Feat"}, axis=1)], axis=1).dropna()
	only_snp2=b.sort_values(by="Imp").set_index("Feat").loc[a][:100]
	print(col, len(only_xai), len(only_snp2), flush=True)   


    xaiffile=f"{xai_path}/{col}.bed"
	pd.DataFrame([g.split("_") for g in list(only_xai.index)]).to_csv(xaiffile, sep="\t", header=None, index=None)
	os.system(f"findMotifsGenome.pl {xaiffile} {genome} {xai_path}/{col} -p 8")


	snp2ffile=f"{snp2_path}/{col}.bed"
	pd.DataFrame([g.split("_") for g in list(only_snp2.index)]).to_csv(snp2ffile, sep="\t", header=None, index=None)
	os.system(f"findMotifsGenome.pl {snp2ffile} {genome} {snp2_path}/{col} -p 8")
