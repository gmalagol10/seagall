import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import os

# HOMER ++++++++++++++++++++++++
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))

print(os.getcwd(), flush=True)
ds_infos=pd.read_csv("../Datasets_infos.tsv", sep="\t", index_col=0)
ds_infos=ds_infos[ds_infos["FsSs"]!="GEX"]
hs1="../../AnnotRef/hs/T2T/chm13v2.0.fa"
ds_infos["Genome"]=[hs1,"hg38",hs1,"hg38","mm10", hs1]

for dataset, featurespace, label, name, genome in zip(ds_infos["DSs"], ds_infos["FsSs"], ds_infos["LBs"], ds_infos["Names"], ds_infos["Genome"]):
	xai=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/{dataset}_{featurespace}_GRAE_kNN_{label}_HPO_XAIFeatures.tsv.gz"
	xai_imp=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/{dataset}_{featurespace}_GRAE_kNN_{label}_HPO_XAIFeatImpCM.tsv.gz"
	de=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/DE/{dataset}_{featurespace}_GRAE_kNN_{label}_DEFeatures.tsv.gz"
	de_imp=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/DE/{dataset}_{featurespace}_GRAE_kNN_{label}_DEFeaturesPvsAdj.tsv.gz"
	
	xai_path=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/MotifAnalysis/HOMER_XAIonly"
	de_path=f"../Datasets/{dataset}/FeatureSpaces/{featurespace}/XAI/DE/MotifAnalysis/HOMER_DEonly"
	Path(xai_path).mkdir(parents=True, exist_ok=True)
	Path(de_path).mkdir(parents=True, exist_ok=True)

	if os.path.isfile(xai) == True and os.path.isfile(de) == True:
		xai=pd.read_csv(xai, sep="\t", index_col=0)[:500]
		de=pd.read_csv(de, sep="\t", index_col=0)[:500]
		xai_imp=pd.read_csv(xai_imp, sep="\t", index_col=0)
		de_imp=pd.read_csv(de_imp, sep="\t", index_col=0)
		inter=intersection([xai.columns, de.columns])
		for col in inter:
			only_xai=xai_imp[list(set(xai[col]).difference(de[col].dropna()))].mean().sort_values()[::-1][:100]
			a=list(set(de[col].dropna()).difference(xai[col]))
			b=pd.concat([pd.DataFrame(de_imp[col]).rename({col : "Imp"}, axis=1), pd.DataFrame(de[col]).rename({col : "Feat"}, axis=1)], axis=1).dropna()
			only_de=b.sort_values(by="Imp").set_index("Feat").loc[a][:100]
			
			print(col, len(only_xai), len(only_de), flush=True)   

			xaiffile=f"{xai_path}/{col}.bed"
			pd.DataFrame([g.split("_") for g in list(only_xai.index)]).to_csv(xaiffile, sep="\t", header=None, index=None)
			os.system(f"findMotifsGenome.pl {xaiffile} {genome} {xai_path}/{col} -p 10")

			deffile=f"{de_path}/{col}.bed"
			pd.DataFrame([g.split("_") for g in list(only_de.index)]).to_csv(deffile, sep="\t", header=None, index=None)
			os.system(f"findMotifsGenome.pl {deffile} {genome} {de_path}/{col} -p 10")
