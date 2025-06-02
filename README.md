# SEAGALL

![alt text](https://github.com/gmalagol10/seagall/blob/main/seagall/GnnFig1.png?raw=true)


Single-cell Explainable geometry Aware Graph Attention Learning pipeLine. Tool from the paper:

Geometry aware graph attention networks to explain single-cell chromatin state and gene expression; Gabriele Malagoli, Patrick Hanel, Anna Danese, Guy Wolf & Maria Colome-Tatche
bioRxiv 2025.05.29.656611; doi: https://doi.org/10.1101/2025.05.29.656611

Data and notebook to reproduce all the results are available at https://github.com/gmalagol10/seagall/tree/main/reproducibility
## Installation

```bash

conda create -n my_env python=3.10.14

conda activate my_env

pip install git+https://github.com/gmalagol10/seagall
```

## Usage

```python
import seagall as sgl
import scanpy as sc
import matplotlib.pyplot as plt

adata=sc.read_h5ad("MouseBrain_GEX.h5ad")

sgl.ee.geometrical_graph(adata, target_label="CellType", path="SEAGALL")

sgl.ee.classify_and_explain(adata, target_label="CellType", path="SEAGALL", hypopt=0.25)

for i, ct in enumerate(sorted(list(set(adata.obs.CellType)))):
    print(i)
    plt.scatter(x=range(0, len(adata.var)), y=sorted(adata.var[f"SEAGALL_Importance_for_{ct}"])[::-1], c=colors_to_use_bright[i])
plt.xscale("log")
plt.yscale("log")
```


## Version History

* 0.1
    * Initial Release


## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details


