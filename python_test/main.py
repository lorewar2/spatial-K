
import scanpy as sc
from _consensus import consensus
import pandas as pd
import numpy as np

def main():
    # Load dataset
    adata_scrna = sc.read_csv("./data/scrna_filtered.csv")
    adata_spatial = sc.read_csv("./data/spatial_data.csv")
    print("Data Loaded!")

    # save origin
    adata_scrna.obs['batch'] = 'scrna'
    adata_spatial.obs['batch'] = 'spatial'
    # spatial data set cells in coloumns so transform
    adata_spatial = adata_spatial.T
    # we have to filter and match the genes on both datasets before concatation
    (scrna_indices, spatial_indices) = get_common_indices()

    print("Before Number of genes (columns) in adata_scrna:", adata_scrna.n_vars)
    print("Before Number of genes (columns) in adata_spatial:", adata_spatial.n_vars)

    # select and rearrange the matrices for common genes
    adata_scrna = adata_scrna[:, scrna_indices]
    adata_spatial = adata_spatial[:, spatial_indices]
    print("After Number of genes (columns) in adata_scrna:", adata_scrna.n_vars)
    print("After Number of genes (columns) in adata_spatial:", adata_spatial.n_vars)
    
    # preprocess scale and log2(x+1)
    sc.pp.scale(adata_spatial)
    sc.pp.log1p(adata_spatial)

    sc.pp.scale(adata_scrna)
    sc.pp.log1p(adata_scrna)

    print("Scaled and logged")
    # mark them 
    adata_scrna.obs['batch'] = 'scrna'
    adata_spatial.obs['batch'] = 'spatial'

    # concat
    adata = adata_scrna.concatenate(adata_spatial, batch_key='batch')
    adata.X = np.nan_to_num(adata.X, nan=0)
    X = adata.X
    # look to see how many. nan
    print("Contains NaN:", np.isnan(X).any())
    print("Contains inf:", np.isinf(X).any())
    print("Min value:", X.min())
    print("Max value:", X.max())
    sc.tl.pca(adata, n_comps= 50, svd_solver='arpack')
    print("PCA calculation")

    
    # separate them
    adata_scrna_pca = adata[adata.obs['batch'] == 'scrna'].copy()
    adata_spatial_pca = adata[adata.obs['batch'] == 'spatial'].copy()

    # run sc3 on both spatial and scrna
    # get the cluster center
    consensus(adata_scrna_pca, n_clusters = 6)
    print("Sc3 on scrna")

    # using the cluster centers find which cluster centers the cells are aligned to
    consensus(adata_spatial_pca, n_clusters = 6)
    print("Sc3 on spatial")

    # adata_spatial.obs['sc3s_4'].to_csv("sc3s_6.csv", index=False)

def get_common_indices():
    print("Getting common indices")
    # Load gene lists
    scrna_gene_list = pd.read_csv("./data/scrna_gene_list.csv", header=None)[0].tolist()
    print(f"Number of genes in scrna: {len(scrna_gene_list)}")

    spatial_gene_list = pd.read_csv("./data/spatial_gene_list.csv", header=None)[0].tolist()
    print(f"Number of genes in spatial: {len(spatial_gene_list)}")

    # Find common genes (sorted alphabetically)
    common_genes = sorted(set(scrna_gene_list).intersection(spatial_gene_list))
    print(f"Number of common genes: {len(common_genes)}")

    # Create lookup dictionaries for fast indexing
    scrna_index = {g: i for i, g in enumerate(scrna_gene_list)}
    spatial_index = {g: i for i, g in enumerate(spatial_gene_list)}

    # Get ordered indices corresponding to the alphabetically sorted common genes
    scrna_common_indices = [scrna_index[g] for g in common_genes]
    spatial_common_indices = [spatial_index[g] for g in common_genes]

    print(scrna_gene_list[scrna_common_indices[255]], spatial_gene_list[spatial_common_indices[255]])
    return (scrna_common_indices, spatial_common_indices)


if __name__ == "__main__":
    main()