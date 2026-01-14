
import scanpy as sc
from _consensus import consensus
import pandas as pd

def main():
    # Load dataset
    adata_scrna = sc.read_csv("./data/scrna_filtered.csv")
    adata_spatial = sc.read_csv("./data/spatial_data.csv")
    print("Data Loaded!")

    # spatial data set cells in coloumns so transform
    adata_spatial = adata_spatial.T

    # we have to filter and match the genes on both datasets before concatation
    (scrna_indices, spatial_indices) = get_common_indices()

    # select and rearrange the matrices for common genes
    adata_scrna = adata_scrna[:, scrna_indices]
    adata_spatial = adata_spatial[:, spatial_indices]

    # preprocess log2(x+1) and scale
    sc.pp.log1p(adata_spatial)
    sc.pp.scale(adata_spatial)
    sc.pp.log1p(adata_scrna)
    sc.pp.scale(adata_scrna)
    print("Scaled and logged")

    # mark them 
    adata_scrna.obs['batch'] = 'scrna'
    adata_spatial.obs['batch'] = 'spatial'

    # concat
    adata = adata_scrna.concatenate(adata_spatial, batch_key='batch')

    # do pca for all 
    sc.tl.pca(adata, n_comps= 50, svd_solver='arpack')
    print("PCA calculation done")

    # separate them
    adata_scrna_pca = adata[adata.obs['batch'] == '0'].copy()
    adata_spatial_pca = adata[adata.obs['batch'] == '1'].copy()

    # verify
    print(adata_scrna_pca)
    print(adata_spatial_pca)
    
    # run sc3 on both spatial and scrna
    consensus(adata_scrna_pca, n_clusters = 6)
    consensus(adata_spatial_pca, n_clusters = 6)
    print("Sc3 on scrna and spatial")

    # save scrna sc3 result
    adata_scrna_pca.obs['sc3s_6'].to_csv("./data/scrna_sc3_result.csv", index=False)

    # save pca result for scrna
    pca_df = pd.DataFrame(
        adata_scrna_pca.obsm["X_pca"],
        index=adata_scrna_pca.obs_names,
        columns=[f"PC{i+1}" for i in range(adata_scrna_pca.obsm["X_pca"].shape[1])]
    )
    pca_df.to_csv("./data/pca_results_scrna.csv")
    # save pca result for spatial
    pca_df = pd.DataFrame(
        adata_spatial_pca.obsm["X_pca"],
        index=adata_spatial_pca.obs_names,
        columns=[f"PC{i+1}" for i in range(adata_spatial_pca.obsm["X_pca"].shape[1])]
    )
    pca_df.to_csv("./data/pca_results_spatial.csv")

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