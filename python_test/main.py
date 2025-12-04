
import scanpy as sc
from _consensus import consensus
import pandas as pd

def main():
    # Load dataset
    adata_scrna = sc.read_csv("./data/scrna_filtered.csv")
    adata_spatial = sc.read_csv("./data/spatial_data.csv")

    # save origin
    adata_scrna.obs['batch'] = 'scrna'
    adata_spatial.obs['batch'] = 'spatial'
    # spatial data set cells in coloumns so transform
    adata_spatial = adata_spatial.T
    # we have to filter and match the genes on both datasets before concatation
    # load gene lists
    gene_list = pd.read_csv("./data/common_genes.csv", header=None)
    gene_list = gene_list[0].tolist()  # convert to Python list
    print(f"Number of genes in list: {len(gene_list)}")

    # concat
    adata = adata_scrna.concatenate(adata_spatial, batch_key='batch')



    # Log transform the data (equivalent to log2(x+1))
    sc.pp.scale(adata_spatial)
    print("Scaling Done")
    sc.pp.log1p(adata_spatial)
    print("Log conversion Done")
    # Perform PCA for dimensionality reduction
    sc.tl.pca(adata_spatial, n_comps= 50, svd_solver='arpack')
    print("PCA Done")
    # Apply SC3 consensus clustering with n_clusters=6 (matching the ks=6)
    consensus(adata_spatial, n_clusters=4)

    #print(adata_spatial.obs['sc3s_6'])
    adata_spatial.obs['sc3s_4'].to_csv("sc3s_6.csv", index=False)

def test():
    print("Test")
    scrna_gene_list = pd.read_csv("./data/scrna_gene_list.csv", header=None)
    scrna_gene_list = scrna_gene_list[0].tolist()
    print(f"Number of genes in scrna: {len(scrna_gene_list)}")
    spatial_gene_list = pd.read_csv("./data/spatial_gene_list.csv", header=None)
    spatial_gene_list = spatial_gene_list[0].tolist()
    print(f"Number of genes in spatial: {len(spatial_gene_list)}")
    common_genes = set(scrna_gene_list).intersection(set(spatial_gene_list))
    common_genes = sorted(list(common_genes))
    print(f"Number of common genes: {len(common_genes)}")

if __name__ == "__main__":
    test()