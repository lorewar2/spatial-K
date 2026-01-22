
import scanpy as sc
import numpy as np
from _consensus import consensus
import pandas as pd
from sklearn import mixture
from sklearn.metrics.cluster import adjusted_rand_score, rand_score

np.random.seed(42)

def main():
    # Load ground truth
    ground_truth_scrna = load_scrna_ground_truth()
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
    sc3_scrna_result = adata_scrna_pca.obs['sc3s_6']
    sc3_spatial_result = adata_spatial_pca.obs['sc3s_6']
    #print(sc3_scrna_result)
    # save scrna sc3 result
    # adata_scrna_pca.obs['sc3s_6'].to_csv("./data/scrna_sc3_result.csv", index=False)

    # # save pca result for scrna
    # do gaussian mixture model clustering using different dimensions
    #X_pca = adata_scrna_pca.obsm["X_pca"]
    print("sc3 ari with ground truth", adjusted_rand_score(sc3_scrna_result, ground_truth_scrna))
    for pca_dim in range(49, 50):
        scrna_X_pca = adata_scrna_pca.obsm["X_pca"][:, :pca_dim] 
        #spatial_X_pca = adata_spatial_pca.obsm["X_pca"][:, :pca_dim] 
        gmm = mixture.GaussianMixture(
            n_components=6,
            covariance_type="spherical",
            random_state=0
        )

        # Fit on PCA space
        gmm.fit(scrna_X_pca)
        # Predict cluster labels
        labels = gmm.predict(scrna_X_pca)
        #for value in zip(labels, sc3_scrna_result):
        #    print(value)
        print("pca_dim ", pca_dim)
        print("scrna ari with sc3", adjusted_rand_score(labels, sc3_scrna_result))
        print("scrna ari with ground truth", adjusted_rand_score(labels, ground_truth_scrna))
        df = pd.DataFrame({
            "label": labels,
            "grount_truth": ground_truth_scrna
        })
        counts = pd.crosstab(df["label"], df["grount_truth"])
        print(counts)

        # spatial_X_pca = adata_spatial_pca.obsm["X_pca"][:, :pca_dim] 
        # labels = gmm.predict(spatial_X_pca)
        # print("using old cc spatial ari with sc3", adjusted_rand_score(labels, sc3_spatial_result))
        # Predict cluster labels
        # gmm.fit(spatial_X_pca)
        # labels = gmm.predict(spatial_X_pca)
        # #for value in zip(labels, sc3_scrna_result):
        # #    print(value)import pandas as pd
        # print("relearned spatial ari with sc3", adjusted_rand_score(labels, sc3_spatial_result), "\n")

        # Saving result
        
        df = pd.DataFrame(
            scrna_X_pca,
            columns=[f"PC{i+1}" for i in range(scrna_X_pca.shape[1])]
        )
        sc3_scrna_result = adata_scrna_pca.obs['sc3s_6'].to_numpy()
        # Replace these with your real arrays
        df["gau"] = labels
        df["leiden"] = ground_truth_scrna
        df["sc3"] = sc3_scrna_result

        df.to_csv("scrna_pca_with_extra_cols.csv", index=False)
    #print(labels)

    #labels = gmm.predict(spatial_X_pca)
    #for element in labels:
    #    print(element)
    # Save labels back to AnnData
    #adata_scrna_pca.obs["gmm_clusters"] = labels.astype(str)
    #print(adata_scrna_pca.obs["gmm_clusters"])
    # pca_df.to_csv("./data/pca_results_scrna.csv")
    # # save pca result for spatial
    # pca_df = pd.DataFrame(
    #     adata_spatial_pca.obsm["X_pca"],
    #     index=adata_spatial_pca.obs_names,
    #     columns=[f"PC{i+1}" for i in range(adata_spatial_pca.obsm["X_pca"].shape[1])]
    # )
    # pca_df.to_csv("./data/pca_results_spatial.csv")

def load_scrna_ground_truth():
    ground_truth_array = []
    known = []
    with open("./data/mod_anno.csv", 'r', encoding="utf-8") as file:
        for (index, line) in enumerate(file):
            if index == 0:
                continue
            line_array = line.strip().split(",")
            if line_array[-6].split("-")[0] not in known:
                known.append(line_array[-6].split("-")[0])
            ground_truth_array.append(known.index(line_array[-6].split("-")[0]))
    print(len(known))
    return ground_truth_array

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
    #load_scrna_ground_truth()
    main()