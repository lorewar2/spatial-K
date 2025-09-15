

import numpy as np

data = np.load("NatGen2022_scRNAseq_sparse_matrix_file.npz", allow_pickle=True)
print(data.files) 
arr = data['arr_0'] 
print(arr.dtype, arr.shape, type(arr))
sparse_matrix = arr.item()
n_rows, n_cols = sparse_matrix.shape

with open("dense.csv", "w") as f:
    for i in range(n_rows):
        row = sparse_matrix.getrow(i).toarray().ravel()
        np.savetxt(f, [row], delimiter=",", fmt="%.6f")

print("\nDone! Dense matrix saved to dense.csv")