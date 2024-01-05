import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import seaborn as sns

# Creating a CSR matrix from COO format
data = [3, 6, 9]  # Non-zero elements
rows = [0, 1, 2]  # Row indices
cols = [1, 2, 4]  # Column indices
H = csr_matrix((data, (rows, cols)), shape=(3, 5))

# Creating a CSR matrix from a dense array
# dense_array = np.array([[0, 2, 0, 0], [0, 0, 4, 0], [0, 0, 0, 6]])
# sparse_from_dense = csr_matrix(dense_array)

sns.heatmap(H.toarray())
plt.show()

# insf = np.array([1,2,3,4,5])

# findex = np.round(((insf - 1)/0.1)+1,0)
# print(findex)