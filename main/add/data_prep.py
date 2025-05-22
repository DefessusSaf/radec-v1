import numpy as np 

input_file = "TMP/k1-imp.fts.sx"
output_file = "k1-imp-data.txt"

data = np.loadtxt(input_file)

data = np.delete(data, 5, axis=1)

new_column = np.zeros((data.shape[0], 1))
data = np.hstack((data, new_column))

np.savetxt(output_file, data, fmt="%10.4f", delimiter="    ")