# test_all_mytranspose.py
import numpy as np
import pandas as pd
import torch
from mytranspose import mytranspose

# === (3) DataFrame ===
print("\n=== (3) DataFrame Test ===")
d = np.array([1, 2, 3, 4])
e = np.array(["red", "white", "red", np.nan])
f = np.array([True, True, True, False])
mydata3 = pd.DataFrame({"d": d, "e": e, "f": f})
print("Original:\n", mydata3)
print("Transposed:\n", mytranspose(mydata3))

# === (4) PyTorch Tensor ===
print("\n=== (4) PyTorch Tensor Test ===")
tensor_pt = torch.tensor([[1, 2], [3, 4]])
print("Original:\n", tensor_pt)
print("Transposed:\n", mytranspose(tensor_pt))
