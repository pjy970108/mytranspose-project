# test_all_mytranspose.py
import numpy as np
import pandas as pd
import torch
from mytranspose import mytranspose

# === (1) Matrix ===
print("=== (1) Matrix Tests ===")
cases_matrix = [
    np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),  # 5x2
    np.empty((0, 0)),                                    # 빈 행렬
    np.array([[1, 2]]),                                  # 1x2
    np.array([[1], [2]]),                                # 2x1
]
for i, case in enumerate(cases_matrix, 1):
    print(f"\nMatrix Case {i}:\n{case}")
    print("Transposed:\n", mytranspose(case))

# === (2) Vector ===
print("\n=== (2) Vector Tests ===")
cases_vector = [
    np.array([1, 2, np.nan, 3]),  # NaN 포함
    np.array([np.nan]),
    np.array([]),
]
for i, case in enumerate(cases_vector, 1):
    print(f"\nVector Case {i}:\n{case}")
    print("Transposed:\n", mytranspose(case))

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
