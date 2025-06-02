# mytranspose.py
import numpy as np
import pandas as pd
import torch

def mytranspose(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x.reshape(-1, 1)  # 1D vector를 column vector로 처리
        elif x.size == 0:
            return x
        else:
            y = np.empty((x.shape[1], x.shape[0]), dtype=x.dtype)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    y[j, i] = x[i, j]
            return y

    elif isinstance(x, list):
        if not x:
            return []
        return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]

    elif isinstance(x, pd.DataFrame):
        return x.transpose()

    elif isinstance(x, torch.Tensor):
        return x.t() if x.ndim == 2 else x

    else:
        raise TypeError("지원되지 않는 타입입니다.")