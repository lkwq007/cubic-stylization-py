from typing import List

import numpy as np
import scipy as sp
from numba import jit,njit

def normalize(V: np.ndarray) -> np.ndarray:
    # normalize vertice
    min_val = np.amin(V, axis=0)
    V = V - min_val
    V = V / np.max(V)
    mean = np.mean(V, axis=0)
    V = V - mean
    return V


def construct_adjacency_list(VF: np.ndarray, NI: np.ndarray) -> List[np.ndarray]:
    # Ni are offsets
    adjacencyFList = []
    # we have VF(NI(i)+j) = f
    total = NI.shape[0] - 1
    for idx in range(0, total):
        start = NI[idx]
        end = NI[idx + 1]
        faces = VF[start:end]
        adjacencyFList.append(faces)
    return adjacencyFList

@njit
def lasso_shrinkage(x: np.ndarray, k: float) -> np.ndarray:
    # Regression shrinkage and selection via the lasso
    return np.maximum(x - k, 0.0) - np.maximum(-x - k, 0.0)

@njit
def fit_rotation(S: np.ndarray):
    # orthogonal Procrustes
    SU, SS, SVH = np.linalg.svd(S, full_matrices=True)
    SVH = np.transpose(SVH)
    R = SVH.dot(np.transpose(SU))
    if np.linalg.det(R) < 0:
        SU[:, 2] = -SU[:, 2]
        R = SVH.dot(np.transpose(SU))
    return R


def columnize(A: np.ndarray, k: np.ndarray) -> np.ndarray:
    # columinize stacked matrix
    m = A.shape[0]
    n = A.shape[1] // k
    result = np.zeros((A.shape[0] * A.shape[1], 1))
    for b in range(0, k):
        for i in range(0, m):
            for j in range(0, n):
                result[j * m * k + i * k + b] = A[i, b * n + j]
    return result
