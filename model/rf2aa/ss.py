import numpy as np

import torch

def get_opt_path(matrix):
    B, L, L = matrix.shape

    path = torch.zeros_like(matrix, dtype=torch.int32)

    for b in range(B):
        for i in range(L):
            path[b, i] = torch.arange(L, device=matrix.device)  # 保证与matrix同设备

    for b in range(B):
        for m in range(L):
            for i in range(L):
                for j in range(L):
                    if matrix[b, i, m] + matrix[b, m, j] < matrix[b, i, j]:
                        matrix[b, i, j] = matrix[b, i, m] + matrix[b, m, j]
                        path[b, i, j] = path[b, i, m]

    return matrix


def calc_offset_matrix(res_dist, c1, c2):
    if len(c1) != len(c2):
        return []
    
    n_aa = res_dist.shape[1]
    res_dist.fill_(n_aa)

    for i in range(n_aa):
       res_dist[0][i][i] = 0

    # linear peptide connection
    for i in range(n_aa - 1):
        res_dist[0][i][i + 1] = 1
        res_dist[0][i + 1][i] = 1

    # nc connection
    res_dist[0][0][n_aa - 1] = 1
    res_dist[0][n_aa - 1][0] = 1

    # ss connection
    for i in range(len(c1)):
        res_dist[0][c1[i]][c2[i]] = 1
        res_dist[0][c2[i]][c1[i]] = 1

    # # get the shortest path
    res_dist = get_opt_path(res_dist)

    # for i in range(res_dist.shape[1]):
    #     for j in range(i+1, res_dist.shape[1], 1):
    #         res_dist[0][i][j] *= -1

    return res_dist
