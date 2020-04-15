import sys
import argparse
from typing import List

import igl
import numpy as np
import scipy as sp

debug = {"flag": False}


def wrapper(*args, **kwargs):
    if debug["flag"]:
        print(*args, **kwargs)


print_debug = wrapper


def shrinkage_step(x: np.ndarray, k: float):
    # Regression shrinkage and selection via the lasso
    return np.maximum(x - k, 0.0) - np.maximum(-x - k, 0.0)


def construct_adjacency_list(VF: np.ndarray, NI: np.ndarray) -> List[np.ndarray]:
    # dont know why there is a preceeding zero
    adjFList = []
    # we have VF(NI(i)+j) = f
    adjLen = NI.shape[0] - 1
    for idx in range(0, adjLen):
        start = NI[idx]
        end = NI[idx + 1]
        faces = VF[start:end]
        adjFList.append(faces)
    return adjFList


def normalize(V: np.ndarray) -> np.ndarray:
    min_val = np.amin(V, axis=0)
    V = V - min_val
    V = V / np.max(V)
    mean = np.mean(V, axis=0)
    V = V - mean
    return V


def orthogonal_procrustes(S: np.ndarray):
    SU, SS, SVH = np.linalg.svd(S, full_matrices=True)
    SVH = np.transpose(SVH)
    R = SVH.dot(np.transpose(SU))
    if np.linalg.det(R) < 0:
        SU[:, 2] = -SU[:, 2]
        R = SVH.dot(np.transpose(SU))
    return R


class Parameter:
    def __init__(self, plambda=0.2):
        self.Lambda = plambda
        self.rhoInit = 1e-3
        self.ABSTOL = 1e-5
        self.RELTOL = 1e-3
        self.mu = 10.0
        self.tao = 2.0

    def reset(self):
        self.rhoInit = 1e-3
        self.ABSTOL = 1e-5
        self.RELTOL = 1e-3
        self.mu = 10.0
        self.tao = 2.0


def columnize(A, k):
    m = A.shape[0]
    n = A.shape[1] // k
    result = np.zeros((A.shape[0] * A.shape[1], 1))
    for b in range(0, k):
        for i in range(0, m):
            for j in range(0, n):
                result[j * m * k + i * k + b] = A[i, b * n + j]
    return result


class Context:
    def __init__(self):
        self.param = Parameter()
        # self.Lambda = 0.2
        # self.rhoInit = 1e-3
        # self.ABSTOL = 1e-6
        # self.RELTOL = 1e-3
        # self.mu = 10.0
        # self.tao = 2.0
        self.maxIter_ADMM = 100.0
        self.objVal = 0.0
        self.reldV = sys.float_info.max
        self.xPlane = 0.0
        self.yPlane = 0.0
        self.zPlane = 0.0

    def reset(self):
        # self.rhoInit = 1e-3
        # self.ABSTOL = 1e-5
        # self.RELTOL = 1e-3
        # self.mu = 10.0
        # self.tao = 2.0
        self.param.reset()
        self.maxIter_ADMM = 100
        self.objVal = 0.0
        self.reldV = sys.float_info.max
        self.objHis = []
        self.UHist = []

    def set_parameters(self, V, F):
        # reset
        self.reset()
        # compute property given V and F
        self.N = igl.per_vertex_normals(V, F)
        self.L = igl.cotmatrix(V, F)
        print_debug("N", self.N[0:5])
        print_debug("L", self.L[0:5, 0:5])
        VA = igl.massmatrix(V, F, 0)
        print_debug("VA", VA[0:5, 0:5])
        self.VA = VA.diagonal()
        # get face adjacency list
        VF, NI = igl.vertex_triangle_adjacency(F, V.shape[0])
        adjFList = construct_adjacency_list(VF, NI)
        # arap
        self.K = igl.arap_rhs(V, F, d=3, energy=1)
        print_debug("K", self.K[0:5, 0:5])
        # they are all list since length can be different
        self.hEList = [None] * V.shape[0]
        self.WVecList = [None] * V.shape[0]
        self.dVList = [None] * V.shape[0]
        for i in range(0, V.shape[0]):
            adjF = adjFList[i]
            len_adjF = adjF.shape[0]
            self.hEList[i] = np.zeros((len_adjF * 3, 2), dtype=int)
            self.WVecList[i] = np.zeros(len_adjF * 3)
            self.dVList[i] = np.zeros((3, 3 * len_adjF))
            for j in range(0, len_adjF):
                vIdx = adjF[j]
                v0 = F[vIdx, 0]
                v1 = F[vIdx, 1]
                v2 = F[vIdx, 2]
                # half edge indices
                # hE = np.array([[v0, v1], [v1, v2], [v2, v0]])
                # self.hEList[i] = hE
                self.hEList[i][3 * j, 0] = v0
                self.hEList[i][3 * j, 1] = v1
                self.hEList[i][3 * j + 1, 0] = v1
                self.hEList[i][3 * j + 1, 1] = v2
                self.hEList[i][3 * j + 2, 0] = v2
                self.hEList[i][3 * j + 2, 1] = v0
                # weight vec
                self.WVecList[i][3 * j] = self.L[v0, v1]
                self.WVecList[i][3 * j + 1] = self.L[v1, v2]
                self.WVecList[i][3 * j + 2] = self.L[v2, v0]
                if j == 0 and i == 0:
                    print_debug("he", self.hEList[i][3 * j:3 * j + 3, :])
                    print_debug("wv", self.WVecList[i][3 * j:3 * j + 3])
            V_hE0 = V[self.hEList[i][:, 0], :]
            V_hE1 = V[self.hEList[i][:, 1], :]
            self.dVList[i] = np.transpose(V_hE1 - V_hE0)
            if i == 0:
                print_debug(self.dVList[i])
        # precompute the slover
        # missing precomputation
        # other var
        numV = V.shape[0]
        self.zAll = np.random.rand(3, numV) * 2.0 - 1.0
        self.uAll = np.random.rand(3, numV) * 2.0 - 1.0
        self.zAll = np.zeros((3, numV))
        self.uAll = np.zeros((3, numV))
        print_debug(self.zAll[0:3, 0:3], "\n", self.uAll[0:3, 0:3])
        self.rhoAll = np.full(numV, self.param.rhoInit)

    def optim_step(self, V, U):
        # local step
        # RAll = np.zeros((3, 3, V.shape[0]))
        RAll = np.zeros((3, 3 * V.shape[0]))
        self.local_step(V, U, RAll)
        # global step
        print_debug(RAll[:, :3])
        Upre = np.array(U)
        self.global_step(V, U, RAll)
        # update relative deltaV
        self.reldV = np.max(np.abs(U - Upre)) / np.max(np.abs(U - V))
        # exit(0)

    def global_step(self, V, U, RAll):
        numV = V.shape[0]
        # Rcol = np.transpose(RAll, (2, 0, 1)).reshape(numV * 9, 1)
        Rcol = columnize(RAll, numV)
        print_debug("R", Rcol[:5])
        print_debug("K", self.K[0:5, :5])
        Bcol = self.K.dot(Rcol)
        print_debug("B", Bcol[:5])
        # ret = igl.min_quad_with_fixed(self.L, B, self.b, self.bc, Aeq, Beq, False)
        for dim in range(0, V.shape[1]):
            b = np.array([self.b])
            bc = np.array([self.bc[dim]])
            Aeq = sp.sparse.csc_matrix((0, 0))
            Beq = np.array([])
            B = Bcol[dim * numV:dim * numV + numV]
            ok, Uc = igl.min_quad_with_fixed(self.L, B, b, bc, Aeq, Beq, False)
            U[:, dim] = Uc
            print_debug("Uc", Uc[0:10], b, bc, Uc[self.b])

    def local_step(self, V, U, RAll):
        numV = V.shape[0]
        self.objValVec = np.zeros(numV)
        for i in range(0, numV):
            # start parameter
            z = self.zAll[:, i]
            u = self.uAll[:, i]
            n = np.transpose(self.N[i, :])
            rho = self.rhoAll[i]
            # energy parameter
            hE = self.hEList[i]
            U_hE0 = U[hE[:, 0], :]
            U_hE1 = U[hE[:, 1], :]
            dU = np.transpose(U_hE1 - U_hE0)

            # compute Spre
            dV = self.dVList[i]
            WVec = self.WVecList[i]
            Spre = dV.dot(np.diag(WVec)).dot(np.transpose(dU))
            if i == 0:
                print_debug("dU", dU[:3, :5])
                print_debug("dV", dV[:3, :3])
                print_debug("WVec", WVec[:3])
                print_debug("Spre", Spre[:3, :3])
            for k in range(0, self.maxIter_ADMM):
                # R step, rho is float
                # S = Spre + rho * n.dot(np.transpose(z - u))
                S = Spre + rho * np.outer(n, z - u)

                if i == 0 and k == 1:
                    print("add", rho * np.outer(n, z - u))
                R = orthogonal_procrustes(S)
                zOld = np.array(z)
                # z step, self.Lambda is float
                z = shrinkage_step(R.dot(n) + u, self.param.Lambda * self.VA[i] / rho)
                # u step
                u = u + R.dot(n) - z
                # residual
                r_norm = np.linalg.norm(z - R.dot(n))
                # rho is float
                s_norm = np.linalg.norm(-rho * (z - zOld))
                if i == 0:
                    print_debug(k)
                    print_debug("S", S[:3, :3])
                    print_debug("R", R[:3, :3])
                    print_debug("zOld", zOld)
                    print_debug("z", z)
                    print_debug("u", u)
                    print_debug("r s norm", r_norm, s_norm)

                # rho setup
                if r_norm > self.param.mu * s_norm:
                    rho = self.param.tao * rho
                    u = u / self.param.tao
                elif s_norm > self.param.mu * r_norm:
                    rho = rho / self.param.tao
                    u = u * self.param.tao
                # stopping
                nz = float(z.shape[0])
                eps_pri = np.sqrt(2.0 * nz) * self.param.ABSTOL + self.param.RELTOL * max(np.linalg.norm(R.dot(n)),
                                                                                          np.linalg.norm(z))
                eps_dual = np.sqrt(nz) * self.param.ABSTOL + self.param.RELTOL * np.linalg.norm(rho * u)
                if k == 0 and i == 0:
                    print_debug("newu", u)
                    print_debug("rho", rho, "nz", nz, "pri", eps_pri, "dual", eps_dual)
                if r_norm < eps_pri and s_norm < eps_dual:
                    # save into ctx
                    self.zAll[:, i] = z
                    self.uAll[:, i] = u
                    self.rhoAll[i] = rho
                    # RAll[:, :, i] = R
                    RAll[:, 3 * i:3 * i + 3] = R
                    # save objVal
                    tmp = np.linalg.norm((R.dot(dV) - dU).dot(np.diag(WVec)).dot(np.transpose(R.dot(dV) - dU)))
                    objVal = 0.5 * tmp * tmp + self.param.Lambda * self.VA[i] * np.sum(np.abs(R.dot(n)))
                    self.objValVec[i] = objVal
                    if i == 0:
                        print_debug(k, rho, z, u)
                        print_debug(objVal)
                    break
        self.objVal = np.sum(self.objValVec)

    def set_constrain(self, V, F):
        self.bc = V[F[0, 0]]
        self.b = F[0, 0]
        print_debug(self.b, self.bc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cubic stylization")
    parser.add_argument("mesh", help="path to target mesh")
    opt = parser.parse_args()
    print_debug(opt)
    filename = "./meshes/kleinBottle.obj"
    parm_lambda = 0.2
    # load mesh
    vertex, _, _, face, _, _ = igl.read_obj(filename)
    V = vertex
    F = face
    print_debug(V.shape, F.shape)
    # init optim context
    ctx = Context()
    ctx.Lambda = 0.2
    # normalize mesh
    V = normalize(V)
    print_debug(V[0:5])
    U = np.array(V)
    # set a constrained point
    ctx.set_constrain(V, F)
    ctx.set_parameters(V, F)
    maxIter = 1000
    stopReldV = 1e-3
    for iter in range(0, maxIter):
        print("Iter at", iter)
        ctx.optim_step(V, U)
        print("Obj", ctx.objVal, "ReldV", ctx.reldV)
        if ctx.reldV < stopReldV:
            break
    igl.write_obj("input.obj", V, F)
    igl.write_obj("ouput.obj", U, F)
    exit(0)
