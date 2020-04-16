import sys
import igl
import numpy as np
import scipy as sp
from utils import *


class Parameter:
    def __init__(self, plambda=0.2):
        self.Lambda = plambda
        self.rhoInit = 1e-3
        self.abstol = 1e-5
        self.reltol = 1e-3
        self.mu = 10.0
        self.tao = 2.0

    def reset(self):
        self.rhoInit = 1e-3
        self.abstol = 1e-5
        self.reltol = 1e-3
        self.mu = 10.0
        self.tao = 2.0


class Context:
    def __init__(self):
        self.param = Parameter()
        self.maxIterADMM = 100.0
        self.energy = 0.0
        self.relativeDeltaV = sys.float_info.max
        self.xPlane = 0.0
        self.yPlane = 0.0
        self.zPlane = 0.0

    def reset(self):
        self.param.reset()
        self.maxIterADMM = 100
        self.energy = 0.0
        self.relativeDeltaV = sys.float_info.max
        self.objHis = []
        self.UHist = []

    def set_lambda(self, val):
        self.param.Lambda = val

    def set_parameters(self, V, F):
        # reset
        self.reset()
        # compute property given V and F
        self.N = igl.per_vertex_normals(V, F)
        self.L = igl.cotmatrix(V, F)
        VA = igl.massmatrix(V, F, 0)
        self.VA = VA.diagonal()
        # get face adjacency list
        VF, NI = igl.vertex_triangle_adjacency(F, V.shape[0])
        adjFList = construct_adjacency_list(VF, NI)
        # arap
        self.K = igl.arap_rhs(V, F, d=3, energy=1)
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
            V_hE0 = V[self.hEList[i][:, 0], :]
            V_hE1 = V[self.hEList[i][:, 1], :]
            self.dVList[i] = np.transpose(V_hE1 - V_hE0)
        # other var
        numV = V.shape[0]
        self.zAll = np.random.rand(3, numV) * 2.0 - 1.0
        self.uAll = np.random.rand(3, numV) * 2.0 - 1.0
        self.zAll = np.zeros((3, numV))
        self.uAll = np.zeros((3, numV))
        self.rhoAll = np.full(numV, self.param.rhoInit)

    def optim_step(self, V, U):
        # init all rotation martrix
        RAll = np.zeros((3, 3 * V.shape[0]))
        # local step
        self.local_step(V, U, RAll)
        # global step
        Upre = np.array(U)
        self.global_step(V, U, RAll)
        # update relative deltaV
        self.relativeDeltaV = np.max(np.abs(U - Upre)) / np.max(np.abs(U - V))

    def global_step(self, V, U, RAll):
        numV = V.shape[0]
        Rcol = columnize(RAll, numV)
        Bcol = self.K.dot(Rcol)
        # ret = igl.min_quad_with_fixed(self.L, B, self.b, self.bc, Aeq, Beq, False)
        for dim in range(0, V.shape[1]):
            b = np.array([self.b])
            bc = np.array([self.bc[dim]])
            Aeq = sp.sparse.csc_matrix((0, 0))
            Beq = np.array([])
            B = Bcol[dim * numV:dim * numV + numV]
            ok, Uc = igl.min_quad_with_fixed(self.L, B, b, bc, Aeq, Beq, False)
            U[:, dim] = Uc

    def local_step(self, V, U, RAll):
        numV = V.shape[0]
        self.energyVec = np.zeros(numV)
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
            for k in range(0, self.maxIterADMM):
                # R step, rho is float
                # S = Spre + rho * n.dot(np.transpose(z - u))
                S = Spre + rho * np.outer(n, z - u)
                R = fit_rotation(S)
                zOld = np.array(z)
                # z step, self.Lambda is float
                z = lasso_shrinkage(R.dot(n) + u, self.param.Lambda * self.VA[i] / rho)
                # u step
                u = u + R.dot(n) - z
                # residual
                r_norm = np.linalg.norm(z - R.dot(n))
                # rho is float
                s_norm = np.linalg.norm(-rho * (z - zOld))

                # rho setup
                if r_norm > self.param.mu * s_norm:
                    rho = self.param.tao * rho
                    u = u / self.param.tao
                elif s_norm > self.param.mu * r_norm:
                    rho = rho / self.param.tao
                    u = u * self.param.tao
                # stopping
                nz = float(z.shape[0])
                eps_pri = np.sqrt(2.0 * nz) * self.param.abstol + self.param.reltol * max(np.linalg.norm(R.dot(n)),
                                                                                          np.linalg.norm(z))
                eps_dual = np.sqrt(nz) * self.param.abstol + self.param.reltol * np.linalg.norm(rho * u)
                if r_norm < eps_pri and s_norm < eps_dual:
                    # save into ctx
                    self.zAll[:, i] = z
                    self.uAll[:, i] = u
                    self.rhoAll[i] = rho
                    # RAll[:, :, i] = R
                    RAll[:, 3 * i:3 * i + 3] = R
                    # save energy
                    tmp = np.linalg.norm((R.dot(dV) - dU).dot(np.diag(WVec)).dot(np.transpose(R.dot(dV) - dU)))
                    energy = 0.5 * tmp * tmp + self.param.Lambda * self.VA[i] * np.sum(np.abs(R.dot(n)))
                    self.energyVec[i] = energy
                    break
        self.energy = np.sum(self.energyVec)

    def set_constrain(self, V, F):
        self.bc = V[F[0, 0]]
        self.b = F[0, 0]
