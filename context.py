import sys
import igl
import numpy as np
import scipy as sp
from utils import *
import multiprocessing as mp
import functools


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


def compute(ctx, U, i):
    # start parameter
    z = ctx.zAll[:, i]
    u = ctx.uAll[:, i]
    n = np.transpose(ctx.N[i, :])
    rho = ctx.rhoAll[i]
    # energy parameter
    hE = ctx.hEList[i]
    U_hE0 = U[hE[:, 0], :]
    U_hE1 = U[hE[:, 1], :]
    dU = np.transpose(U_hE1 - U_hE0)
    # compute Spre
    dV = ctx.dVList[i]
    WVec = ctx.WVecList[i]
    Spre = dV.dot((WVec)).dot(np.transpose(dU))
    for k in range(0, ctx.maxIterADMM):
        # R step, rho is float
        # S = Spre + rho * n.dot(np.transpose(z - u))
        S = Spre + rho * np.outer(n, z - u)
        R = fit_rotation(S)
        zOld = np.array(z)
        # z step, ctx.Lambda is float
        z = lasso_shrinkage(R.dot(n) + u, ctx.param.Lambda * ctx.VA[i] / rho)
        # u step
        u = u + R.dot(n) - z
        # residual
        r_norm = np.linalg.norm(z - R.dot(n))
        # rho is float
        s_norm = np.linalg.norm(-rho * (z - zOld))

        # rho setup
        if r_norm > ctx.param.mu * s_norm:
            rho = ctx.param.tao * rho
            u = u / ctx.param.tao
        elif s_norm > ctx.param.mu * r_norm:
            rho = rho / ctx.param.tao
            u = u * ctx.param.tao
        # stopping
        nz = float(z.shape[0])
        eps_pri = np.sqrt(2.0 * nz) * ctx.param.abstol + ctx.param.reltol * max(np.linalg.norm(R.dot(n)),
                                                                                np.linalg.norm(z))
        eps_dual = np.sqrt(nz) * ctx.param.abstol + ctx.param.reltol * np.linalg.norm(rho * u)
        if r_norm < eps_pri and s_norm < eps_dual:
            # save into ctx
            # ctx.zAll[:, i] = z
            # ctx.uAll[:, i] = u
            # ctx.rhoAll[i] = rho
            # RAll[:, :, i] = R
            # RAll[:, 3 * i:3 * i + 3] = R
            # save energy
            tmp = np.linalg.norm((R.dot(dV) - dU).dot((WVec)).dot(np.transpose(R.dot(dV) - dU)))
            energy = 0.5 * tmp * tmp + ctx.param.Lambda * ctx.VA[i] * np.sum(np.abs(R.dot(n)))
            # energyVec[i] = energy
            return z, u, rho, R, energy


def local_step_mp(V, U, RAll, ctx):
    numV = V.shape[0]
    energyVec = [0] * numV
    RList = [None] * numV
    with mp.Pool(processes=2) as pool:
        ret = pool.map(functools.partial(compute, ctx, U), range(0, numV))
    for i in range(0, numV):
        z, u, rho, R, energy = ret[i]
        energyVec[i] = energy
        RAll[:, 3 * i:3 * i + 3] = R
        ctx.zAll[:, i] = z
        ctx.uAll[:, i] = u
        ctx.rhoAll[i] = rho
    ctx.energyVec = energyVec
    ctx.energy = np.sum(ctx.energyVec)


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
            self.WVecList[i] = np.diag(self.WVecList[i])
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
        # local_step_mp(V, U, RAll, self)
        # global step
        Upre = np.array(U)
        self.global_step(V, U, RAll)
        # update relative deltaV
        self.relativeDeltaV = np.max(np.abs(U - Upre)) / np.max(np.abs(U - V))

    def global_step(self, V, U, RAll):
        numV = V.shape[0]
        Rcol = columnize(RAll, numV)
        Bcol = self.K @ Rcol
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
        energyVec = [0] * numV
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
            # Spre = dV.dot(np.diag(WVec)).dot(np.transpose(dU))
            Spre = dV @ (WVec) @ (np.transpose(dU))
            zOld = np.array(z)
            for k in range(0, self.maxIterADMM):
                # R step, rho is float
                # S = Spre + rho * n.dot(np.transpose(z - u))
                S = Spre + rho * np.outer(n, z - u)
                R = fit_rotation(S)
                # zOld = np.array(z)
                zOld = z + 0
                # z step, self.Lambda is float
                Rn = R @ n
                z = lasso_shrinkage(Rn + u, self.param.Lambda * self.VA[i] / rho)
                # u step
                u = u + Rn - z
                # residual
                r_norm = np.linalg.norm(z - Rn)
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
                eps_pri = np.sqrt(2.0 * nz) * self.param.abstol + self.param.reltol * max(np.linalg.norm(Rn),
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
                    RdVminusDU = R @ dV - dU
                    tmp = np.linalg.norm((RdVminusDU) @ (WVec) @ (np.transpose(RdVminusDU)))
                    energy = 0.5 * tmp * tmp + self.param.Lambda * self.VA[i] * np.sum(np.abs(Rn))
                    energyVec[i] = energy
                    break
        self.energyVec = energyVec
        self.energy = np.sum(self.energyVec)

    def local_step_mp(self, V, U, RAll):
        numV = V.shape[0]
        energyVec = [0] * numV

        RList = [None] * numV

        def compute(i):
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
            Spre = dV.dot((WVec)).dot(np.transpose(dU))
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
                    # self.zAll[:, i] = z
                    # self.uAll[:, i] = u
                    # self.rhoAll[i] = rho
                    # RAll[:, :, i] = R
                    # RAll[:, 3 * i:3 * i + 3] = R
                    # save energy
                    tmp = np.linalg.norm((R.dot(dV) - dU).dot((WVec)).dot(np.transpose(R.dot(dV) - dU)))
                    energy = 0.5 * tmp * tmp + self.param.Lambda * self.VA[i] * np.sum(np.abs(R.dot(n)))
                    # energyVec[i] = energy
                    return z, u, rho, R, energy

        with mp.Pool(processes=4) as pool:
            ret = pool.map(compute, range(0, numV))
        for i in range(0, numV):
            z, u, rho, R, energy = ret[i]
            energyVec[i] = energy
            RAll[:, 3 * i:3 * i + 3] = R
            self.zAll[:, i] = z
            self.uAll[:, i] = u
            self.rhoAll[i] = rho
        self.energyVec = energyVec
        self.energy = np.sum(self.energyVec)

    def set_constrain(self, V, F):
        self.bc = V[F[0, 0]]
        self.b = F[0, 0]
