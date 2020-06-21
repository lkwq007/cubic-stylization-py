# REF: https://github.com/HTDerekLiu/CubicStylization_Cpp
import os
import sys
import argparse

import igl
import numpy as np
import scipy as sp

from utils import *
from context import *

import timeit

# captialized
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description="cubic stylization")
    parser.add_argument("mesh", help="path to target mesh")
    parser.add_argument("--plambda", help="parameter to control cubeness", type=float, default=0.2)
    opt = parser.parse_args()
    print(opt)
    filename = opt.mesh
    # load mesh
    V, _, _, F, _, _ = igl.read_obj(filename)
    # init optim context
    ctx = Context()
    # normalize mesh
    V = normalize(V)
    U = np.array(V)
    # set a constrained point
    ctx.set_constrain(V, F)
    ctx.set_parameters(V, F)
    ctx.set_lambda(opt.plambda)
    # optim loop
    maxIter = 1000
    stopRelativeDeltaV = 1e-3
    start = timeit.default_timer()
    for iter in range(0, maxIter):
        ctx.optim_step(V, U)
        print("Iter at", iter, "Energy", ctx.energy, "RelativeDeltaV", ctx.relativeDeltaV)
        if ctx.relativeDeltaV < stopRelativeDeltaV:
            break
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    output_name = os.path.splitext(filename)
    tmp="_cubic_{:.4}".format(opt.plambda)
    output_filename = tmp.join(output_name)
    normalized_filename = "_norm".join(output_name)
    igl.write_obj(normalized_filename, V, F)
    igl.write_obj(output_filename, U, F)
