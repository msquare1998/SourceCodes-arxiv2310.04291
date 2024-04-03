# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ VQITE for the triangular model with different initial params
#                   100 samples have been tested with seed 7
#                       and 100 of them succeeded
#                   see more details in 'theShots.log'
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 26, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import numpy as np
import tensorcircuit as tc
from time import time
import model
import argparse

np.random.seed(7)
tc.set_backend("jax")
tc.set_dtype("complex128")

parser = argparse.ArgumentParser()
parser.add_argument("--Jx", type=float, default="0.9")
parser.add_argument("--Ja", type=float, default="1")
parser.add_argument("--Lx", type=int, default="4")
parser.add_argument("--Ly", type=int, default="4")
parser.add_argument("--dTau", type=float, default="0.05")
parser.add_argument("--maxItr", type=int, default="80")
parser.add_argument("--nShots", type=int, default="5")
args = parser.parse_args()

Jx, Ja = args.Jx, args.Ja
Lx, Ly = args.Lx, args.Ly
nQ = int(Lx * Ly)

dTau, maxItr = args.dTau, args.maxItr
stdDev = 0.05
nShots = args.nShots

l, p = 1, 9

reEvolve, plot = True, True
doCounting = True

def varWaveFunc(theta, psi0):
    theta = tc.backend.reshape(theta, [l, nQ, p])
    circ = tc.Circuit(nQ, inputs=psi0)

    for i in range(l):
        for j in range(nQ):
            circ.rz(j, theta=theta[i, j, 0])
            circ.rx(j, theta=theta[i, j, 1])
            circ.rz(j, theta=theta[i, j, 2])

        for pair in linkNNX:
            q0, q1 = int(pair[0]), int(pair[1])
            circ.exp1(q0, q1, theta=theta[i, q0, 3], unitary=tc.gates._yz_matrix)
            circ.exp1(q0, q1, theta=theta[i, q0, 4], unitary=tc.gates._zy_matrix)

        traversed = np.zeros(nQ)
        for pair in linkNNA:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q1] == 0:
                circ.exp1(q0, q1, theta=theta[i, q0, 5], unitary=tc.gates._yz_matrix)
                circ.exp1(q0, q1, theta=theta[i, q0, 6], unitary=tc.gates._zy_matrix)
                traversed[q1] = 1
            else:
                circ.exp1(q0, q1, theta=theta[i, q0, 7], unitary=tc.gates._yz_matrix)
                circ.exp1(q0, q1, theta=theta[i, q0, 8], unitary=tc.gates._zy_matrix)

    return circ.state()

# ∂ψ/∂θ
pPsi_pTheta = tc.backend.jit(tc.backend.jacfwd(varWaveFunc, argnums=0))
def innerProduct_ij(i, j):
    return tc.backend.tensordot(tc.backend.conj(i), j, 1)

@tc.backend.jit
def lhs_matrix(theta, psi0):
    vij = tc.backend.vmap(innerProduct_ij, vectorized_argnums=0)    # Vectorized operation for calculating a batch of vij
    vvij = tc.backend.vmap(vij, vectorized_argnums=1)
    jacobian = pPsi_pTheta(theta, psi0=psi0)
    jacobian = tc.backend.transpose(jacobian)
    fim = vvij(jacobian, jacobian)
    fim = tc.backend.real(fim)
    return fim

@tc.backend.jit
def rhs_vector(theta, psi0):
    def energy(theta, psi0):
        w = varWaveFunc(theta, psi0)
        wl = tc.backend.stop_gradient(w)
        wl = tc.backend.conj(wl)
        wr = w
        wl = tc.backend.reshape(wl, [1, -1])
        wr = tc.backend.reshape(wr, [-1, 1])
        e = wl @ h @ wr
        return tc.backend.real(e)[0, 0]

    eg = tc.backend.grad(energy, argnums=0)
    rhs = eg(theta, psi0)
    rhs = -tc.backend.real(rhs)
    return rhs

@tc.backend.jit
def update(theta, lhs, rhs, tau):
    eps = 1e-4
    lhs += eps * tc.backend.eye(l * nQ * p, dtype=lhs.dtype)
    return (theta + tc.backend.cast(tau * tc.backend.solve(lhs, rhs, assume_a="sym"), dtype=theta.dtype))

@tc.backend.jit
def calcEnergy(vec):
    return (tc.backend.conj(tc.backend.transpose(vec)) @ h @ vec).real


if __name__ == "__main__":
    time0 = time()
    h, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)
    linkNNX = model.makeLinkNNX_PBC(Lx, Ly)
    linkNNA = model.makeLinkNNA_PBC(Lx, Ly)

    totalSteps = 0
    success = 0

    for y in range(nShots):
        print(f"The {y}st shot:", end="\t")

        theta = np.random.normal(0, stdDev, size=[l * nQ * p])
        theta = tc.array_to_tensor(theta)
        psi0 = model.makeIniVec(nQ)

        """-----------------------------------------
        ◆ Evolution
        -----------------------------------------"""
        energyList = []
        psi = varWaveFunc(theta, psi0)

        for n in range(maxItr):
            psi = varWaveFunc(theta, psi0)
            lhs = lhs_matrix(theta, psi0)
            rhs = rhs_vector(theta, psi0)
            theta = update(theta, lhs, rhs, dTau)

            energy = calcEnergy(psi).real
            energyList.append(energy)

            if abs(energy - E0) < 1e-4:
                print(f"n = {n}, E = {energy}")
                totalSteps += (n + 1)
                success += 1
                break
            elif n == maxItr - 1:
                print(f"Failed to reach E0 and E = {energyList[-1]}")

    print("-" * 77)
    print(f"* E0 = {E0}\n* E1 = {E1}")
    print(f"* Success = {success} / {nShots}")
    print(f"* Average steps = {totalSteps / nShots}")
    # --------------------------------------------------------------------

    time1 = time()
    model.reportTimeUsed(time0, time1)