# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ VQITE for the triangular model
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 26, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import numpy as np
import tensorcircuit as tc
from time import time
import model
import matplotlib.pyplot as plt
import argparse

#np.random.seed(7)
tc.set_backend("jax")
tc.set_dtype("complex128")

parser = argparse.ArgumentParser()
parser.add_argument("--Jx", type=float, default="0.9")
parser.add_argument("--Ja", type=float, default="1")
parser.add_argument("--Lx", type=int, default="4")
parser.add_argument("--Ly", type=int, default="4")
parser.add_argument("--dTau", type=float, default="0.1")
parser.add_argument("--steps", type=int, default="100")
args = parser.parse_args()

Jx, Ja = args.Jx, args.Ja
Lx, Ly = args.Lx, args.Ly
nQ = int(Lx * Ly)

dTau, steps = args.dTau, args.steps
BETA = steps * dTau
stdDev = 0.05

l, p = 1, 9

reEvolve, plot = False, True
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
    h, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)

    if reEvolve:
        time0 = time()

        linkNNX = model.makeLinkNNX_PBC(Lx, Ly)
        linkNNA = model.makeLinkNNA_PBC(Lx, Ly)


        theta = np.random.normal(0, stdDev, size=[l * nQ * p])
        theta = tc.array_to_tensor(theta)

        psi0 = model.makeIniVec(nQ)

        """-----------------------------------------
        ◆ Evolution
        -----------------------------------------"""
        energyList = []
        psi = varWaveFunc(theta, psi0)

        for n in range(steps):
            psi = varWaveFunc(theta, psi0)
            lhs = lhs_matrix(theta, psi0)
            rhs = rhs_vector(theta, psi0)
            theta = update(theta, lhs, rhs, dTau)

            print("β = %.3f / %.3f" % (n * dTau, BETA), end="\t")
            energy = calcEnergy(psi).real
            energyList.append(energy)
            print(f"VQITE: {energyList[n]}")

        model.saveState(psi, nQ, Lx, Ly, Jx, Ja, dTau, stdDev)
        model.saveE_list(f"VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}", E_list=energyList)
        time1 = time()
        model.reportTimeUsed(time0, time1)

    # --------------------------------------------------------------------
    if doCounting:
        print("-" * 77)
        nPreserved = model.countPreserved(model.loadState(nQ, Lx, Ly, Jx, Ja, dTau, stdDev), threshold=1e-3)
        print(f"Number of the preserved state:\n\t{nPreserved} / {2 ** nQ}")

    # --------------------------------------------------------------------
    if plot:
        stepsCut = 40

        print("-" * 77)
        E_list = model.loadE_list(f"VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}")
        exactE_list = model.load_otherList(f"../../ite/model_tri/exactEnergyList/ITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print(f"Final energies derived by VQITE: {E_list[-1]}")
        plt.figure(figsize=(6.5, 4.5), dpi=100)

        plt.plot([dTau * i for i in range(stepsCut)], exactE_list[:stepsCut],
                 color="#B22222", linewidth=1.0, linestyle='-',
                 marker="<", markersize=10, markevery=int(stepsCut / 10 - 1),
                 label="ITE")

        plt.plot([dTau * i for i in range(stepsCut)], E_list[:stepsCut],
                 color="#1E90FF", linewidth=1.0, linestyle='-',
                 marker="P", markersize=10, markevery=int(stepsCut / 10 - 1),
                 label="VQITE")

        plt.plot(
            [dTau * i for i in range(stepsCut)], [E0 for _ in range(stepsCut)],
            color="black", linewidth=1.0, linestyle='-', label=r"$E_0$"
        )
        plt.plot(
            [dTau * i for i in range(stepsCut)], [E1 for _ in range(stepsCut)],
            color="black", linewidth=1.0, linestyle=':', label=r"$E_1$"
        )

        plt.xlabel(r"$\tau$", fontsize=15)
        plt.ylabel(r"$\langle H_0\rangle$", fontsize=15)
        #plt.xlim((-0.1, dTau * steps + 0.1))
        plt.xlim((-0.1, dTau * stepsCut + 0.1))
        #plt.title(f"{Lx} × {Ly}, Jx = {Jx}, Ja = {Ja}, dTau = {dTau}, stdDev = {stdDev}")
        plt.legend(fontsize=15)
        plt.savefig(fname=f"./figs/VQITE_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}_dTau{dTau}_stdDev{stdDev}.eps")
        plt.show()