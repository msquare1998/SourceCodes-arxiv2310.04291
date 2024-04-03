# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ VQITE for the square model
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 15, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import numpy as np
import tensorcircuit as tc
from time import time
import model
from model import linkJ, linkK, linkKp
import argparse
import matplotlib.pyplot as plt

np.random.seed(7)
tc.set_backend("jax")
tc.set_dtype("complex128")
parser = argparse.ArgumentParser()
parser.add_argument("--Kp", type=float, default="-0.9")
parser.add_argument("--dTau", type=float, default="0.1")
parser.add_argument("--steps", type=int, default="120")
args = parser.parse_args()

l = 1
p = 7

J, K, Kp = 1.0, -1.0, args.Kp
Lx, Ly = 4, 4   # Do not change the size
nQ = int(Lx * Ly)
dTau, steps, BETA = args.dTau, args.steps, args.dTau * args.steps
stdDev = 0.05

reEvolve, plot = False, True
doCounting = True
def varWaveFunc(params, psi0):
    params = tc.backend.reshape(params, [l, nQ * p])
    circ = tc.Circuit(nQ, inputs=psi0)

    for ll in range(l):
        for q in range(nQ):
            circ.rz(q, theta=params[ll, p * q])
            circ.rx(q, theta=params[ll, p * q + 1])
            circ.rz(q, theta=params[ll, p * q + 2])

        traversed = np.zeros(nQ)  # each site relates to two bonds

        for pair in linkJ:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q0] == 0:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q0] = 1
            else:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)

        for pair in linkK:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q0] == 0:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q0] = 1
            else:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)

        for pair in linkKp:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q0] == 0:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q0] = 1
            else:
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=params[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)

    return circ.state()

dPsi_dTheta = tc.backend.jit(tc.backend.jacfwd(varWaveFunc, argnums=0))

def innerProd(i, j):
    return tc.backend.tensordot(tc.backend.conj(i), j, 1)

@tc.backend.jit
def lhs_matrix(theta, psi0):
    vij = tc.backend.vmap(innerProd, vectorized_argnums=0)    # Vectorized operation for calculating a batch of vij
    vvij = tc.backend.vmap(vij, vectorized_argnums=1)
    jacobian = dPsi_dTheta(theta, psi0=psi0)
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
        e = wl @ H @ wr
        return tc.backend.real(e)[0, 0]

    eg = tc.backend.grad(energy, argnums=0)
    rhs = eg(theta, psi0)
    rhs = -tc.backend.real(rhs)
    return rhs

@tc.backend.jit
def update(theta, lhs, rhs, tau):
    eps = 1e-4
    lhs += eps * tc.backend.eye(l * (nQ * p), dtype=lhs.dtype)
    dotTheta = tc.backend.solve(lhs, rhs, assume_a="sym")
    #dotTheta = jnpl.pinv(lhs) @ rhs
    return (theta + tc.backend.cast(tau * dotTheta, dtype=theta.dtype))

@tc.backend.jit
def calcEnergy(vec):
    return (tc.backend.conj(tc.backend.transpose(vec)) @ H @ vec).real

@tc.backend.jit
def testFunc(theta):
    psi = varWaveFunc(theta, vec0)
    E = calcEnergy(psi)
    return tc.backend.real(E)

def loadExactEnergy():
    f0 = open("./exactEnergy/exactEnergy_J{}_K{}_Kp{}_dTau{}.dat".format(J, K, Kp, dTau), "r", encoding="utf-8")
    exactList = []
    while True:
        line0 = f0.readline()
        if line0:
            exactList.append(float(line0.strip('\n')))
        else:
            break
    f0.close()
    return np.array(exactList)


if __name__ == '__main__':
    H, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, J, K, Kp)

    # --------------------------------------------------------------------
    if reEvolve:
        time0 = time()
        vec0 = tc.array_to_tensor(model.makeIniVec(nQ))
        theta = np.random.normal(0, stdDev, size=[l * (nQ * p)])
        theta = tc.backend.cast(tc.array_to_tensor(theta), dtype=float)

        energyList = []
        vec = varWaveFunc(theta, vec0)

        for n in range(steps):
            vec = varWaveFunc(theta, vec0)
            lhs = lhs_matrix(theta, vec0)
            rhs = rhs_vector(theta, vec0)
            theta = update(theta, lhs, rhs, dTau)

            print("β = %.4f / %.4f" % (n * dTau, BETA), end="\t")
            energy = calcEnergy(vec)
            energyList.append(energy)
            print(f"VQITE: {energy}", end="\n")

        model.saveState(vec, nQ, J, K, Kp, dTau, stdDev)
        model.saveE_list(f"VQITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}_stdDev{stdDev}", E_list=energyList)

        time1 = time()
        model.reportTimeUsed(time0, time1)

    # --------------------------------------------------------------------
    if doCounting:
        print("-" * 77)
        nPreserved = model.countPreserved(model.loadState(nQ, J, K, Kp, dTau, stdDev), threshold=1e-3)
        print(f"Number of the preserved state:\n\t{nPreserved} / {2 ** nQ}")

    # --------------------------------------------------------------------
    if plot:
        stepsCut = 120

        print("-" * 77)
        E_list = model.loadE_list(f"VQITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}_stdDev{stdDev}")
        exactE_list = model.load_otherList(f"../../ite/model_sq/exactEnergyList/ITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}")

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
        plt.xlim((-0.1, dTau * stepsCut + 0.1))
        #plt.title(f"Energy variation, dTau = {dTau}, J = {J}, K = {K}, Kp = {Kp}")
        plt.legend(fontsize=13)
        plt.savefig(fname=f"./figs/VQITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}_stdDev{stdDev}.eps")
        plt.show()