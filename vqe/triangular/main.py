import numpy as np
import tensorflow as tf
from jax.config import config
from time import time
import tensorcircuit as tc
import model
import argparse
import matplotlib.pyplot as plt

#tf.random.set_seed(7)
config.update("jax_enable_x64", True)
tc.set_backend("tensorflow")
tc.set_dtype("complex128")
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default="0.02")
parser.add_argument("--maxItr", type=int, default="1000")
args = parser.parse_args()

J, K = 0.9, 1.0
Lx, Ly = 4, 4
nQ = int(Lx * Ly)
maxItr = args.maxItr
lr = args.lr
stdDev = 0.05
opt = tf.keras.optimizers.Adam(lr)

reEvolve = False
plot = True

l = 1
p = 7

def countPreserved(nQ, vec, threshold):
    states = np.array([
        43690, 23210, 42410, 21930,
        43610, 23130, 42330, 21850,
        43685, 23205, 42405, 21925,
        43605, 23125, 42325, 21845
        ])

    cc = 0

    print("State(s) preserved:")
    for i in range(len(vec)):
        prob = abs(vec[i]) ** 2
        if prob > threshold:
            bin0 = f"{bin(i)[2:]:0>{nQ}}"
            bin_rev = list(bin0)
            bin_rev.reverse()
            bin1 = "".join(bin_rev)
            dec1 = int(bin1, 2)
            if dec1 not in states:
                print("NOT IN EXPECTED LIST", end="\t")

            print(f"|{dec1:0>5}>" + f"\t|{bin1}〉with prob = {prob}")
            cc += 1
    return cc


def getState(param):
    circ = tc.Circuit(nQ)
    circ.h(range(nQ))

    paramc = tc.backend.cast(param, tc.dtypestr)  # We assume the input param with dtype float64

    for ll in range(l):
        for q in range(nQ):
            circ.rz(q, theta=paramc[ll, p * q])
            circ.rx(q, theta=paramc[ll, p * q + 1])
            circ.rz(q, theta=paramc[ll, p * q + 2])

        traversed = np.zeros(nQ)  # each site relates to two bonds

        for pair in linkJ:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q1] == 0:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q1] = 1
            else:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)

        for pair in linkK:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q1] == 0:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q1] = 1
            else:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)

    return circ.state()

@tc.backend.jit
def calcEnergy(circ):
    E = 0.0

    for pair in linkJ:
        q0, q1 = int(pair[0]), int(pair[1])
        E += J * circ.expectation((tc.gates.z(), [q0]), (tc.gates.z(), [q1]))

    for pair in linkK:
        q0, q1 = int(pair[0]), int(pair[1])
        E += K * circ.expectation((tc.gates.z(), [q0]), (tc.gates.z(), [q1]))

    return tc.backend.real(E)

def vqe_topo(param):
    circ = tc.Circuit(nQ)
    circ.h(range(nQ))

    paramc = tc.backend.cast(param, tc.dtypestr)  # We assume the input param with dtype float64

    for ll in range(l):
        for q in range(nQ):
            circ.rz(q, theta=paramc[ll, p * q])
            circ.rx(q, theta=paramc[ll, p * q + 1])
            circ.rz(q, theta=paramc[ll, p * q + 2])

        traversed = np.zeros(nQ)  # each site relates to two bonds

        for pair in linkJ:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q1] == 0:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q1] = 1
            else:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)

        for pair in linkK:
            q0, q1 = int(pair[0]), int(pair[1])
            if traversed[q1] == 0:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 3], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 4], unitary=tc.gates._yz_matrix)
                traversed[q1] = 1
            else:
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 5], unitary=tc.gates._zy_matrix)
                circ.exp1(q0, q1, theta=paramc[ll, p * q0 + 6], unitary=tc.gates._yz_matrix)

    E = calcEnergy(circ)
    return E


vqe_topo_vag = tc.backend.jit(tc.backend.value_and_grad(vqe_topo))

def train_step_tf():
    param = tf.Variable(
        initial_value=tf.random.normal(
            shape=[l, nQ * p], stddev=stdDev, dtype=getattr(tf, tc.rdtypestr)
        )
    )

    energyList = []
    for i in range(maxItr):
        energy, grad = vqe_topo_vag(param)
        opt.apply_gradients([(grad, param)])
        print(f"{i} / {maxItr}\tVQE = {energy}")
        energyList.append(float(energy))
        if abs(energy - E0) < 1e-4:
            break

    vec = getState(param)
    nPre = countPreserved(nQ, vec, 1e-4)
    print(f"# of states preserved: {nPre}")

    return energyList


if __name__ == "__main__":
    linkJ = model.makeLinkNNX_PBC(Lx, Ly)
    linkK = model.makeLinkNNA_PBC(Lx, Ly)
    H, E0, E1 = model.makeH_sp(Lx, Ly, J, K, linkJ, linkK)

    if reEvolve:
        time0 = time()
        # ----------------------------------------------------------------

        energyList = train_step_tf()
        model.saveList(f"./energyList/vqe_tri_Lx{Lx}_Ly{Ly}_J{J}_K{K}_lr{lr}_stdDev{stdDev}.dat", energyList)
        # ----------------------------------------------------------------
        time1 = time()
        model.reportTimeUsed(time0, time1)

    if plot:
        stepCut = 120

        energyList = model.loadList(f"energyList/vqe_tri_Lx{Lx}_Ly{Ly}_J{J}_K{K}_lr{lr}_stdDev{stdDev}.dat")
        print("-")
        print(f"* Final energy = {energyList[-1]}")
        print(f"* E0 = {E0}\n* E1 = {E1}")

        fig = plt.figure(figsize=(7.5, 5.0), dpi=300)

        lineWidth = 2.5
        plotParams = {
            'axes.labelsize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 18,
        }
        plt.rcParams.update(plotParams)
        ax1 = plt.subplots_adjust(left=0.16, bottom=0.15, right=0.96, top=0.99)

        plt.plot(
            [i for i in range(stepCut)], energyList[:stepCut],
            color="#9900ff", linewidth=lineWidth, linestyle='-', label=r"VQE"
        )
        plt.plot(
            [i for i in range(stepCut)], [E0 for _ in range(stepCut)],
            color="black", linewidth=lineWidth, linestyle='-.', label=r"$E_0$"
        )
        plt.plot(
            [i for i in range(stepCut)], [E1 for _ in range(stepCut)],
            color="black", linewidth=lineWidth, linestyle=':', label=r"$E_1$"
        )

        plt.xlabel("Iteration")
        plt.ylabel(r"$\langle \hat H_0\rangle$")
        plt.legend()

        # ------------------------------------------------------------------
        #       ax2
        # ------------------------------------------------------------------

        plt.savefig(f"figs/vqe_tri_Lx{Lx}_Ly{Ly}_J{J}_K{K}_lr{lr}_stdDev{stdDev}.pdf", dpi=300)
        plt.show()




