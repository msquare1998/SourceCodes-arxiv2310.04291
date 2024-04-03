import modules as md
from time import time
import numpy as np
import matplotlib.pyplot as plt

Lx, Ly = 4, 4
J, K, Kp = 1.0, -1.0, -0.9
nQ = int(Lx * Ly)
linkJ = np.array([[4, 5], [5, 6], [6, 7], [7, 4],
                  [12, 13], [13, 14], [14, 15], [15, 12]])
linkK = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                  [8, 9], [9, 10], [10, 11], [11, 8],
                  [0, 4], [2, 6], [5, 9], [7, 11], [8, 12], [10, 14], [13, 1], [15, 3]])
linkKp = np.array([[1, 5], [3, 7], [4, 8], [6, 10],
                   [9, 13], [11, 15], [12, 0], [14, 2]])

dTau, steps = 0.05, 1000
reEvolve, plot = True, True

def calcEnergy(numQ, vec, H):
    dim = 2 ** numQ
    E = 0
    for i in range(dim):
        E += H[i][0] * (abs(vec[i][0]) ** 2)
    return E

def main():
    H, E0, E1 = md.makeH(nQ, J, K, Kp, linkJ, linkK, linkKp)

    if reEvolve:
        stateVec = md.makeIniVec(nQ)

        imgEvo = md.makeImEvoOperator(nQ, dTau, H)

        f0 = open(f"./exactEnergyList/ITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}.dat", "w", encoding="utf-8")

        for i in range(steps):
            print("τ = {}...".format(i * dTau), end="\t")
            energy = calcEnergy(nQ, stateVec, H)
            print("Exact energy = {}".format(energy))
            f0.write(str(energy) + "\n")

            stateVec = stateVec * imgEvo
            md.normalization(nQ, stateVec)
            md.checkNormalized(nQ, stateVec)

        f0.close()
        print("The list of energies is saved.")

    if plot:
        f0 = open(f"./exactEnergyList/ITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}.dat", "r", encoding="utf-8")
        E_list = []
        idx = 0
        while True:
            line0 = f0.readline()
            if line0:
                E_list.append(float(line0.strip('\n')))
                idx += 1
            else:
                break
        f0.close()

        print(f"Exact energy:\n\tE0 = {E0}\n\tE1 = {E1}")
        print(f"Final energies derived by ITE:\n\t{E_list[-1]}")

        plt.figure(figsize=(6, 4), dpi=100)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
        plt.plot(
            [i * dTau for i in range(steps)], E_list,
            color="red", linewidth=1.2, linestyle='-', label="QA, circuit"
        )
        plt.plot(
            [i * dTau for i in range(steps)], [E0 for _ in range(steps)],
            color="#0000CD", linewidth=1.2, linestyle='-.', label=r"$E_0$"
        )

        plt.xlabel("s")
        plt.ylabel("Energy")

        plt.legend()
        plt.show()
        plt.savefig(f"./figs/QA_J{J}_K{K}_Kp{Kp}.eps", dpi=300)


if __name__ == "__main__":
    startTime = time()
    main()
    endTime = time()
    md.reportTimeUsed(startTime, endTime)
