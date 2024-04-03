import matplotlib.pyplot as plt
import numpy as np
import model
from time import time
from scipy import sparse
import scipy

Lx, Ly = 4, 4
#J, K, Kp = 0.1, 0.1, 0.1
#J, K, Kp = 1.0, 1.0, 1.0
J, K, Kp = 1.0, -1.0, -0.9

ds = 1e-3
N = int(1 / ds) + 1

multiEigenVals = []
nE = 5
nE_plot = 3

calc, plot = False, True
findPhaseTransition = True

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Report the environment
    # --------------------------------------------------------------------------
    print("-" * 77)
    print("* Exact energy of the QA Hamiltonian")
    print(f"* {Lx} by {Ly} model, J = {J}, K = {K}, Kp = {Kp}")
    print(f"* ds = {ds}, total number of steps: {N}")
    print("-" * 77)

    if calc:
        t0 = time()
        # --------------------------------------------------------------------------
        # Making H0 and H1
        # --------------------------------------------------------------------------
        print("# Preparations...")
        print("\t- Making H0_sp...")
        H0_sp = sparse.csr_matrix(model.makeH_sp(Lx, Ly, J, K, Kp).todense())
        print("\t- Making H1_sp... (this may take some time)")
        H1_sp = sparse.csr_matrix(model.makeH_aux(int(Lx * Ly)))
        print("-" * 77)

        # --------------------------------------------------------------------------
        # Change s
        # --------------------------------------------------------------------------
        print("# Calculations...")
        for i in range(N):
            s = i * ds
            if i % 10 == 0:
                print(f"\t- s = {s}...")
            H_sp = s * H0_sp + (1 - s) * H1_sp
            val, _ = scipy.sparse.linalg.eigsh(H_sp, k=nE, which="SA")
            multiEigenVals.append(val)
        print("-" * 77)

        # --------------------------------------------------------------------------
        # Saving energyLists
        # --------------------------------------------------------------------------
        for i in range(nE):
            model.saveE_list(f"multiEnergies_J{J}_K{K}_Kp{Kp}_E{i}", np.array(multiEigenVals)[:, i])

        # --------------------------------------------------------------------------
        # Time used
        # --------------------------------------------------------------------------
        t1 = time()
        model.reportTimeUsed(t0, t1)
        print("-" * 77)

    if findPhaseTransition:
        E0_list = model.loadE_list(f"multiEnergies_J{J}_K{K}_Kp{Kp}_E{0}")
        E1_list = model.loadE_list(f"multiEnergies_J{J}_K{K}_Kp{Kp}_E{1}")
        E2_list = model.loadE_list(f"multiEnergies_J{J}_K{K}_Kp{Kp}_E{2}")
        for i, e in enumerate(E0_list):
            if abs(E0_list[i] - E1_list[i]) < 1e-2:
                print(f"\tFind possible transition point at s = {i * ds}")
                print(f"\t\tE0(s) = {E0_list[i]}")
                print(f"\t\tE1(s) = {E1_list[i]}")
                print(f"\t\tGap between E0, E1 = {abs(E0_list[i] - E1_list[i])}")
                print(f"\t\tGap between E1, E2 = {abs(E1_list[i] - E2_list[i])}")
                break
        print("-" * 77)
        print(f"GAP at 800: {E2_list[850] - E1_list[800]}")


    if plot:
        markers = ["s", "x", "^"]

        plt.figure(figsize=(6.5, 4.5), dpi=100)
        plt.subplots_adjust(left=0.12, bottom=0.13, right=0.93, top=0.9)

        for i in range(nE_plot):
            plt.plot(
                [i * ds for i in range(N)], model.loadE_list(f"multiEnergies_J{J}_K{K}_Kp{Kp}_E{i}"),
                marker=markers[i], markevery=int((N + 1) / 10),linewidth=1.0, linestyle='--', label=fr"$E_{i}$"
            )

        plt.xlabel(r"$s$", fontsize=13)
        plt.ylabel("Energy", fontsize=13)

        """if J != K:
            plt.title(f"{Lx} × {Ly} lattice, square model, J = {J}, K = {K}, Kp = {Kp}", fontsize=13)
        else:
            plt.title(f"{Lx} × {Ly} lattice, 2D Ising model, J = {J}")"""
        plt.legend(fontsize=13)

        #plt.xlim((0.6, 0.9))
        #plt.ylim((-14.5, -13.5))

        plt.savefig(f"./figs/multiEnergies_J{J}_K{K}_Kp{Kp}.eps", dpi=300)
        plt.show()