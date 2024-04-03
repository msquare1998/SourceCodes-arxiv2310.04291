

# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ QA for the triangular model
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Aug 15, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import model
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time

tc.set_backend("jax")
tc.set_dtype("complex128")

ds = 1e-4
dt = 0.1
N = int(1 / ds)

Jx, Ja = 0.9, 1.0
Lx, Ly = 4, 4
nQ = int(Lx * Ly)


if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)
    print("Exact energy:")
    print(f"\tE0 = {E0}\n\tE1 = {E1}")

    plt.figure(figsize=(6.5, 4.5), dpi=100)
    plt.subplots_adjust(left=0.12, bottom=0.13, right=0.93, top=0.9)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    dsList = [1e-4, 1e-5]
    colors = ["#FFA500", "#B22222"]
    markers = ["s", "x"]
    labels = [r"QA, $T = 10^3$", r"QA, $T = 10^4$"]
    sizes = [10, 10]

    for i, ds in enumerate(dsList):
        N = int(1 / ds)
        E_list = model.loadE_list(f"QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}")
        print(f"Final energies derived by QA: {E_list[-1]}")

        plt.plot(
            [i * ds for i in range(N)], E_list,
            color=colors[i], linewidth=1.0, linestyle='-',
            marker=markers[i], markersize=sizes[i], markevery=int(N / 10 - 1),
            label=labels[i]
        )

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    ds_ = 1e-3
    N_ = int(1 / ds_)
    plt.plot(
        [i * ds_ for i in range(N_)], [E0 for _ in range(N_)],
        color="black", linewidth=1.0, linestyle='-', label=r"$E_0$"
    )
    plt.plot(
        [i * ds_ for i in range(N_)], [E1 for _ in range(N_)],
        color="black", linewidth=1.0, linestyle=':', label=r"$E_1$"
    )

    plt.xlabel(r"$s$", fontsize=15)
    plt.ylabel(r"$\langle H\rangle$", fontsize=15)
    #plt.title(r"${}\times {}$ Lattice".format(Lx, Ly), fontsize=13)
    plt.legend(fontsize=15)
    plt.savefig(f"./figs/diffT_QA_Lx{Lx}_Ly{Ly}_Jx{Jx}_Ja{Ja}.eps", dpi=300)
    plt.show()