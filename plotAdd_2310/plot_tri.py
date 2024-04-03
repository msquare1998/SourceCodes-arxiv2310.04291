# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Plotting the comparison between QA and SQA for the triangular model
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Oct 06, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import model_qa_tri as model
import matplotlib.pyplot as plt
import tensorcircuit as tc
from time import time

tc.set_backend("jax")
tc.set_dtype("complex128")

ds = 1e-5
ds_sqa = 1.6e-5
dt = 0.1
N = int(1 / ds)

Jx, Ja = 0.9, 1.0
Lx, Ly = 4, 4
nQ = int(Lx * Ly)

if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)

    # ------------------------------------------------------
    #   Plotting and figure saving
    # ------------------------------------------------------

    lineWidth = 2.5

    E_list_QA = model.loadList(f"../qa/model_tri/energyList/QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}.dat")

    E_list_SQA = model.loadList(f"./dataList_tri/SQA_E_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds_sqa}.dat")
    t_list_SQA = model.loadList(f"./dataList_tri/SQA_T_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds_sqa}.dat")

    print("Exact energy:")
    print(f"\tE0 = {E0}\n\tE1 = {E1}")
    print(f"Final energies derived by QA: {E_list_QA[-1]}")

    fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
    plotParams = {
        'axes.labelsize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 18,
    }
    plt.rcParams.update(plotParams)
    ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

    plt.plot(
        [i * dt for i in range(N)], E_list_QA,
        color="#B22222", linewidth=lineWidth, linestyle='-',
        #marker="s", markersize=10, markevery=int(N/10 - 1),
        label="QA"
    )

    plt.plot(
        t_list_SQA[:100000], E_list_SQA[:100000],
        color="#0000cc", linewidth=lineWidth, linestyle='-',
        # marker="s", markersize=10, markevery=int(100000/10) -  1,
        label="SQA"
    )

    plt.plot(
        [i * dt for i in range(N)], [E0 for _ in range(N)],
        color="black", linewidth=lineWidth, linestyle='-.', label=r"$E_0$"
    )
    plt.plot(
        [i * dt for i in range(N)], [E1 for _ in range(N)],
        color="black", linewidth=lineWidth, linestyle=':', label=r"$E_1$"
    )

    plt.xlabel(r"$T$")
    plt.ylabel(r"$\langle \hat H_0\rangle$")

    #plt.title(f"QA, {Lx} × {Ly}, Jx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}", fontsize=13)
    plt.legend()
    plt.savefig(f"./figs/SQA_QA_Lx{Lx}_Ly{Ly}_ds{ds}_dt{dt}_Jx{Jx}_Ja{Ja}.pdf", dpi=300)
    plt.show()
