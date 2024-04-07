# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Sweeping QA for the triangular lattice
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Sep 30, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import tensorcircuit as tc
import matplotlib.pyplot as plt
from time import time
import model
from model import calcEnergy
import schematic
from schematic import f_0, f_1, f_2, f_3, h_star, s_star

tc.set_backend("jax")
tc.set_dtype("complex128")
reEvolve = False
plot = True

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

dt = 0.1
#ds = 1.6e-5
ds = 1.6e-5

Jx, Ja = 0.9, 1.0
Lx, Ly = 4, 4   # Do not change the size!
nQ = int(Lx * Ly)

def subEvo_edgeCooling(vec, s, whichEdge):
    edgeList = model.edges[whichEdge]

    # ---------------------------------
    # expm[-i * s * H_0 dt]
    # ---------------------------------
    vec = model.makeDiagEvoOp(nQ, s * H_dg, dt) * vec

    circ = tc.Circuit(nQ, inputs=vec)

    # ---------------------------------
    # expm[-i * (1 - s) * H_1 dt]
    # ---------------------------------
    for q in range(nQ):
        if q not in edgeList:
            circ.rx(q, theta=2 * (1 - s) * dt)

    # ---------------------------------
    # expm[-i * f * H_e * dt]
    # ---------------------------------
    if whichEdge == 0:
        for q in edgeList:
            circ.rx(int(q), theta=2 * f_0(s) * dt)
    elif whichEdge == 1:
        for q in edgeList:
            circ.rx(int(q), theta=2 * f_1(s) * dt)
    elif whichEdge == 2:
        for q in edgeList:
            circ.rx(int(q), theta=2 * f_2(s) * dt)
    elif whichEdge == 3:
        for q in edgeList:
            circ.rx(int(q), theta=2 * f_3(s) * dt)

    return circ.state()

def subEvo_standardQA(vec, s):
    # ---------------------------------
    # expm(-i H_0 dt)
    # ---------------------------------
    vec = model.makeDiagEvoOp(nQ, s * H_dg, dt) * vec

    # ---------------------------------
    # expm(-i H_1 dt)
    # ---------------------------------
    circ = tc.Circuit(nQ, inputs=vec)
    for q in range(nQ):
        circ.rx(q, theta=2 * (1 - s) * dt)

    return circ.state()


def subEvoEdgeWarming(vec, s_fix, f_edge):
    # ---------------------------------
    # expm[-i * s * H_0 dt]
    # ---------------------------------
    vec = model.makeDiagEvoOp(nQ, s_fix * H_dg, dt) * vec

    circ = tc.Circuit(nQ, inputs=vec)

    # ---------------------------------
    # expm[-i * (1 - s) * H_1 dt]
    # ---------------------------------
    for q in range(nQ):
        if q not in edgeList:
            circ.rx(q, theta=2 * (1 - s_fix) * dt)

    # ---------------------------------
    # expm[-i * f * H_e * dt]
    # ---------------------------------
    for q in edgeList:
        circ.rx(int(q), theta=2 * f_edge * dt)
        # circ.rx(int(q), theta=2 * (1 - s_fix) * dt)

    return circ.state()

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == "__main__":
    H_sp, H_dg, E0, E1 = model.makeH_sp(Lx, Ly, Jx, Ja)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    """
    ███████████████████████████████████████████████████████████████████████████████████
        @   SQA with quantum circuit
    ███████████████████████████████████████████████████████████████████████████████████"""
    if reEvolve:
        time_0 = time()
        print(f"Sweeping Quantum Annealing:\n\tJx = {Jx}, Ja = {Ja}, ds = {ds}, dt = {dt}")
        print("-" * 77)

        vec = model.makeIniVec(nQ)
        t = 0
        s = 0

        t_list = []
        E_list = []

        # ------------------------------------------------------------------------
        #   Cooling edge 0
        # ------------------------------------------------------------------------
        edgeLabel = 0
        print(f"Cooling edge {edgeLabel}...")
        for i in range(int(schematic.A_s / ds) - 1):
            vec = subEvo_edgeCooling(vec, s, edgeLabel)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   Standard QA
        # ------------------------------------------------------------------------
        print("Standard QA...")
        for i in range(int((schematic.s_star / 4 - schematic.A_s) / ds)):
            vec = subEvo_standardQA(vec, s)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   The first warming of edge
        # ------------------------------------------------------------------------
        """
        0.25 * s_max / ds == (h_max - (1 - s_fix)) / ds_warming
        """

        edgeLabel = 1
        print(f"Warming edge {edgeLabel}...")
        s_fix = s - ds              # Fix s
        edgeList = model.edges[edgeLabel]

        equiv_distance = schematic.h_star - (1 - s_fix)
        ds_warming = equiv_distance * ds / (0.25 * schematic.s_star)
        N_warming = int(equiv_distance / ds_warming)

        f_edge = 1 - s_fix + ds_warming     # from ( 1 - s_fix ) to (h_star)

        for i in range(N_warming):
            vec = subEvoEdgeWarming(vec, s_fix, f_edge)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\tf_edge = {f_edge}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            f_edge += ds_warming
            t += dt
        print("")

        # ------------------------------------------------------------------------
        #   Cooling edge 1
        # ------------------------------------------------------------------------
        print(f"Cooling edge {edgeLabel}...")
        for i in range(int((schematic.B_s - schematic.s_star / 4) / ds)):
            vec = subEvo_edgeCooling(vec, s, edgeLabel)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)
            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   Standard QA
        # ------------------------------------------------------------------------
        print("Standard QA...")
        for i in range(int((schematic.s_star / 2 - schematic.B_s) / ds)):
            vec = subEvo_standardQA(vec, s)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   The 2nd warming of edge
        # ------------------------------------------------------------------------
        edgeLabel = 2
        print(f"Warming edge {edgeLabel}...")

        s_fix = s - ds  # Fix s
        edgeList = model.edges[edgeLabel]

        equiv_distance = schematic.h_star - (1 - s_fix)
        ds_warming = equiv_distance * ds / (0.25 * schematic.s_star)
        N_warming = int(equiv_distance / ds_warming)

        f_edge = 1 - s_fix + ds_warming  # from ( 1 - s_fix ) to (h_star)

        for i in range(N_warming):
            vec = subEvoEdgeWarming(vec, s_fix, f_edge)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\tf_edge = {f_edge}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            f_edge += ds_warming
            t += dt
        print("")

        # ------------------------------------------------------------------------
        #   Cooling edge 2
        # ------------------------------------------------------------------------
        print(f"Cooling edge {edgeLabel}...")
        for i in range(int((schematic.C_s - schematic.s_star / 2) / ds)):
            vec = subEvo_edgeCooling(vec, s, edgeLabel)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)
            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   Standard QA
        # ------------------------------------------------------------------------
        print("Standard QA...")
        for i in range(int((3 * schematic.s_star / 4 - schematic.C_s) / ds)):
            vec = subEvo_standardQA(vec, s)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)
            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   The 3rd warming of edge
        # ------------------------------------------------------------------------
        edgeLabel = 3
        print(f"Warming edge {edgeLabel}...")

        s_fix = s - ds  # Fix s
        edgeList = model.edges[edgeLabel]

        equiv_distance = schematic.h_star - (1 - s_fix)
        ds_warming = equiv_distance * ds / (0.25 * schematic.s_star)
        N_warming = int(equiv_distance / ds_warming)

        f_edge = 1 - s_fix + ds_warming  # from ( 1 - s_fix ) to (h_star)

        for i in range(N_warming):
            vec = subEvoEdgeWarming(vec, s_fix, f_edge)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)

            print(f"\r\tf_edge = {f_edge}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            f_edge += ds_warming
            t += dt
        print("")

        # ------------------------------------------------------------------------
        #   Cooling edge 3
        # ------------------------------------------------------------------------
        print(f"Cooling edge {edgeLabel}...")
        for i in range(int((schematic.D_s - 3 * schematic.s_star / 4) / ds)):
            vec = subEvo_edgeCooling(vec, s, edgeLabel)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)
            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        # ------------------------------------------------------------------------
        #   Standard QA
        # ------------------------------------------------------------------------
        print("Standard QA...")
        for i in range(int((1 - schematic.D_s) / ds)):
            vec = subEvo_standardQA(vec, s)
            energy = calcEnergy(vec, H_sp)

            t_list.append(t)
            E_list.append(energy)
            print(f"\r\ts = {s}" + f"\t\tt = {t}" + f"\t\tE = {energy}", end="")

            s += ds
            t += dt

        print("")

        time_1 = time()

        model.saveList(f"./dataList/SQA_E_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds}.dat", E_list)
        model.saveList(f"./dataList/SQA_T_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds}.dat", t_list)

        model.reportTimeUsed(time_0, time_1)
        print("-" * 77)

    """
    ███████████████████████████████████████████████████████████████████████████████████
        @   Plotting
    ███████████████████████████████████████████████████████████████████████████████████"""
    if plot:
        lineWidth = 2.5

        E_list = model.loadList(f"./dataList/SQA_E_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds}.dat")
        t_list = model.loadList(f"./dataList/SQA_T_list_Lx{Lx}_Ly{Ly}_dt{dt}_{ds}.dat")

        print("Exact energy:")
        print(f"\tE0 = {E0}\n\tE1 = {E1}")
        print(f"Final energies derived by QA: {E_list[-1]}")

        fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
        plotParams = {
            'axes.labelsize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 18,
        }
        plt.rcParams.update(plotParams)
        #ax1 = plt.subplots_adjust(left=0.2, bottom=0.13, right=0.93, top=0.9)
        ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

        plt.plot(
            t_list[:100000], E_list[:100000],
            color="#BA55D3", linewidth=lineWidth, linestyle='-',
            #marker="s", markersize=10, markevery=int(100000/10) -  1,
            label="SQA"
        )

        plt.plot(
            [_ * dt for _ in range(len(E_list[:100000]))], [E0 for _ in range(len(E_list[:100000]))],
            color="black", linewidth=lineWidth, linestyle='-.', label=r"$E_0$"
        )
        plt.plot(
            [_ * dt for _ in range(len(E_list[:100000]))], [E1 for _ in range(len(E_list[:100000]))],
            color="black", linewidth=lineWidth, linestyle=':', label=r"$E_1$"
        )

        plt.xlabel(r"$T$")
        plt.ylabel(r"$\langle \hat H_0\rangle$")
        plt.legend()
        print(len(E_list))

        plt.savefig(f"./figs/SQA_tri_ds{ds}.pdf", dpi=300)
        plt.show()
