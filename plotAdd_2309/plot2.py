# ███████████████████████████████████████████████████████████████████████████████████
#       Description:
#           ◆ Some additional plots about the paper
#       Author:
#           ◆ Yiming-Ding @Westlake University
#       Updated on:
#           ◆ Sep 19, 2023
# ███████████████████████████████████████████████████████████████████████████████████
import matplotlib.pyplot as plt
from header import loadList, getE0E1_sq

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ Public params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
Lx, Ly = 4, 4       # fixed
J, K, Kp = 1.0, -1.0, -0.9
E0, E1 = getE0E1_sq(Lx, Ly, J, K, Kp)

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ QA params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
dt = 0.1
ds = 0.0025
N = int(1 / ds)
print(f"T for QA: {int(1 / ds * dt)}")

""" ------------------------------------------------------------------------
----------------------------------------------------------------------------
        @ VQITE params
----------------------------------------------------------------------------
------------------------------------------------------------------------ """
stepsCut = 800
dTau = 0.05
stdDev = 0.05

lineWidth = 2.5

E_list_ITE = loadList(f'../ite/model_sq/exactEnergyList/ITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}.dat')
E_list_VQITE = loadList(f'../vqite/model_sq/energyList/VQITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}_stdDev{stdDev}.dat')
E_list_QA = loadList(f'../qa/model_sq/energyList/QA_J{J}_K{K}_Kp{Kp}_ds{ds}_dt{dt}.dat')
E_list_VQITE_DIAG = loadList(f'../diag-vqite/model_sq/energyList/VQITE_J{J}_K{K}_Kp{Kp}_dTau{dTau}_stdDev{stdDev}.dat')

print("Final energy:")
print(f"\tITE: {E_list_ITE[-1]}\n\tVQITE: {E_list_VQITE[-1]}\n\tQA: {E_list_QA[-1]}")


fig = plt.figure(figsize=(7.5, 5.0), dpi=300)
plotParams = {
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 18,
}
plt.rcParams.update(plotParams)
ax1 = plt.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.99)

# ------------------------------------------------------------------
#       ax1
# ------------------------------------------------------------------

plt.plot(
    [i * dt for i in range(N)], E_list_QA,
    color="#B22222", linewidth=lineWidth, linestyle='-',
    #marker="s", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QA"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_ITE[:stepsCut],
    color="#1E90FF", linewidth=lineWidth, linestyle='-',
    #marker="o", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE[:stepsCut],
    color="#EEAD0E", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE_DIAG[:stepsCut],
    color="#339933", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"Diag-VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], [E0 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle='-.', label=r'$E_0$'
)

plt.plot(
    [i * dTau for i in range(stepsCut)], [E1 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle=':', label=r'$E_1$'
)

plt.xlabel(r"$T$")
plt.ylabel(r"$\langle \hat H_0\rangle$")
plt.legend()
# ------------------------------------------------------------------
#       ax2
# ------------------------------------------------------------------
left, bottom, width, height = 0.425, 0.75, 0.2, 0.2
ax2 = fig.add_axes([left, bottom, width, height])
plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_ITE[:stepsCut],
    color="#1E90FF", linewidth=lineWidth, linestyle='-',
    #marker="o", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"QITE"
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE[:stepsCut],
    color="#EEAD0E", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE_DIAG[:stepsCut],
    color="#339933", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"VQITE-Diag"
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], [E0 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle='-.', label=r'$E_0$'
)

ax2.plot(
    [i * dTau for i in range(stepsCut)], [E1 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle=':', label=r'$E_1$'
)

plt.xlim((0, 12))
plt.ylim((-17, -15))

# ------------------------------------------------------------------
#       ax3
# ------------------------------------------------------------------
left, bottom, width, height = 0.75, 0.32, 0.15, 0.15
ax3 = fig.add_axes([left, bottom, width, height])
plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_ITE[:stepsCut],
    color="#1E90FF", linewidth=lineWidth, linestyle='-',
    label=r"QITE"
)

ax3.plot(
    [i * dt for i in range(N)], E_list_QA,
    color="#B22222", linewidth=lineWidth, linestyle='-',
    label=r"QA"
)

ax3.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE[:stepsCut],
    color="#EEAD0E", linewidth=lineWidth, linestyle='-',
    label=r"VQITE"
)

plt.plot(
    [i * dTau for i in range(stepsCut)], E_list_VQITE_DIAG[:stepsCut],
    color="#339933", linewidth=lineWidth, linestyle='-',
    #marker=">", markersize=10, markevery=int(stepsCut/10 - 1),
    label=r"Diag-VQITE"
)

ax3.plot(
    [i * dTau for i in range(stepsCut)], [E0 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle='-.', label=r'$E_0$'
)

ax3.plot(
    [i * dTau for i in range(stepsCut)], [E1 for i in range(stepsCut)],
    color="black", linewidth=lineWidth, linestyle=':', label=r'$E_1$'
)

plt.xlim((35, 41))
plt.ylim((-17, -15))

plt.savefig(f"./figs/QA_ITE_VQITE_sq_Lx{Lx}_Ly{Ly}_J{J}_K{K}_Kp{Kp}_ds{ds}_dt{dt}.pdf", dpi=300)
plt.show()