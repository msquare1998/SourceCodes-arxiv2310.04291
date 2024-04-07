import matplotlib.pyplot as plt
import numpy as np

h_star = 2
s_star = 0.8

ds = 0.001
N = int(1 / ds)

f_aux = lambda x: - x + 1
f_0 = lambda x: - 4 * h_star / s_star * x + h_star
f_1 = lambda x: - 4 * h_star / s_star * x + 2 * h_star
f_2 = lambda x: - 4 * h_star / s_star * x + 3 * h_star
f_3 = lambda x: - 4 * h_star / s_star * x + 4 * h_star

A_s = (1 - h_star) / (1 - 4 * h_star / s_star)
B_s = (1 - 2 * h_star) / (1 - 4 * h_star / s_star)
C_s = (1 - 3 * h_star) / (1 - 4 * h_star / s_star)
D_s = (1 - 4 * h_star) / (1 - 4 * h_star / s_star)

if __name__ == "__main__":
    plt.figure(figsize=(7, 7), dpi=300)
    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.93, top=0.9)

    """"███████████████████████████████████████████████████████████████████████████████████
            @ Intensities
                - SS: solid start
                - SE: solid end
                - DS: dashed start
                - DE: dahsed end
    ███████████████████████████████████████████████████████████████████████████████████"""
    # -------------------------------------------------------------------------------------
    #   f0
    # -------------------------------------------------------------------------------------
    f0_SS, f0_SE = 0, A_s
    f0_DS, f0_DE = A_s, s_star / 4
    f0_S_length, f0_D_length = f0_SE - f0_SS, f0_DE - f0_DS

    plt.plot([i * ds for i in range(int(f0_S_length / ds) + 1)],
             np.array([f_0(i * ds) for i in range(int(f0_S_length / ds) + 1)]),
             color="#0066ff", linewidth=3, linestyle='-')

    plt.plot([i * ds for i in range(int(f0_S_length / ds) + 1, int(f0_DE / ds + 1))],
             np.array([f_0(i * ds) for i in range(int(f0_S_length / ds) + 1, int(f0_DE / ds + 1))]),
             color="#0066ff", linewidth=1, linestyle=':')

    # -------------------------------------------------------------------------------------
    #   f1
    # -------------------------------------------------------------------------------------
    f1_SS, f1_SE = s_star / 4, B_s
    f1_DS, f1_DE = B_s, 2 * s_star / 4
    f1_S_length, f1_D_length = f1_SE - f1_SS, f1_DE - f1_DS

    plt.plot([i * ds for i in range(
        int(f1_SS / ds + 1),
        int(f1_SS / ds + 1) + int(f1_S_length / ds) + 1
    )],
             np.array([f_1(i * ds) for i in range(
                 int(f1_SS / ds + 1),
                 int(f1_SS / ds + 1) + int(f1_S_length / ds) + 1
             )]),
             color="#0066ff", linewidth=3, linestyle='-', label=r"Glueing")

    plt.plot([i * ds for i in range(
        int(f1_SS / ds + 1) + int(f1_S_length / ds) + 1,
        int(f1_DS / ds + 1) + int(f1_D_length / ds) + 1
    )],
             np.array([f_1(i * ds) for i in range(
                 int(f1_SS / ds + 1) + int(f1_S_length / ds) + 1,
                 int(f1_DS / ds + 1) + int(f1_D_length / ds) + 1
             )]),
             color="#0066ff", linewidth=1, linestyle=':')

    # -------------------------------------------------------------------------------------
    #   f2
    # -------------------------------------------------------------------------------------
    f2_SS, f2_SE = 2 * s_star / 4, C_s
    f2_DS, f2_DE = C_s, 3 * s_star / 4
    f2_S_length, f2_D_length = f2_SE - f2_SS, f2_DE - f2_DS

    plt.plot([i * ds for i in range(
        int(f2_SS / ds + 1),
        int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1
    )],
             np.array([f_2(i * ds) for i in range(
                 int(f2_SS / ds + 1),
                 int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1
             )]),
             color="#0066ff", linewidth=3, linestyle='-')

    plt.plot([i * ds for i in range(
        int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1,
        int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1 + int(f2_D_length / ds)
    )],
             np.array([f_2(i * ds) for i in range(
                 int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1,
                 int(f2_SS / ds + 1) + int(f2_S_length / ds) + 1 + int(f2_D_length / ds)
             )]),
             color="#0066ff", linewidth=1, linestyle=':')

    # -------------------------------------------------------------------------------------
    #   f3
    # -------------------------------------------------------------------------------------
    f3_SS, f3_SE = 3 * s_star / 4, D_s
    f3_DS, f3_DE = D_s, s_star
    f3_S_length, f3_D_length = f3_SE - f3_SS, f3_DE - f3_DS

    plt.plot([i * ds for i in range(
        int(f3_SS / ds + 1),
        int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1
    )],
             np.array([f_3(i * ds) for i in range(
                 int(f3_SS / ds + 1),
                 int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1
             )]),
             color="#0066ff", linewidth=3, linestyle='-')

    plt.plot([i * ds for i in range(
        int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1,
        int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1 + int(f3_D_length / ds)
    )],
             np.array([f_3(i * ds) for i in range(
                 int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1,
                 int(f3_SS / ds + 1) + int(f3_S_length / ds) + 1 + int(f3_D_length / ds)
             )]),
             color="#0066ff", linewidth=1, linestyle=':')


    """███████████████████████████████████████████████████████████████████████████████████
                @ Auxiliary lines
    ███████████████████████████████████████████████████████████████████████████████████"""
    plt.plot([i * ds for i in range(N + 1)],
             np.array([f_aux(i * ds) for i in range(N + 1)]),
             color="black", linewidth=3, linestyle='-', label=r"Standard QA")

    plt.plot([i * ds for i in range(N + 1)], [0.0 for i in range(N + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([i * ds for i in range(N + 1)], [h_star for i in range(N + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([1.0 for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([0.0 for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle=':')

    plt.plot([s_star / 4 for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([2 * s_star / 4 for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([3 * s_star / 4 for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle=':')
    plt.plot([s_star for i in range(int(h_star / ds) + 1)], [i * ds for i in range(int(h_star / ds) + 1)],
             color="black", linewidth=1, linestyle='-.')

    a0 = h_star - f_aux(s_star / 4)
    plt.plot([s_star / 4 for i in range(int(f_aux(s_star / 4) / ds), int((f_aux(s_star / 4) + a0) / ds))],
             [i * ds for i in range(int(f_aux(s_star / 4) / ds), int((f_aux(s_star / 4) + a0) / ds))],
             color="#B22222", linewidth=3, linestyle='-', label="Opening")

    plt.xlabel(r"$s$", fontsize=15)

    plt.legend(fontsize=15)

    plt.savefig("./figs/schematic_SQA.eps", dpi=300)
    plt.show()
