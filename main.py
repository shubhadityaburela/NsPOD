import argparse
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np

cmap = "YlOrRd"


# =============================================================================
# CLI Argument Router
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Specific Tensor Decomposition test case and Plotting")
    parser.add_argument(
        "--run",
        choices=["Single_wave", "Crossing_StraightCubic_waves", "Crossing_sine_StraightCubic_waves", "Wildlandfire_1d"],
        default="Single_wave",
        help="Specify which test run to execute"
    )
    args = parser.parse_args()

    print(f"Executing Pipeline for Run: {args.run}...")

    # =============================================================================
    # Pipeline Conditionals
    # =============================================================================
    if args.run == "Single_wave":
        # 1. Trigger the underlying compute script
        print("-> Running Single_wave.py...")
        subprocess.run(["python3", "Single_wave.py"], check=True)

        # 2. Run the specialized plotting routine
        print("-> Generating plots...")
        impath = "plots/Single_wave/"
        immpath = "data/Single_wave/"
        Q = np.load(immpath + "Q.npy")
        Q_tilde = np.load(immpath + "Q_tilde.npy")
        Q1 = np.load(immpath + "Q1.npy")
        shift_true = np.load(immpath + "shift_true.npy")
        shift = np.load(immpath + "shift.npy")
        vmin = np.min(Q)
        vmax = np.max(Q)

        # Plot the separated frames
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)

        # Original
        im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$Q$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed (\tilde{Q} = T^1Q^1)
        im1 = axs[1].pcolormesh(Q_tilde.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$\tilde{Q}$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im2 = axs[2].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$Q^\theta_1$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
        plt.close(fig)

        rec_err = np.linalg.norm(Q - Q_tilde) / np.linalg.norm(Q)
        print(f"RecErr: {rec_err}")

        # Plot the shifts
        t_start = -10.0
        t_end = 10.0
        t = np.linspace(t_start, t_end, Q.shape[1])
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(t, shift, label="Predicted shift")
        axs.plot(t, shift_true, label="True shift")
        axs.set_ylabel(r"shift")
        axs.set_xlabel(r"$t$")
        axs.grid()
        axs.legend()
        fig.savefig(impath + "shift_comp" + ".pdf", format='pdf', dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)

    elif args.run == "Crossing_StraightCubic_waves":
        # 1. Trigger the underlying compute script
        print("-> Running Crossing_StraightCubic_waves.py...")
        subprocess.run(["python3", "Crossing_StraightCubic_waves.py"], check=True)

        # 2. Run the specialized plotting routine
        print("-> Generating plots...")
        impath = "plots/Crossing_StraightCubic_waves/"
        immpath = "data/Crossing_StraightCubic_waves/"
        Q = np.load(immpath + "Q.npy")
        Q1 = np.load(immpath + "Q1.npy")
        Q2 = np.load(immpath + "Q2.npy")
        T1Q1 = np.load(immpath + "T1Q1.npy")
        T2Q2 = np.load(immpath + "T2Q2.npy")
        Q_tilde = np.load(immpath + "Q_tilde.npy")
        shifts1_true = np.load(immpath + "shifts1_true.npy")
        shifts1 = np.load(immpath + "shifts1.npy")
        shifts2_true = np.load(immpath + "shifts2_true.npy")
        shifts2 = np.load(immpath + "shifts2.npy")
        vmin = np.min(Q)
        vmax = np.max(Q)

        # Plot the separated frames
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
        # Original
        im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$Q$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im1 = axs[1].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$(Q_z)^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed
        im3 = axs[3].pcolormesh(Q_tilde.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$\tilde{Q}$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
        plt.close(fig)

        # Plot the low-rank frames
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
        # Original
        im0 = axs[0].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$(Q_z)^\theta_1$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$Q^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed
        im3 = axs[3].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$Q^\theta_2$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti_low", dpi=300, transparent=True)
        plt.close(fig)

        rec_err = np.linalg.norm(Q - (T1Q1 + T2Q2)) / np.linalg.norm(Q)
        print(f"RecErr: {rec_err}")

        # Plot the shifts
        t_start = -10.0
        t_end = 10.0
        t = np.linspace(t_start, t_end, Q.shape[1])
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(t, shifts1, label="Predicted shift (frame 1)")
        axs.plot(t, shifts1_true, label="True shift (frame 1)")
        axs.plot(t, shifts2, label="Predicted shift (frame 2)")
        axs.plot(t, shifts2_true, label="True shift (frame 2)")
        axs.set_ylabel(r"shift")
        axs.set_xlabel(r"$t$")
        axs.grid()
        axs.legend()
        fig.savefig(impath + "shift_comp" + ".pdf", format='pdf', dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)

    elif args.run == "Crossing_sine_StraightCubic_waves":
        # 1. Trigger the underlying compute script
        print("-> Running Crossing_sine_StraightCubic_waves.py...")
        subprocess.run(["python3", "Crossing_sine_StraightCubic_waves.py"], check=True)

        # 2. Run the specialized plotting routine
        print("-> Generating plots...")
        impath = "plots/Crossing_sine_StraightCubic_waves/"
        immpath = "data/Crossing_sine_StraightCubic_waves/"
        Q = np.load(immpath + "Q.npy")
        Q1 = np.load(immpath + "Q1.npy")
        Q2 = np.load(immpath + "Q2.npy")
        T1Q1 = np.load(immpath + "T1Q1.npy")
        T2Q2 = np.load(immpath + "T2Q2.npy")
        Q_tilde = np.load(immpath + "Q_tilde.npy")
        shifts1_true = np.load(immpath + "shifts1_true.npy")
        shifts1 = np.load(immpath + "shifts1.npy")
        shifts2_true = np.load(immpath + "shifts2_true.npy")
        shifts2 = np.load(immpath + "shifts2.npy")
        vmin = np.min(Q)
        vmax = np.max(Q)

        # Plot the separated frames
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
        # Original
        im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$Q$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im1 = axs[1].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$(Q_z)^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed
        im3 = axs[3].pcolormesh(Q_tilde.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$\tilde{Q}$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
        plt.close(fig)

        # Plot the low-rank frames
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
        # Original
        im0 = axs[0].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$(Q_z)^\theta_1$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$Q^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed
        im3 = axs[3].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$Q^\theta_2$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti_low", dpi=300, transparent=True)
        plt.close(fig)

        rec_err = np.linalg.norm(Q - (T1Q1 + T2Q2)) / np.linalg.norm(Q)
        print(f"RecErr: {rec_err}")

        # Plot the shifts
        t_start = -10.0
        t_end = 10.0
        t = np.linspace(t_start, t_end, Q.shape[1])
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(t, shifts1, label="Predicted shift (frame 1)")
        axs.plot(t, shifts1_true, label="True shift (frame 1)")
        axs.plot(t, shifts2, label="Predicted shift (frame 2)")
        axs.plot(t, shifts2_true, label="True shift (frame 2)")
        axs.set_ylabel(r"shift")
        axs.set_xlabel(r"$t$")
        axs.grid()
        axs.legend()
        fig.savefig(impath + "shift_comp" + ".pdf", format='pdf', dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)

    elif args.run == "Wildlandfire_1d":
        # 1. Trigger the underlying compute script
        print("-> Running Wildlandfire_1d.py...")
        subprocess.run(["python3", "Wildlandfire_1d.py"], check=True)

        # 2. Run the specialized plotting routine
        print("-> Generating plots...")
        impath = "plots/Wildlandfire_1d/"
        immpath = "data/Wildlandfire_1d/"
        immmpath = "data/Wildlandfire_1d/Original_data/"
        Q = np.load(immpath + "Q.npy")
        Q1 = np.load(immpath + "Q1.npy")
        Q2 = np.load(immpath + "Q2.npy")
        Q3 = np.load(immpath + "Q3.npy")
        T1Q1 = np.load(immpath + "T1Q1.npy")
        T2Q2 = np.load(immpath + "T2Q2.npy")
        T3Q3 = np.load(immpath + "T3Q3.npy")
        Q_tilde = np.load(immpath + "Q_tilde.npy")
        shifts1_true = np.load(immpath + "shifts1_true.npy")
        shifts1 = np.load(immpath + "shifts1.npy")
        t = np.load(immmpath + "Time.npy", allow_pickle=True)
        X = np.load(immmpath + "1D_grid.npy", allow_pickle=True)
        x = X[0]
        vmin = np.min(Q)
        vmax = np.max(Q)

        # Plot the separated frames
        fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True, sharex=True)
        # Original
        im0 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$Q$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1
        im1 = axs[1].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$(Q_z)^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 3
        im3 = axs[3].pcolormesh(T3Q3.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$(Q_z)^\theta_3$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        # Reconstructed
        im4 = axs[4].pcolormesh(Q_tilde.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[4].set_title(r"$\tilde{Q}$")
        axs[4].set_ylabel(r"$t$")
        axs[4].set_xlabel(r"$x$")
        axs[4].set_xticks([])
        axs[4].set_yticks([])
        fig.colorbar(im4, ax=axs[4], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti", dpi=300, transparent=True)
        plt.close(fig)

        # Plot the low-rank frames
        fig, axs = plt.subplots(1, 6, figsize=(24, 6), sharey=True, sharex=True)
        # Frame 1
        im0 = axs[0].pcolormesh(T1Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title(r"$(Q_z)^\theta_1$")
        axs[0].set_ylabel(r"$t$")
        axs[0].set_xlabel(r"$x$")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 1 (shifted)
        im1 = axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title(r"$Q^\theta_1$")
        axs[1].set_ylabel(r"$t$")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2
        im2 = axs[2].pcolormesh(T2Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].set_title(r"$(Q_z)^\theta_2$")
        axs[2].set_ylabel(r"$t$")
        axs[2].set_xlabel(r"$x$")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 2 (shifted)
        im3 = axs[3].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3].set_title(r"$Q^\theta_2$")
        axs[3].set_ylabel(r"$t$")
        axs[3].set_xlabel(r"$x$")
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        fig.colorbar(im3, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 3
        im4 = axs[4].pcolormesh(T3Q3.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[4].set_title(r"$(Q_z)^\theta_3$")
        axs[4].set_ylabel(r"$t$")
        axs[4].set_xlabel(r"$x$")
        axs[4].set_xticks([])
        axs[4].set_yticks([])
        fig.colorbar(im4, ax=axs[4], orientation="vertical", fraction=0.046, pad=0.04)

        # Frame 3 (shifted)
        im5 = axs[5].pcolormesh(Q3.T, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[5].set_title(r"$Q^\theta_3$")
        axs[5].set_ylabel(r"$t$")
        axs[5].set_xlabel(r"$x$")
        axs[5].set_xticks([])
        axs[5].set_yticks([])
        fig.colorbar(im5, ax=axs[5], orientation="vertical", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(impath + "Q_opti_low", dpi=300, transparent=True)
        plt.close(fig)

        rec_err = np.linalg.norm(Q - (T1Q1 + T2Q2 + T3Q3)) / np.linalg.norm(Q)
        print(f"RecErr: {rec_err}")

        # Plot the shifts
        val1 = - shifts1 + x[-1] // 2
        val2 = shifts1_true.copy()
        min_val = (val1 - val2).min()

        t_start = t[0]
        t_end = t[-1]
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(t, - shifts1 + x[-1] // 2, label="Predicted shift")
        axs.plot(t, shifts1_true + min_val, label="True shift")
        axs.set_ylabel(r"shift")
        axs.set_xlabel(r"$t$")
        axs.grid()
        axs.legend()
        fig.savefig(impath + "shift_comp" + ".pdf", format='pdf', dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)

    print("Execution pipeline completed successfully.")


if __name__ == "__main__":
    main()