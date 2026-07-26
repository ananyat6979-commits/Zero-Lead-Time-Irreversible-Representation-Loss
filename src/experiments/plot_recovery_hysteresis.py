# NOTE (flagged, not fixed): verified directly this session that
# results/recovery_depth_sweep.json is a genuine no-op, config.iterations
# is never consumed anywhere inside run_recovery. All six recovery_iters
# values produce bit identical output, js_recovered equal to
# 0.027955425229532538 in every row. The resulting figure this script
# produces shows a completely flat red line, which visually looks like
# convergence to a hysteresis floor despite more recovery effort, but is
# actually just the same fixed number plotted six times. This is not
# evidence of a hard hysteresis floor, it is an artifact of the sweep
# never varying anything. Do not cite the generated figure
# figures/zl_irl_hysteresis_floor.png as a real finding until
# run_recovery_depth_sweep.py's no-op is resolved with a real design
# decision on what recovery iterations should mean.

from pathlib import Path
import json
import matplotlib.pyplot as plt


plt.style.use("dark_background")

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    phase35 = load("results/phase35_diagnostics.json")
    depth = load("results/recovery_depth_sweep.json")

    js_clean = [d["js_divergence"] for d in phase35]
    js_recovery = [d["js_recovered"] for d in depth]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhspan(
        min(js_clean),
        max(js_clean),
        color="#2ecc71",
        alpha=0.2,
        label="Fully reversible regime (Phase 3.5)"
    )

    ax.plot(
        [d["recovery_iters"] for d in depth],
        js_recovery,
        color="#e74c3c",
        marker="o",
        label="Recovered after self-training"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Recovery iterations (log scale)")
    ax.set_ylabel("JS divergence to D₀")

    ax.set_title("Recovery exhibits a hard hysteresis floor")
    ax.legend(frameon=False)

    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/zl_irl_hysteresis_floor.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
