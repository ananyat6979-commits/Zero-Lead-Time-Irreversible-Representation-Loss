from pathlib import Path
import json
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    depth = load("results/recovery_depth_sweep.json")

    recovery_iters = [d["recovery_iters"] for d in depth]
    js_recovered = [d["js_recovered"] for d in depth]
    js_control = [d["js_pristine_control"] for d in depth]
    js_minus_control = [d["js_recovered_minus_control"] for d in depth]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(recovery_iters, js_recovered, color="#e74c3c", marker="o", label="recovered")
    ax1.plot(recovery_iters, js_control, color="#2ecc71", marker="o", label="pristine control (matched count)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Recovery iterations (log scale)")
    ax1.set_ylabel("JS divergence to D0")
    ax1.set_title("Raw recovered vs. count-matched pristine control")
    ax1.legend(frameon=False)

    ax2.axhline(0, color="#888888", linewidth=1, linestyle="--")
    ax2.plot(recovery_iters, js_minus_control, color="#60A5FA", marker="o")
    ax2.set_xscale("log")
    ax2.set_xlabel("Recovery iterations (log scale)")
    ax2.set_ylabel("JS divergence (recovered - pristine control)")
    ax2.set_title("Recovery signal after removing training-count artifact")

    fig.suptitle(
        "Recovery depth sweep: no clear trend once training-count artifact is removed",
        y=1.03,
    )

    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/zl_irl_hysteresis_floor.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()