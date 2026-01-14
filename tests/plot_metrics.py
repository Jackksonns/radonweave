# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import argparse
import json
import os


def _default_out_path(metrics_path: str) -> str:
    parts = os.path.normpath(metrics_path).split(os.sep)
    model = "model"
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            name = parts[idx + 1]
            model = name.split("_stop")[0]
    return os.path.join("plot", model, "val_loss.png")


def main():
    parser = argparse.ArgumentParser(description="Plot val_loss curve from metrics_history.json")
    parser.add_argument("--metrics", required=True, help="Path to metrics_history.json")
    parser.add_argument("--out", default=None, help="Output path for val_loss.png")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    out_path = args.out or _default_out_path(args.metrics)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required: {exc}")

    with open(args.metrics, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = [d["epoch"] for d in data]
    val_loss = [d["val_loss"] for d in data]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(epochs, val_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title(args.title or "Val Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
