import csv
import os
from typing import List

import matplotlib.pyplot as plt


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    input_csv = "results/tables/resnet_phase1_summary.csv"
    output_png = "results/figures/resnet_phase1_latency_vs_batch.png"

    rows = read_csv(input_csv)
    batch_sizes = [int(r["batch_size"]) for r in rows]
    iteration_time = [float(r["iteration_time_ms"]) for r in rows]

    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, iteration_time, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Iteration Time (ms)")
    plt.title("ResNet-152 Phase 1: Iteration Latency vs Batch Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()

    print(f"Saved figure to: {output_png}")


if __name__ == "__main__":
    main()