import csv
import os
import sys
from typing import List

import matplotlib.pyplot as plt


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(model):
    if model == "resnet":
        input_csv = "results/tables/resnet_phase1_summary.csv"
        output_png = "results/figures/resnet_phase1_memory_breakdown_stacked.png"
        title = "ResNet-152 Phase 1: Memory Breakdown by Batch Size"
    else:
        input_csv = "results/tables/bert_phase1_summary.csv"
        output_png = "results/figures/bert_phase1_memory_breakdown_stacked.png"
        title = "BERT Phase 1: Memory Breakdown by Batch Size"

    rows = read_csv(input_csv)

    # sort by batch size just in case
    rows = sorted(rows, key=lambda r: int(r["batch_size"]))

    batch_sizes = [int(r["batch_size"]) for r in rows]
    parameter_mb = [float(r["parameter_memory_mb"]) for r in rows]
    gradient_mb = [float(r["gradient_memory_mb"]) for r in rows]
    activation_mb = [float(r["total_activation_mb"]) for r in rows]

    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.bar(batch_sizes, parameter_mb, label="Parameters")
    plt.bar(
        batch_sizes,
        gradient_mb,
        bottom=parameter_mb,
        label="Gradients",
    )
    plt.bar(
        batch_sizes,
        activation_mb,
        bottom=[p + g for p, g in zip(parameter_mb, gradient_mb)],
        label="Activations",
    )

    plt.xlabel("Batch Size")
    plt.ylabel("Memory (MB)")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()

    print(f"Saved figure to: {output_png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [resnet|bert]")
        sys.exit(1)

    model = sys.argv[1]
    main(model)