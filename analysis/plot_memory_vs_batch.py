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
        output_png = "results/figures/resnet_phase1_memory_vs_batch.png"
        title = "ResNet-152 Phase 1: Peak Memory vs Batch Size"
    else:
        input_csv = "results/tables/bert_phase1_summary.csv"
        output_png = "results/figures/bert_phase1_memory_vs_batch.png"
        title = "BERT Phase 1: Peak Memory vs Batch Size"

    rows = read_csv(input_csv)
    batch_sizes = [int(r["batch_size"]) for r in rows]
    peak_memory = [float(r["peak_memory_allocated_mb"]) for r in rows]

    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, peak_memory, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Peak Allocated GPU Memory (MB)")

    plt.title(title)
    plt.grid(True)
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