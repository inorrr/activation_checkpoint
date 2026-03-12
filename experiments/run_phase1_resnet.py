import csv
import os
from typing import List, Dict, Any

from training.train import run_phase1_resnet

def summarize_trace(batch_size: int, trace: Dict[str, Any]) -> Dict[str, Any]:
    total_activation_bytes = sum(
        act["size_bytes"] for act in trace["activations"].values()
    )

    avg_forward_ms = (
        sum(op["duration_ms"] for op in trace["forward_ops"]) / len(trace["forward_ops"])
        if trace["forward_ops"] else 0.0
    )
    avg_backward_ms = (
        sum(op["duration_ms"] for op in trace["backward_ops"]) / len(trace["backward_ops"])
        if trace["backward_ops"] else 0.0
    )

    return {
        "batch_size": batch_size,
        "loss": trace["loss"],
        "iteration_time_ms": trace["iteration_time_ms"],
        "backward_only_time_ms": trace["backward_only_time_ms"],
        "peak_memory_allocated_mb": trace["peak_memory_allocated_mb"],
        "peak_memory_reserved_mb": trace["peak_memory_reserved_mb"],
        "num_forward_ops": len(trace["forward_ops"]),
        "num_backward_ops": len(trace["backward_ops"]),
        "num_activations": len(trace["activations"]),
        "total_activation_mb": total_activation_bytes / (1024 ** 2),
        "parameter_memory_mb": trace["parameter_memory_mb"],
        "gradient_memory_mb": trace["gradient_memory_mb"],
        "avg_forward_op_ms": avg_forward_ms,
        "avg_backward_op_ms": avg_backward_ms,
    }


def save_summary_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    batch_sizes = [1, 2, 4, 6, 8, 12, 16]
    summary_rows = []

    for bs in batch_sizes:
        print("=" * 80)
        print(f"Running Phase 1 profiling for batch size = {bs}")
        trace = run_phase1_resnet(
            batch_size=bs,
            save_dir="results/logs",
            save_trace=True,
        )
        summary = summarize_trace(bs, trace)
        summary_rows.append(summary)

    summary_path = "results/tables/resnet_phase1_summary.csv"
    save_summary_csv(summary_rows, summary_path)
    print("=" * 80)
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    main()