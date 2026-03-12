import os
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import load_resnet152
from profiler.graph_profiler import GraphProfiler


def make_dummy_batch(batch_size: int, num_classes: int, device: torch.device):
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def run_phase1_resnet(
    batch_size: int = 4,
    num_classes: int = 1000,
    save_dir: str = "results/logs",
    save_trace: bool = True,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet152(num_classes=num_classes).to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    profiler = GraphProfiler(model, device)
    profiler.attach()

    x, y = make_dummy_batch(batch_size, num_classes, device)
    trace = profiler.profile_one_iteration(
        batch_inputs=x,
        batch_targets=y,
        optimizer=optimizer,
        criterion=criterion,
    )

    profiler.detach()

    if save_trace:
        out_path = os.path.join(save_dir, f"resnet152_bs{batch_size}_phase1_trace.json")
        profiler.save_trace(trace, out_path)
        print(f"Saved trace to: {out_path}")

    print(f"Loss: {trace['loss']:.4f}")
    print(f"Iteration time (ms): {trace['iteration_time_ms']:.2f}")
    print(f"Peak allocated memory (MB): {trace['peak_memory_allocated_mb']:.2f}")
    print(f"Peak reserved memory (MB): {trace['peak_memory_reserved_mb']:.2f}")
    print(f"Forward ops recorded: {len(trace['forward_ops'])}")
    print(f"Backward ops recorded: {len(trace['backward_ops'])}")
    print(f"Tracked activations: {len(trace['activations'])}")

    return trace


if __name__ == "__main__":
    run_phase1_resnet(batch_size=4)