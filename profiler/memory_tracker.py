from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float


class MemoryTracker:
    """
    Utility for tracking CUDA memory usage.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"

    def reset_peak_stats(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> MemorySnapshot:
        if not self.enabled:
            return MemorySnapshot(0.0, 0.0, 0.0, 0.0)

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        return MemorySnapshot(
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            max_reserved_mb=max_reserved,
        )

    def summary_dict(self) -> Dict[str, float]:
        s = self.snapshot()
        return {
            "allocated_mb": s.allocated_mb,
            "reserved_mb": s.reserved_mb,
            "max_allocated_mb": s.max_allocated_mb,
            "max_reserved_mb": s.max_reserved_mb,
        }