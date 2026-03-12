from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class OpRecord:
    name: str
    op_type: str                  # forward, backward, optimizer
    module_type: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    input_bytes: int
    output_bytes: int
    input_category: str
    output_category: str
    memory_allocated_mb: float
    memory_reserved_mb: float
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)