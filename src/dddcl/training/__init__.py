"""
Training and optimization components for DDDCL.
Includes pace decisions, training loops, and checkpointing.
"""

from dddcl.training.pace import (
    PaceDecision,
    compute_pace_decision,
    update_lambda,
    should_reintroduce_hard,
)

__all__ = [
    "PaceDecision",
    "compute_pace_decision",
    "update_lambda",
    "should_reintroduce_hard",
]
