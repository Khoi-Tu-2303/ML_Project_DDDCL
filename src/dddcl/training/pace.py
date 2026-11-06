"""
Pure functional Self-Paced Learning (SPL) pace decision logic.
Implements Decision D07, D08 & Assumption A06, A07.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class PaceDecision:
    """
    Immutable container representing the SPL decision for batches in an epoch.
    Pure data structure, fully decoupled from PyTorch optimizer state.
    """
    easy_batch_ids: List[int]
    hard_batch_ids: List[int]
    ordered_easy_batches: List[int]
    ordered_hard_batches: List[int]
    fallback_reason: Optional[str]
    lambda_used: float

    @property
    def total_batches(self) -> int:
        return len(self.easy_batch_ids) + len(self.hard_batch_ids)

    @property
    def easy_ratio(self) -> float:
        tot = self.total_batches
        return len(self.easy_batch_ids) / tot if tot > 0 else 0.0


def compute_pace_decision(
    batch_losses: List[float],
    lambda_val: float,
    empty_easy_fallback: str = "min_loss_batch",
) -> PaceDecision:
    """
    Evaluate error loss L_b for each batch b against threshold lambda:
    - Easy: L_b < lambda
    - Hard: L_b >= lambda (equality belongs to Hard)
    
    If Easy is empty:
    - Fallback: Select the single batch with the lowest error loss as Easy.
    
    Returns PaceDecision with easy and hard batches sorted from lowest to highest loss.
    """
    if not batch_losses:
        return PaceDecision(
            easy_batch_ids=[],
            hard_batch_ids=[],
            ordered_easy_batches=[],
            ordered_hard_batches=[],
            fallback_reason="no_batches",
            lambda_used=lambda_val,
        )

    easy_pairs: List[Tuple[int, float]] = []
    hard_pairs: List[Tuple[int, float]] = []

    for b_id, loss in enumerate(batch_losses):
        loss_val = float(loss)
        if loss_val < lambda_val:
            easy_pairs.append((b_id, loss_val))
        else:
            # Equality strictly belongs to hard
            hard_pairs.append((b_id, loss_val))

    fallback_reason: Optional[str] = None

    # Handle empty easy set
    if not easy_pairs:
        if empty_easy_fallback == "min_loss_batch":
            # Pick batch with minimum loss (stable tie-break by b_id)
            hard_pairs.sort(key=lambda item: (item[1], item[0]))
            min_pair = hard_pairs.pop(0)
            easy_pairs.append(min_pair)
            fallback_reason = "empty_easy_picked_min_loss"
        else:
            fallback_reason = "empty_easy_no_fallback"

    # Sort easy batches from lowest to highest loss (stable tie-break by batch_id)
    easy_pairs.sort(key=lambda item: (item[1], item[0]))
    ordered_easy = [item[0] for item in easy_pairs]
    easy_ids = [item[0] for item in easy_pairs]

    # Sort hard batches from lowest to highest loss (stable tie-break by batch_id)
    hard_pairs.sort(key=lambda item: (item[1], item[0]))
    ordered_hard = [item[0] for item in hard_pairs]
    hard_ids = [item[0] for item in hard_pairs]

    return PaceDecision(
        easy_batch_ids=easy_ids,
        hard_batch_ids=hard_ids,
        ordered_easy_batches=ordered_easy,
        ordered_hard_batches=ordered_hard,
        fallback_reason=fallback_reason,
        lambda_used=lambda_val,
    )


def update_lambda(
    current_lambda: float,
    lambda_max: float = 0.6,
    growth: float = 1.1,
) -> float:
    """
    Update SPL threshold lambda for the next epoch:
    lambda_{e+1} = min(lambda_max, current_lambda * growth)
    """
    return float(min(lambda_max, current_lambda * growth))


def should_reintroduce_hard(mean_easy_loss: float, lambda_val: float) -> bool:
    """
    Algorithm 2 line 11 condition:
    If model's loss on easy data is below lambda, re-add hard batches.
    """
    return bool(mean_easy_loss < lambda_val)

