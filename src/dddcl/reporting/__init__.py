"""Reporting and result aggregation module for DDDCL."""

from .aggregate import (
    aggregate_r1_headline,
    aggregate_r2_ensemble,
    generate_reproduction_report,
)

__all__ = [
    "aggregate_r1_headline",
    "aggregate_r2_ensemble",
    "generate_reproduction_report",
]
