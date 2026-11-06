"""
Curriculum learning module for DDDCL.
Includes geometry, KDE level partition, SMOTE resampling, and curriculum scoring strategies.
"""

from dddcl.curriculum.geometry import (
    GeometryResult,
    compute_centroids,
    compute_class_distances,
    compute_geometry,
)
from dddcl.curriculum.partition import (
    PartitionResult,
    kde_level_partition_v1,
    partition_by_class,
)
from dddcl.curriculum.resample import (
    ResampleResult,
    resample_curriculum_bins,
)
from dddcl.curriculum.scoring import (
    CurriculumResult,
    compute_policy_hash,
    create_curriculum,
)

__all__ = [
    "GeometryResult",
    "compute_centroids",
    "compute_class_distances",
    "compute_geometry",
    "PartitionResult",
    "kde_level_partition_v1",
    "partition_by_class",
    "ResampleResult",
    "resample_curriculum_bins",
    "CurriculumResult",
    "compute_policy_hash",
    "create_curriculum",
]

