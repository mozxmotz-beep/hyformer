"""Dataset utilities and schema definitions for PCVRHyFormer.

This module intentionally keeps the public API used by ``train.py`` stable:

* ``FeatureSchema``
* ``NUM_TIME_BUCKETS``
* ``get_pcvr_data``

The previous version of this file accidentally contained model code, which made
``from dataset import FeatureSchema`` fail during startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# Time-bucket config used by the model. Keep this importable constant stable.
BUCKET_BOUNDARIES: Tuple[int, ...] = (60, 300, 1800, 3600, 6 * 3600, 24 * 3600, 3 * 24 * 3600)
NUM_TIME_BUCKETS: int = len(BUCKET_BOUNDARIES) + 1


@dataclass(frozen=True)
class FeatureSchema:
    """Schema for a feature block.

    Attributes:
        entries: list of ``(feature_id, offset, length)`` tuples.
        total_dim: flattened feature width for this block.
    """

    entries: List[Tuple[str, int, int]]
    total_dim: int


@dataclass(frozen=True)
class PCVRDatasetMeta:
    """Metadata object consumed by ``train.py`` for model construction."""

    user_int_schema: FeatureSchema
    item_int_schema: FeatureSchema
    user_dense_schema: FeatureSchema
    item_dense_schema: FeatureSchema
    user_int_vocab_sizes: Sequence[int]
    item_int_vocab_sizes: Sequence[int]
    seq_domain_vocab_sizes: dict
    seq_domains: Sequence[str]


def get_pcvr_data(*args, **kwargs):
    """Load training/validation dataloaders and dataset metadata.

    This project version does not include the original Parquet data pipeline.
    The function remains as a compatibility hook and raises a clear error so
    runtime failures are explicit and actionable.
    """

    raise NotImplementedError(
        "get_pcvr_data is not implemented in this repository snapshot. "
        "Please restore the dataset loader implementation that builds "
        "(train_loader, valid_loader, pcvr_dataset)."
    )
