"""Provisional split contract. Region and time holdouts are immutable once published."""

from __future__ import annotations

from dataclasses import dataclass


# Placeholder identifiers only. Replace after the one-page spec is reviewed.
# Do not fill these with real scene IDs until the geography and cutoff are signed off.
PILOT_TRAIN_REGIONS = ("holdout-unset-train",)
PILOT_TEST_REGIONS = ("holdout-unset-test",)
TEMPORAL_CUTOFF = "unset"


@dataclass(frozen=True)
class SplitContract:
    train_regions: tuple[str, ...]
    test_regions: tuple[str, ...]
    temporal_cutoff: str

    def validate(self) -> None:
        overlap = set(self.train_regions) & set(self.test_regions)
        if overlap:
            raise ValueError(f"geographic leakage: {sorted(overlap)}")
        if not self.temporal_cutoff:
            raise ValueError("temporal cutoff is required")


def pilot_contract() -> SplitContract:
    return SplitContract(
        train_regions=PILOT_TRAIN_REGIONS,
        test_regions=PILOT_TEST_REGIONS,
        temporal_cutoff=TEMPORAL_CUTOFF,
    )
