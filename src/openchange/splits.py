"""Provisional split contract. Region and time holdouts are immutable once accepted."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Placeholder identifiers only. Candidates are in reports/SPLIT_PROPOSAL.md (PROPOSED).
# Do not fill these with real regions or scene IDs until the spec is reviewed by a human.
PILOT_TRAIN_REGIONS = ("holdout-unset-train",)
PILOT_VALIDATION_REGIONS: tuple[str, ...] = ()
PILOT_TEST_REGIONS = ("holdout-unset-test",)
TEMPORAL_CUTOFF = "unset"
STATUS = "provisional"

STATUSES = ("provisional", "accepted")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitContract:
    train_regions: tuple[str, ...]
    test_regions: tuple[str, ...]
    temporal_cutoff: str
    validation_regions: tuple[str, ...] = ()
    status: str = "provisional"

    def validate(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        groups = {
            "train": set(self.train_regions),
            "validation": set(self.validation_regions),
            "test": set(self.test_regions),
        }
        names = list(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise ValueError(f"geographic leakage {left}/{right}: {sorted(overlap)}")
        if not self.train_regions or not self.test_regions:
            raise ValueError("train and test regions are required")
        if not self.temporal_cutoff:
            raise ValueError("temporal cutoff is required")
        if self.temporal_cutoff == "unset":
            if self.status == "accepted":
                raise ValueError("an accepted contract needs a real temporal cutoff")
            return
        self.cutoff_date()

    def cutoff_date(self) -> date | None:
        if self.temporal_cutoff == "unset":
            return None
        try:
            return date.fromisoformat(self.temporal_cutoff)
        except ValueError as exc:
            raise ValueError("temporal cutoff must be an ISO date (YYYY-MM-DD)") from exc

    @property
    def scoreable(self) -> bool:
        return self.status == "accepted" and self.cutoff_date() is not None

    def check_sample(self, split: str, region: str, observed: date) -> None:
        """Raise if a sample breaks the region or date rule for its split.

        Train and validation dates are on or before the cutoff; test dates are after it.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        cutoff = self.cutoff_date()
        if cutoff is None:
            raise ValueError("temporal cutoff is unset; no sample can be assigned")
        regions = {
            "train": self.train_regions,
            "validation": self.validation_regions,
            "test": self.test_regions,
        }[split]
        if region not in regions:
            raise ValueError(f"region {region!r} is not a {split} region")
        if split == "test" and observed <= cutoff:
            raise ValueError(f"temporal leakage: test date {observed} is not after {cutoff}")
        if split != "test" and observed > cutoff:
            raise ValueError(f"temporal leakage: {split} date {observed} is after {cutoff}")


def pilot_contract() -> SplitContract:
    return SplitContract(
        train_regions=PILOT_TRAIN_REGIONS,
        test_regions=PILOT_TEST_REGIONS,
        temporal_cutoff=TEMPORAL_CUTOFF,
        validation_regions=PILOT_VALIDATION_REGIONS,
        status=STATUS,
    )
