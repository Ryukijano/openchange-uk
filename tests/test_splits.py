import unittest
from datetime import date

from openchange.splits import SplitContract, pilot_contract


def contract(**overrides):
    values = dict(
        train_regions=("a", "b"),
        validation_regions=("v",),
        test_regions=("t",),
        temporal_cutoff="2021-12-31",
        status="provisional",
    )
    values.update(overrides)
    return SplitContract(**values)


class SplitContractTest(unittest.TestCase):
    def test_pilot_regions_do_not_overlap(self) -> None:
        pilot_contract().validate()

    def test_pilot_is_not_scoreable(self) -> None:
        self.assertFalse(pilot_contract().scoreable)

    def test_shared_region_is_leakage(self) -> None:
        contract_ = SplitContract(
            train_regions=("east",),
            test_regions=("east",),
            temporal_cutoff="2024-01-01",
        )
        with self.assertRaises(ValueError):
            contract_.validate()

    def test_validation_overlap_is_leakage(self) -> None:
        with self.assertRaises(ValueError):
            contract(validation_regions=("t",)).validate()
        with self.assertRaises(ValueError):
            contract(validation_regions=("a",)).validate()

    def test_cutoff_rules(self) -> None:
        with self.assertRaises(ValueError):
            contract(temporal_cutoff="unset", status="accepted").validate()
        with self.assertRaises(ValueError):
            contract(temporal_cutoff="end of 2021").validate()
        with self.assertRaises(ValueError):
            contract(status="frozen").validate()
        self.assertTrue(contract(status="accepted").scoreable)

    def test_check_sample(self) -> None:
        c = contract()
        c.check_sample("train", "a", date(2021, 12, 31))
        c.check_sample("validation", "v", date(2020, 1, 1))
        c.check_sample("test", "t", date(2022, 1, 1))
        with self.assertRaises(ValueError):
            c.check_sample("test", "t", date(2021, 12, 31))
        with self.assertRaises(ValueError):
            c.check_sample("train", "a", date(2022, 1, 1))
        with self.assertRaises(ValueError):
            c.check_sample("train", "t", date(2020, 1, 1))
        with self.assertRaises(ValueError):
            pilot_contract().check_sample("train", "holdout-unset-train", date(2020, 1, 1))


if __name__ == "__main__":
    unittest.main()
