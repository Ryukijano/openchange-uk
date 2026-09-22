import unittest

from openchange.splits import SplitContract, pilot_contract


class SplitContractTest(unittest.TestCase):
    def test_pilot_regions_do_not_overlap(self) -> None:
        pilot_contract().validate()

    def test_shared_region_is_leakage(self) -> None:
        contract = SplitContract(
            train_regions=("east",),
            test_regions=("east",),
            temporal_cutoff="2024-01-01",
        )
        with self.assertRaises(ValueError):
            contract.validate()


if __name__ == "__main__":
    unittest.main()
