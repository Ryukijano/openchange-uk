from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("install .venv-jepa with CPU torch")

import numpy as np
import yaml

from openchange.jepa.data import (
    S1_BANDS,
    S2_BANDS,
    BalancedStepSampler,
    LocalObservationDataset,
    ManifestError,
)
from openchange.sources import DEFAULT_PATH, SourceNotApproved, load_sources


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PilotFixture:
    """Only temporary synthetic pixels and temporary, explicit imagery approvals."""

    def __init__(self, root: Path, count: int = 16):
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.sources_path = root / "sources.yaml"
        self.sources = {"approved_public_sources": [{
            "id": "fixture-imagery", "url": "https://example.org/fixture",
            "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Synthetic test fixtures", "expected_size_bytes": 1_000_000,
            "use": "imagery", "splits": ["pilot"], "redistribution": "manifest-only",
            "approved_by": "test-only-reviewer", "approved_on": "2026-01-01",
        }]}
        self.save_sources()
        self.split = {
            "schema_version": 1, "status": "accepted", "immutable": True,
            "accepted_by": "test-only-reviewer", "accepted_on": "2026-01-01",
            "temporal_cutoff": "2025-12-31T23:59:59Z",
            "geographic_policy": {"method": "footprint_buffer", "buffer_m": 1000,
                                  "audit_sha256": "a" * 64},
            "locations": [{"location_id": f"location-{i}", "geographic_group": "train-group",
                           "role": "train"} for i in range(count)] + [
                {"location_id": "validation-location", "geographic_group": "validation-group",
                 "role": "validation"},
                {"location_id": "test-location", "geographic_group": "test-group", "role": "test"},
            ],
        }
        self.normalization = {
            "schema_version": 1, "id": "fixture-v1", "origin": "provider", "split_sha256": None,
            "s1": {"bands": list(S1_BANDS), "units": "dB", "mean": [-8.0] * 2, "std": [2.0] * 2},
            "s2": {"bands": list(S2_BANDS), "units": "reflectance", "mean": [0.2] * 12,
                   "std": [0.2] * 12},
        }
        self.manifest = {
            "schema_version": 1, "stage": "pilot", "role": "train",
            "source_ids": ["fixture-imagery"], "split": {}, "normalization": {},
            "thresholds": {"clear_max": 0.0, "degraded_min": 0.5},
            "max_pair_gap_seconds": 7 * 24 * 3600,
            "mask_semantics": "true_is_valid_false_is_cloud", "records": [],
        }
        self.save_split()
        self.save_normalization()
        for i in range(count):
            mask = np.ones((4, 4), dtype=np.bool_)
            if i >= count // 2:
                mask[:2] = False
            record = {
                "sample_id": 1000 + i, "location_id": f"location-{i}",
                "geographic_group": "train-group", "role": "train",
                "s1_timestamp": "2025-06-01T10:00:00Z", "s2_timestamp": "2025-06-03T10:00:00Z",
                "cloud_fraction": float(1 - mask.mean()),
                "footprint": {"crs": "EPSG:4326", "bounds": [20.0, 30.0, 20.1, 30.1]},
                "grid": {"crs": "EPSG:32634", "shape": [4, 4], "pixel_size_m": 10,
                         "resampling": "bilinear-once", "nodata_policy": "excluded"},
            }
            self.manifest["records"].append(record)
            for sensor, array in (
                ("s1", np.full((2, 4, 4), -10, dtype=np.float32)),
                ("s2", np.full((12, 4, 4), 0.4, dtype=np.float32)), ("s2_valid", mask),
            ):
                self.save_array(i, sensor, array)
        self.save_manifest()

    def save_sources(self):
        self.sources_path.write_text(yaml.safe_dump(self.sources))

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def save_split(self):
        path = self.root / "split.json"
        path.write_text(json.dumps(self.split))
        self.manifest["split"] = {"path": path.name, "sha256": _hash(path)}
        self.save_manifest()

    def save_normalization(self):
        path = self.root / "normalization.json"
        path.write_text(json.dumps(self.normalization))
        self.manifest["normalization"] = {"path": path.name, "sha256": _hash(path)}
        self.save_manifest()

    def save_array(self, index, sensor, array):
        path = self.root / f"{index}-{sensor}.npy"
        np.save(path, array, allow_pickle=array.dtype == object)
        self.manifest["records"][index][sensor] = {
            "path": path.name, "sha256": _hash(path), "source_id": "fixture-imagery", "revision": "b" * 40,
        }
        self.save_manifest()

    def dataset(self):
        return LocalObservationDataset(self.manifest_path, self.sources_path)


class LocalDatasetTest(unittest.TestCase):
    def test_revision_matches_immutable_approved_source_url(self):
        fixture = self.fixture()
        fixture.sources["approved_public_sources"][0]["url"] = "https://huggingface.co/datasets/example/fixture/resolve/" + "c" * 40 + "/s1.tar"
        fixture.save_sources()
        with patch("numpy.load") as load:
            with self.assertRaisesRegex(ManifestError, "revision differs"):
                fixture.dataset()
            load.assert_not_called()

    def fixture(self, count=16):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return PilotFixture(Path(temporary.name), count)

    def before_arrays(self, fixture, exception=ManifestError):
        with patch("openchange.jepa.data.np.load") as load:
            with self.assertRaises(exception):
                fixture.dataset()
            load.assert_not_called()

    def test_explicit_approved_local_arrays_and_provenance(self):
        fixture = self.fixture()
        with patch("openchange.jepa.data.require_approved", return_value=SimpleNamespace(use="imagery")) as gate:
            dataset = fixture.dataset()
        gate.assert_called_once_with("fixture-imagery", "pilot", path=fixture.sources_path)
        self.assertEqual(len(dataset), 16)
        self.assertEqual(dataset.eligible_indices(0), tuple(range(8)))
        self.assertEqual(dataset.eligible_indices(1), tuple(range(8, 16)))
        self.assertEqual(dataset.eligible_indices(2), tuple(range(16)))
        self.assertEqual(dataset.eligible_indices(3), tuple(range(16)))
        self.assertEqual(dataset.manifest_sha256, _hash(fixture.manifest_path))
        self.assertEqual(dataset.split_sha256, fixture.manifest["split"]["sha256"])
        self.assertEqual(dataset.normalization_sha256, fixture.manifest["normalization"]["sha256"])
        metadata = dataset.metadata(0)
        metadata["s1"]["revision"] = "changed"
        self.assertEqual(dataset.metadata(0), fixture.manifest["records"][0])

    def test_normalization_and_invalid_optical_pixels_zeroed(self):
        dataset = self.fixture().dataset()
        clear = dataset[0, 0]
        np.testing.assert_array_equal(clear.s1, -np.ones((2, 4, 4), dtype=np.float32))
        np.testing.assert_array_equal(clear.s2, np.ones((12, 4, 4), dtype=np.float32))
        degraded = dataset[8, 1]
        np.testing.assert_array_equal(degraded.s2[:, :2], np.zeros((12, 2, 4), dtype=np.float32))
        np.testing.assert_array_equal(degraded.s2[:, 2:], np.ones((12, 2, 4), dtype=np.float32))
        self.assertEqual(degraded.mask_fraction, 0.5)
        self.assertFalse(degraded.synthetic_data)
        self.assertFalse(degraded.synthetic_condition)
        self.assertEqual(degraded.intervention, "none")

    def test_withheld_sensors_are_absent_not_invented(self):
        dataset = self.fixture().dataset()
        s1_only = dataset[8, 2]
        s2_only = dataset[8, 3]
        self.assertIsNone(s1_only.s2)
        self.assertIsNone(s1_only.s2_valid)
        self.assertIsNone(s2_only.s1)
        self.assertEqual(s1_only.intervention, "withhold_s2")
        self.assertEqual(s2_only.intervention, "withhold_s1")
        self.assertTrue(s1_only.synthetic_condition)
        self.assertTrue(s2_only.synthetic_condition)
        self.assertFalse(s1_only.synthetic_data)
        self.assertEqual(s1_only.sample_id, s2_only.sample_id)
        np.testing.assert_array_equal(s2_only.s2[:, :2], 0.0)

    def test_samples_do_not_mutate_source_arrays_or_each_other(self):
        fixture = self.fixture()
        dataset = fixture.dataset()
        first = dataset[0, 0]
        first.s1[:] = 999
        first.s2[:] = 999
        first.s2_valid[:] = False
        second = dataset[0, 0]
        np.testing.assert_array_equal(second.s1, -1.0)
        self.assertTrue(second.s2_valid.all())
        np.testing.assert_array_equal(np.load(fixture.root / "0-s1.npy", allow_pickle=False), -10.0)

    def test_numpy_loading_is_readonly_and_disallows_pickle(self):
        fixture = self.fixture()
        original = np.load
        with patch("openchange.jepa.data.np.load", wraps=original) as load:
            dataset = fixture.dataset()
            dataset[0, 0]
        for call in load.call_args_list:
            self.assertEqual(call.kwargs, {"allow_pickle": False, "mmap_mode": "r"})

    def test_auxiliary_approval_never_authorizes_arrays(self):
        fixture = self.fixture()
        fixture.sources["approved_public_sources"][0]["use"] = "auxiliary"
        fixture.save_sources()
        self.before_arrays(fixture, SourceNotApproved)

    def test_actual_repo_metadata_sources_do_not_authorize_arrays(self):
        fixture = self.fixture()
        sources = load_sources(DEFAULT_PATH)
        fixture.manifest["source_ids"] = [source.id for source in sources]
        fixture.save_manifest()
        fixture.sources_path = DEFAULT_PATH
        self.before_arrays(fixture, SourceNotApproved)

    def test_all_declared_sources_checked_before_any_array(self):
        fixture = self.fixture()
        fixture.manifest["source_ids"].append("unapproved-other")
        fixture.save_manifest()
        self.before_arrays(fixture, SourceNotApproved)

    def test_missing_or_nonpilot_approval_is_refused(self):
        for approved in ([], ["train"]):
            with self.subTest(approved=approved):
                fixture = self.fixture(4)
                if approved:
                    fixture.sources["approved_public_sources"][0]["splits"] = approved
                else:
                    fixture.sources["approved_public_sources"] = []
                fixture.save_sources()
                self.before_arrays(fixture, SourceNotApproved)

    def test_unknown_payload_source_refused_before_first_array(self):
        fixture = self.fixture()
        fixture.manifest["records"][-1]["s1"]["source_id"] = "undeclared-source"
        fixture.save_manifest()
        self.before_arrays(fixture)

    def test_manifest_and_records_train_only(self):
        for target, role in (("manifest", "validation"), ("manifest", "test"), ("record", "validation"),
                             ("record", "test")):
            with self.subTest(target=target, role=role):
                fixture = self.fixture(4)
                mapping = fixture.manifest if target == "manifest" else fixture.manifest["records"][-1]
                mapping["role"] = role
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_unaccepted_or_mutable_or_unsigned_split_refused(self):
        for key, value in (("status", "proposed"), ("immutable", False), ("accepted_by", "TBD"),
                           ("accepted_on", "2099-01-01")):
            with self.subTest(key=key):
                fixture = self.fixture(4)
                fixture.split[key] = value
                fixture.save_split()
                self.before_arrays(fixture)

    def test_split_hash_and_normalization_hash_checked_before_arrays(self):
        for sidecar in ("split", "normalization"):
            with self.subTest(sidecar=sidecar):
                fixture = self.fixture(4)
                fixture.manifest[sidecar]["sha256"] = "0" * 64
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_geographic_and_location_role_leakage_refused(self):
        for change in ("group", "location", "heldout", "duplicate", "no_test"):
            with self.subTest(change=change):
                fixture = self.fixture(4)
                if change == "group":
                    fixture.split["locations"][-1]["geographic_group"] = "train-group"
                elif change == "location":
                    fixture.manifest["records"][0]["geographic_group"] = "wrong-group"
                elif change == "heldout":
                    fixture.manifest["records"][0]["location_id"] = "test-location"
                elif change == "duplicate":
                    fixture.split["locations"].append(copy.deepcopy(fixture.split["locations"][0]))
                else:
                    fixture.split["locations"].pop()
                fixture.save_split()
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_one_pair_per_location_and_unique_sample_ids(self):
        for field in ("location_id", "sample_id"):
            with self.subTest(field=field):
                fixture = self.fixture(4)
                fixture.manifest["records"][-1][field] = fixture.manifest["records"][0][field]
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_pilot_item_and_location_caps(self):
        fixture = self.fixture(4)
        fixture.manifest["records"] = [fixture.manifest["records"][0]] * 501
        fixture.save_manifest()
        self.before_arrays(fixture)
        fixture = self.fixture(4)
        fixture.split["locations"] = [fixture.split["locations"][0]] * 501
        fixture.save_split()
        self.before_arrays(fixture)

    def test_pair_gap_boundaries_timezone_and_cutoff(self):
        fixture = self.fixture(4)
        fixture.manifest["max_pair_gap_seconds"] = 2 * 24 * 3600
        fixture.save_manifest()
        self.assertEqual(len(fixture.dataset()), 4)
        for stamp in ("2025-06-03T10:00:01Z", "2025-06-03T10:00:00", "2026-01-01T00:00:00Z", "bad"):
            with self.subTest(stamp=stamp):
                fixture.manifest["records"][0]["s2_timestamp"] = stamp
                fixture.save_manifest()
                self.before_arrays(fixture)
        fixture.manifest["max_pair_gap_seconds"] = 10**9
        fixture.manifest["records"][0]["s2_timestamp"] = "2026-01-01T00:00:00Z"
        fixture.save_manifest()
        self.before_arrays(fixture)

    def test_cloud_thresholds_are_disjoint_and_grey_zone_only_withheld(self):
        fixture = self.fixture(4)
        mask = np.ones((4, 4), dtype=np.bool_)
        mask[:1] = False
        fixture.save_array(0, "s2_valid", mask)
        fixture.manifest["records"][0]["cloud_fraction"] = 0.25
        fixture.save_manifest()
        dataset = fixture.dataset()
        self.assertNotIn(0, dataset.eligible_indices(0))
        self.assertNotIn(0, dataset.eligible_indices(1))
        self.assertIn(0, dataset.eligible_indices(2))
        self.assertIn(0, dataset.eligible_indices(3))
        with self.assertRaises(ManifestError):
            dataset[0, 0]
        for clear, degraded in ((0.5, 0.5), (0.7, 0.5), (-0.1, 0.5), (0.0, 1.1), (False, 0.5)):
            with self.subTest(clear=clear, degraded=degraded):
                fixture.manifest["thresholds"] = {"clear_max": clear, "degraded_min": degraded}
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_cloud_fraction_must_match_real_mask_and_no_synthetic_occlusion(self):
        fixture = self.fixture(4)
        fixture.manifest["records"][0]["cloud_fraction"] = 0.5
        fixture.save_manifest()
        with self.assertRaisesRegex(ManifestError, "cloud_fraction"):
            fixture.dataset()
        fixture.manifest["records"][0]["cloud_fraction"] = 0.0
        fixture.manifest["mask_semantics"] = "synthetic_occlusion"
        fixture.save_manifest()
        self.before_arrays(fixture)

    def test_payload_checksum_checked_before_numpy_load(self):
        fixture = self.fixture(4)
        fixture.manifest["records"][0]["s1"]["sha256"] = "0" * 64
        fixture.save_manifest()
        self.before_arrays(fixture)

    def test_missing_payloads_refused_even_when_sensor_would_be_withheld(self):
        fixture = self.fixture(4)
        dataset = fixture.dataset()
        (fixture.root / "0-s2.npy").unlink()
        with self.assertRaises(ManifestError):
            dataset[0, 2]
        self.before_arrays(fixture)

    def test_changed_payloads_refused_after_construction(self):
        fixture = self.fixture(4)
        dataset = fixture.dataset()
        fixture.save_array(0, "s1", np.zeros((2, 4, 4), dtype=np.float32))
        with patch("openchange.jepa.data.np.load") as load:
            with self.assertRaises(ManifestError):
                dataset[0, 0]
            load.assert_not_called()

    def test_path_traversal_absolute_and_windows_paths_refused(self):
        for path in ("../0-s1.npy", "nested/../../0-s1.npy", "/tmp/0-s1.npy", "C:\\x.npy",
                     "nested\\0-s1.npy", "./0-s1.npy", "https://example.org/x.npy", "0-s1.npz"):
            with self.subTest(path=path):
                fixture = self.fixture(4)
                fixture.manifest["records"][0]["s1"]["path"] = path
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_symlink_escape_and_symlink_parent_refused(self):
        for directory in (False, True):
            with self.subTest(directory=directory):
                fixture = self.fixture(4)
                outside = self.fixture(4)
                target = outside.root if directory else outside.root / "0-s1.npy"
                link = fixture.root / ("linked" if directory else "linked.npy")
                link.symlink_to(target, target_is_directory=directory)
                fixture.manifest["records"][0]["s1"]["path"] = "linked/0-s1.npy" if directory else "linked.npy"
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_symlink_replacement_after_validation_is_refused(self):
        fixture = self.fixture(4)
        dataset = fixture.dataset()
        path = fixture.root / "0-s1.npy"
        path.unlink()
        path.symlink_to(fixture.root / "1-s1.npy")
        with self.assertRaises(ManifestError):
            dataset[0, 0]

    def test_dtype_shape_finite_mask_and_pickle_rejections(self):
        cases = (
            ("s1", np.zeros((2, 4, 4), dtype=np.float64)),
            ("s1", np.zeros((1, 4, 4), dtype=np.float32)),
            ("s1", np.full((2, 4, 4), np.nan, dtype=np.float32)),
            ("s2", np.full((12, 4, 4), np.inf, dtype=np.float32)),
            ("s2", np.zeros((12, 5, 4), dtype=np.float32)),
            ("s2_valid", np.ones((4, 4), dtype=np.uint8)),
            ("s2_valid", np.ones((4, 3), dtype=np.bool_)),
            ("s1", np.array([{"payload": "must-not-unpickle"}], dtype=object)),
        )
        for sensor, array in cases:
            with self.subTest(sensor=sensor, dtype=array.dtype, shape=array.shape):
                fixture = self.fixture(4)
                fixture.save_array(0, sensor, array)
                with self.assertRaises(ManifestError):
                    fixture.dataset()

    def test_normalization_schema_and_train_only_hash(self):
        for field, value in (("std", [0.0] * 2), ("std", [-1.0] * 2), ("std", [1e-50] * 2),
                             ("mean", [float("nan")] * 2), ("mean", [1e100] * 2),
                             ("bands", ["VH", "VV"]), ("units", "linear_power")):
            with self.subTest(field=field, value=value):
                fixture = self.fixture(4)
                fixture.normalization["s1"][field] = value
                fixture.save_normalization()
                self.before_arrays(fixture)
        fixture = self.fixture(4)
        fixture.normalization["origin"] = "heldout"
        fixture.save_normalization()
        self.before_arrays(fixture)
        fixture.normalization["origin"] = "train_only"
        fixture.normalization["split_sha256"] = "f" * 64
        fixture.save_normalization()
        self.before_arrays(fixture)
        fixture.normalization["split_sha256"] = fixture.manifest["split"]["sha256"]
        fixture.save_normalization()
        self.assertEqual(len(fixture.dataset()), 4)

    def test_normalization_overflow_refused(self):
        fixture = self.fixture(4)
        fixture.normalization["s1"]["std"] = [1e-40] * 2
        fixture.save_normalization()
        with self.assertRaisesRegex(ManifestError, "nonfinite"):
            fixture.dataset()

    def test_payload_revision_grid_and_footprint_validated(self):
        for category in ("revision", "grid", "footprint", "shape", "nodata"):
            with self.subTest(category=category):
                fixture = self.fixture(4)
                record = fixture.manifest["records"][-1]
                if category == "revision":
                    record["s1"]["revision"] = "main"
                elif category == "grid":
                    record["grid"]["pixel_size_m"] = 20
                elif category == "footprint":
                    record["footprint"]["crs"] = "EPSG:32634"
                elif category == "shape":
                    record["grid"]["shape"] = [0, 4]
                else:
                    record["grid"]["nodata_policy"] = "interpolated"
                fixture.save_manifest()
                self.before_arrays(fixture)

    def test_unknown_missing_duplicate_keys_and_invalid_version_refused(self):
        for category in ("extra", "missing", "duplicate", "version", "bool_version", "record_extra"):
            with self.subTest(category=category):
                fixture = self.fixture(4)
                if category == "extra":
                    fixture.manifest["synthetic_occlusion"] = True
                elif category == "missing":
                    del fixture.manifest["thresholds"]
                elif category in ("version", "bool_version"):
                    fixture.manifest["schema_version"] = 2 if category == "version" else True
                elif category == "record_extra":
                    fixture.manifest["records"][-1]["label"] = "not-a-target"
                fixture.save_manifest()
                if category == "duplicate":
                    text = fixture.manifest_path.read_text()
                    fixture.manifest_path.write_text(text.replace('"schema_version": 1',
                                                                '"schema_version": 1, "schema_version": 1', 1))
                self.before_arrays(fixture)

    def test_invalid_dataset_keys_refused(self):
        dataset = self.fixture(4).dataset()
        for key in (0, [0, 0], (0,), (0, 4), (0, True), (0, -1), (2, 0)):
            with self.subTest(key=key), self.assertRaises(ManifestError):
                dataset[key]
        for index in (-1, 4, True, 0.0):
            with self.subTest(index=index), self.assertRaises(IndexError):
                dataset[index, 0]


class BalancedSamplerTest(unittest.TestCase):
    def fixture(self, count=16):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return PilotFixture(Path(temporary.name), count).dataset()

    def test_deterministic_stratum_blocks_distinct_locations_and_step_seed(self):
        dataset = self.fixture()
        sampler = BalancedStepSampler(dataset, (4, 4, 4, 4), seed=17, total_steps=4)
        batches = list(sampler)
        self.assertEqual(len(sampler), 4)
        self.assertEqual(batches, list(sampler))
        self.assertNotEqual(batches, list(BalancedStepSampler(dataset, (4,) * 4, seed=18, total_steps=4)))
        self.assertNotEqual([index for _, index, _ in batches[0]], [index for _, index, _ in batches[1]])
        for step, batch in enumerate(batches):
            self.assertEqual([m for _, _, m in batch], [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
            self.assertEqual(len({index for _, index, _ in batch}), 16)
            self.assertEqual(len({dataset.metadata(index)["location_id"] for _, index, _ in batch}), 16)
            self.assertTrue(all(item_step == step for item_step, _, _ in batch))
            self.assertTrue(all(index in dataset.eligible_indices(m) for _, index, m in batch))

    def test_multi_rank_layout_reconstructs_global_and_equal_local_quotas(self):
        dataset = self.fixture()
        for world in (1, 2, 4):
            with self.subTest(world=world):
                samplers = [BalancedStepSampler(dataset, (4,) * 4, seed=17, rank=r,
                                               world_size=world, total_steps=1) for r in range(world)]
                batches = [next(iter(sampler)) for sampler in samplers]
                global_batch = samplers[0].global_batch(0)
                flattened = [item for batch in batches for item in batch]
                self.assertEqual(len({index for _, index, _ in flattened}), 16)
                self.assertEqual([batches[r][j] for j in range(16 // world) for r in range(world)], global_batch)
                for rank, batch in enumerate(batches):
                    self.assertEqual(batch, global_batch[rank::world])
                    self.assertEqual([sum(m == s for _, _, m in batch) for s in range(4)], [4 // world] * 4)

    def test_resume_has_same_step_batches_and_empty_tail(self):
        dataset = self.fixture()
        all_steps = list(BalancedStepSampler(dataset, (2,) * 4, seed=4, rank=1, world_size=2, total_steps=7))
        resumed = BalancedStepSampler(dataset, (2,) * 4, seed=4, rank=1, world_size=2,
                                      start_step=3, total_steps=7)
        self.assertEqual(len(resumed), 4)
        self.assertEqual(list(resumed), all_steps[3:])
        empty = BalancedStepSampler(dataset, (2,) * 4, seed=4, start_step=7, total_steps=7)
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])

    def test_constrained_inventory_chosen_before_single_sensor_dropout(self):
        dataset = self.fixture(8)
        for seed in range(100):
            batch = next(iter(BalancedStepSampler(dataset, (3, 3, 1, 1), seed=seed, total_steps=1)))
            self.assertEqual(len({i for _, i, _ in batch}), 8)
            self.assertTrue(all(i in dataset.eligible_indices(m) for _, i, m in batch))

    def test_insufficient_pilot_cannot_fake_n1024_or_oversubscribe_strata(self):
        dataset = self.fixture()
        for quotas in ((256, 256, 256, 256), (9, 1, 1, 1), (1, 9, 1, 1), (4, 4, 5, 4)):
            with self.subTest(quotas=quotas), self.assertRaisesRegex(ManifestError, "insufficient"):
                BalancedStepSampler(dataset, quotas, seed=4)

    def test_sampler_argument_rejections(self):
        dataset = self.fixture()
        cases = (
            {"quotas": (1, 1, 1)}, {"quotas": (0, 0, 0, 0)}, {"quotas": (-1, 1, 1, 1)},
            {"quotas": (True, 1, 1, 1)}, {"quotas": (1.0, 1, 1, 1)},
            {"world_size": 0}, {"world_size": True}, {"world_size": 3},
            {"rank": -1}, {"rank": 2, "world_size": 2}, {"seed": -1}, {"seed": True},
            {"start_step": -1}, {"start_step": 5, "total_steps": 4}, {"total_steps": -1},
        )
        for override in cases:
            args = {"quotas": (2,) * 4, "seed": 4, **override}
            with self.subTest(override=override), self.assertRaises(ManifestError):
                BalancedStepSampler(dataset, **args)

    def test_invalid_eligibility_pools_refused(self):
        class FakeDataset:
            def __init__(self, pools):
                self.pools = pools

            def __len__(self):
                return 8

            def eligible_indices(self, m):
                return self.pools[m]

        valid = [list(range(4)), list(range(4, 8)), list(range(8)), list(range(8))]
        for m, pool in ((0, [0, 0]), (0, [-1, 0]), (1, [0, 4]), (2, list(range(7))), (3, [False] + list(range(1, 8)))):
            with self.subTest(m=m, pool=pool):
                pools = copy.deepcopy(valid)
                pools[m] = pool
                with self.assertRaises(ManifestError):
                    BalancedStepSampler(FakeDataset(pools), (1,) * 4, seed=3)


if __name__ == "__main__":
    unittest.main()
