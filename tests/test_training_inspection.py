import copy
import json
import os
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import inspect_training_setup as inspection

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/training_inspection.json").read_text())


class TrainingInspectionTest(unittest.TestCase):
    def test_repository_config_and_bounded_schema(self):
        self.assertEqual(inspection.validate_config(CONFIG), CONFIG)
        cases = [None, {}, [], {**CONFIG, "command": "arbitrary"}]
        for key in CONFIG:
            missing = copy.deepcopy(CONFIG)
            del missing[key]
            cases.append(missing)
        for key in ("max_depth", "max_entries", "max_files", "deadline_seconds"):
            for value in (0, -1, True, 100000, "2"):
                cases.append({**CONFIG, key: value})
        cases.extend([
            {**CONFIG, "data_roots": ["/data"] * 5}, {**CONFIG, "data_roots": [None]},
            {**CONFIG, "expected_git_sha": "short"}, {**CONFIG, "seed": True},
            {**CONFIG, "repo_path": None},
        ])
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                inspection.validate_config(config)

    def test_only_storage_environment_variables_are_expanded(self):
        with patch.dict(os.environ, {"PROJECTDIR": "/projects/example", "PRIVATE_TOKEN": "do-not-export"}, clear=True):
            self.assertEqual(inspection.expand_path("${PROJECTDIR}/data"), Path("/projects/example/data"))
            for value in ("$PRIVATE_TOKEN", "${SCRATCHDIR}/data", "$UNKNOWN", "${malformed"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    inspection.expand_path(value)

    def test_inventory_caps_entries_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(10):
                (root / f"{i}.nc").write_bytes(b"not opened")
            for limits, expected in (({"max_files": 2}, "max_files"), ({"max_entries": 3}, "max_entries")):
                result = inspection.inventory(root, {**CONFIG, **limits})
                self.assertIn(expected, result["limits_hit"])
                self.assertLessEqual(result["entries_seen"], 3)
                self.assertLessEqual(len(result["files"]), 3)

    def test_storage_sample_is_shallow_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(70):
                (root / f"dir{i}").mkdir()
                (root / f"dir{i}" / "nested-file.nc").write_bytes(b"not opened")
            result = inspection.storage_sample(root)
            self.assertEqual(result["status"], "entry_limit")
            self.assertEqual(result["entries_seen"], 50)
            self.assertEqual(len(result["entries"]), 50)
            self.assertNotIn("nested-file", json.dumps(result))

    def test_inventory_excludes_hidden_sensitive_symlinks_and_special_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scene.nc", ".env", "access_token.json", "private.key", "credentials.json"):
                (root / name).write_bytes(b"must not be exported")
            (root / ".ssh").mkdir()
            (root / ".ssh" / "id_ed25519").write_text("private")
            (root / "alias.nc").symlink_to(root / "scene.nc")
            os.mkfifo(root / "pipe.npy")
            result = inspection.inventory(root, CONFIG)
            self.assertEqual([row["path"] for row in result["files"]], ["scene.nc"])
            self.assertNotIn("must not be exported", json.dumps(result))

    def test_refuses_broad_system_and_symlink_roots(self):
        for root in (Path("/"), Path.home(), Path("/projects"), Path("/etc"), Path("/proc/self")):
            with self.subTest(root=root):
                self.assertTrue(inspection.inventory(root, CONFIG)["status"].startswith("refused"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "alias").symlink_to(root / "data")
            self.assertTrue(inspection.inventory(root / "alias", CONFIG)["status"].startswith("refused"))
            self.assertEqual(inspection.inventory(root / "missing", CONFIG)["status"], "missing_or_not_directory")

    def test_depth_limit_and_no_following_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            (nested / "file.nc").write_bytes(b"x")
            result = inspection.inventory(root, {**CONFIG, "max_depth": 1})
            self.assertEqual(result["files"], [])
            self.assertIn("max_depth", result["limits_hit"])
            (root / "link").symlink_to(root / "a")
            result = inspection.inventory(root / "link" / "b", CONFIG)
            self.assertTrue(result["status"].startswith("refused"))

    def test_npy_header_without_array_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npy"
            header = repr({"descr": "<f4", "fortran_order": False, "shape": (3, 128, 128)}).encode() + b"\n"
            # There is intentionally no array payload.
            path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
            self.assertEqual(inspection.npy_header(path), {
                "status": "header_only", "shape": [3, 128, 128], "dtype": "<f4",
            })

    def test_npy_malformed_oversize_and_code_headers_are_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npy"
            contents = [b"not npy", b"\x93NUMPY\x09\x00", b"\x93NUMPY\x01\x00\x01",
                        b"\x93NUMPY\x01\x00" + struct.pack("<H", 9000)]
            malicious = b"__import__('os').environ.clear()"
            contents.append(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(malicious)) + malicious)
            with patch.dict(os.environ, {"SENTINEL": "preserved"}, clear=True):
                for value in contents:
                    path.write_bytes(value)
                    self.assertNotEqual(inspection.npy_header(path)["status"], "header_only")
                    self.assertEqual(os.environ["SENTINEL"], "preserved")

    def test_structured_dtype_field_names_are_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npy"
            header = repr({"descr": [("private_field", "<f4")], "shape": (10,)}).encode()
            path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
            self.assertNotIn("private_field", json.dumps(inspection.npy_header(path)))

    def test_git_queries_are_read_only_no_shell_no_remote_output(self):
        with patch.object(inspection.subprocess, "run") as run:
            run.return_value.stdout = "abc\n"
            self.assertEqual(inspection.run_git(ROOT, "rev-parse", "HEAD"), "abc")
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["git", "-C", str(ROOT), "rev-parse", "HEAD"])
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(kwargs["timeout"], 5)

    def test_report_has_no_secrets_gpu_pass_or_false_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**CONFIG, "repo_path": str(ROOT), "sif_path": str(root / "missing.sif")}
            path, output = root / "config.json", root / "report.json"
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"PRIVATE_TOKEN": "DO_NOT_EXPORT", "SECRET_KEY": "DO_NOT_EXPORT"}, clear=True):
                self.assertEqual(inspection.main(["--config", str(path), "--output", str(output)]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "inspection_complete_not_training_ready")
            self.assertEqual(report["gpu_readiness"], "not_tested")
            self.assertEqual(report["container_execution"], "not_tested")
            self.assertEqual(report["training_readiness"], "not_tested")
            self.assertEqual(report["quota_and_allocation"], "not_checked")
            self.assertEqual(report["sif"]["sha256"], "not_verified")
            self.assertEqual(report["data"], [])
            self.assertNotIn("DO_NOT_EXPORT", output.read_text())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                inspection.main(["--config", str(path), "--output", str(output)])

    def test_timeout_preserves_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, output = root / "config.json", root / "report.json"
            config.write_text(json.dumps(CONFIG))
            with patch.object(inspection, "inspect", side_effect=TimeoutError):
                self.assertEqual(inspection.main(["--config", str(config), "--output", str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "inspection_incomplete")
            self.assertEqual(report["error_type"], "TimeoutError")
            self.assertEqual(report["training_readiness"], "not_tested")


if __name__ == "__main__":
    unittest.main()
