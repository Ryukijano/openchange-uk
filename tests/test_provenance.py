import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openchange import provenance


class ProvenanceTest(unittest.TestCase):
    def test_filtered_env_drops_secrets_and_unrelated(self) -> None:
        env = {
            "SLURM_JOB_ID": "42",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENCHANGE_SIF": "/p/x.sif",
            "OPENCHANGE_API_TOKEN": "nope",
            "HF_TOKEN": "nope",
            "SSH_AUTH_SOCK": "/tmp/agent",
            "HOME": "/home/u",
        }
        kept = provenance.filtered_env(env)
        self.assertEqual(set(kept), {"SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "OPENCHANGE_SIF"})

    def test_env_git_sha_wins(self) -> None:
        with mock.patch.dict(os.environ, {"OPENCHANGE_GIT_SHA": "abc", "OPENCHANGE_GIT_DIRTY": "1"}):
            self.assertEqual(provenance.git_state(), {"sha": "abc", "dirty": True, "source": "env"})

    def test_run_id_uses_slurm_job(self) -> None:
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            self.assertEqual(provenance.run_id(), "slurm123")

    def test_start_run_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text("seed: 7\n")
            env = {"SLURM_JOB_ID": "99", "OPENCHANGE_GIT_SHA": "deadbeef", "HF_TOKEN": "x"}
            with mock.patch.dict(os.environ, env):
                run_dir, manifest = provenance.start_run("t", config, 7, Path(tmp) / "out")
                with self.assertRaises(FileExistsError):
                    provenance.start_run("t", config, 7, Path(tmp) / "out")
            self.assertEqual(run_dir.name, "t-slurm99")
            written = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(written["seed"], 7)
            self.assertEqual(written["git"]["sha"], "deadbeef")
            self.assertEqual(written["config"]["sha256"], provenance.sha256_file(config))
            self.assertNotIn("HF_TOKEN", written["env"])
            self.assertIn("machine", written["environment"])
            self.assertEqual(manifest["run_dir"], str(run_dir))


if __name__ == "__main__":
    unittest.main()
