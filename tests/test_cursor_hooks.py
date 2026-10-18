import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hooks", ROOT / ".cursor" / "hooks" / "hooks.py")
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)


class DenyHookTest(unittest.TestCase):
    def test_denied(self) -> None:
        # Only clifton stays denied: auth is interactive SSO, run by the human.
        # Slurm and SSH are permitted after human authentication (AGENTS.md).
        for command in [
            "clifton auth",
            "sudo clifton auth --identity ~/.ssh/id_ed25519",
            "FOO=1 clifton whoami",
            "echo $(clifton token)",
        ]:
            self.assertIsNotNone(hooks.decide(command), command)

    def test_allowed(self) -> None:
        for command in [
            "make test",
            "python scripts/check_sbatch.py",
            "cat slurm/00_smoke.sbatch",
            "ssh git@github.com",
            "sbatch slurm/00_smoke.sbatch",
            "make test && squeue --me",
            "/usr/bin/srun --gpus=1 hostname",
            "sudo scancel 1",
            "FOO=1 sacct",
            "ssh u6xn.aip2.isambard",
            "scp file u6xn.aip2.isambard:~",
            "echo $(sinfo)",
            "git status",
            "",
        ]:
            self.assertIsNone(hooks.decide(command), command)


if __name__ == "__main__":
    unittest.main()
