import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_sbatch", ROOT / "scripts" / "check_sbatch.py")
check_sbatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_sbatch)

GOOD = """#!/bin/bash
#SBATCH --job-name=t
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
# comments may mention curl and pip install
set -euo pipefail
echo "git_sha=$(git rev-parse HEAD)"
apptainer exec --nv x.sif python3 -c 1
"""


class CheckSbatchTest(unittest.TestCase):
    def test_repo_templates_pass(self) -> None:
        for path in sorted((ROOT / "slurm").glob("*.sbatch")):
            self.assertEqual(check_sbatch.check(path), [], path.name)

    def test_good_template(self) -> None:
        self.assertEqual(check_sbatch.check_text(GOOD), [])

    def test_bad_templates(self) -> None:
        cases = {
            "no time": GOOD.replace("#SBATCH --time=00:10:00\n", ""),
            "too long": GOOD.replace("00:10:00", "00:45:00"),
            "two nodes": GOOD.replace("--nodes=1", "--nodes=2"),
            "no nodes": GOOD.replace("#SBATCH --nodes=1\n", ""),
            "no gpus": GOOD.replace("#SBATCH --gpus=1\n", ""),
            "eight gpus": GOOD.replace("--gpus=1", "--gpus=8"),
            "exclusive": GOOD + "#SBATCH --exclusive\n",
            "curl": GOOD + "curl -O https://example.org/x\n",
            "hf download": GOOD + "hf download org/model\n",
            "pip": GOOD + "python3 -m pip install torch\n",
            "pull": GOOD + "apptainer pull x.sif docker://y\n",
            "no nv": GOOD.replace("--nv ", ""),
            "env dump": GOOD + "env | sort\n",
            "mpirun": GOOD + "mpirun -n 4 a.out\n",
            "no sha": GOOD.replace("git rev-parse HEAD", "true"),
            "no strict": GOOD.replace("set -euo pipefail", ""),
        }
        for name, text in cases.items():
            self.assertTrue(check_sbatch.check_text(text), name)

    def test_marks(self) -> None:
        long_job = GOOD.replace("00:10:00", "01:00:00") + "# OPENCHANGE_LONG_JOB\n"
        self.assertEqual(check_sbatch.check_text(long_job), [])
        pull = GOOD + "# OPENCHANGE_CONTAINER_PULL\napptainer pull x.sif docker://y\n"
        self.assertEqual(check_sbatch.check_text(pull), [])


if __name__ == "__main__":
    unittest.main()
