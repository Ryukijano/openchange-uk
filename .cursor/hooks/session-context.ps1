# Remind every session of the Isambard loop. Does not open a connection.
$ErrorActionPreference = "Stop"
try {
    [Console]::In.ReadToEnd() | Out-Null
    $text = @"
Isambard-AI Phase 2 (project u6xn) is aarch64 GH200. One GPU request is one GH200. Login nodes have no GPUs. Do not ssh, sbatch, srun, scancel, or run Clifton. Propose commands; the human runs them after https://docs.isambard.ac.uk . Order: make test, make check-sbatch, 10-minute 1-GPU smoke, then a dry-run. Cap suggested --time at 00:30:00. Container: nvcr.io/nvidia/pytorch:25.05-py3 with apptainer exec --nv. Never pip-install an x86 torch wheel. Workflow: docs/WORKFLOW.md.
"@
    $payload = @{ additional_context = $text.Trim() }
    Write-Output ($payload | ConvertTo-Json -Compress)
    exit 0
}
catch {
    Write-Output "{}"
    exit 0
}
