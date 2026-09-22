# After a tool call, if a Slurm file was touched, report the local checker.
$ErrorActionPreference = "Stop"
try {
    $raw = [Console]::In.ReadToEnd()
    $blob = ""
    if ($raw) { $blob = $raw }
    if ($blob -notmatch "\.sbatch|[/\\]slurm[/\\]") {
        Write-Output "{}"
        exit 0
    }
    $root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $checker = Join-Path $root "scripts\check_sbatch.py"
    if (-not (Test-Path $checker)) {
        $checker = Join-Path $root "airr-science\scripts\check_sbatch.py"
    }
    $python = $null
    foreach ($candidate in @("python", "py", "H:\anaconda\python.exe")) {
        if ($candidate -eq "H:\anaconda\python.exe") {
            if (Test-Path $candidate) { $python = $candidate; break }
            continue
        }
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $python = $cmd.Source; break }
    }
    if (-not $python -or -not (Test-Path $checker)) {
        Write-Output '{"additional_context":"Slurm file edited. Run python scripts/check_sbatch.py before asking the human to submit. Do not sbatch."}'
        exit 0
    }
    $output = & $python $checker 2>&1 | Out-String
    $code = $LASTEXITCODE
    $note = if ($code -eq 0) {
        "Slurm templates pass scripts/check_sbatch.py. Do not submit them. The human runs sbatch."
    } else {
        "Slurm check failed. Fix the templates before proposing submission.`n$output"
    }
    $payload = @{ additional_context = $note.Trim() }
    Write-Output ($payload | ConvertTo-Json -Compress)
    exit 0
}
catch {
    Write-Output "{}"
    exit 0
}
