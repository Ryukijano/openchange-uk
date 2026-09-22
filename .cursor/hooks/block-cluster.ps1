# Deny cluster control commands. Propose them; do not run them.
$ErrorActionPreference = "Stop"
try {
    $raw = [Console]::In.ReadToEnd()
    $cmd = ""
    if ($raw) {
        $json = $raw | ConvertFrom-Json
        foreach ($name in @("command", "cmd")) {
            if ($json.PSObject.Properties.Name -contains $name -and $json.$name) {
                $cmd = [string]$json.$name
                break
            }
        }
    }
    $denied = $false
    $why = ""
    foreach ($part in ($cmd -split "[;&|]+")) {
        $trimmed = $part.Trim()
        if (-not $trimmed) { continue }
        $token = ($trimmed -split "\s+", 2)[0]
        $token = Split-Path -Leaf $token
        if ($token -match "^(sbatch|srun|salloc|scancel|squeue|sinfo|sacct|scontrol)(\.exe)?$") {
            $denied = $true
            $why = "Slurm command '$token' is for you to run on Isambard, not for the agent."
            break
        }
        if ($token -match "^clifton(\.exe)?$") {
            $denied = $true
            $why = "Clifton auth stays in your terminal. The agent must not see the passphrase."
            break
        }
        if ($token -match "^ssh(\.exe)?$" -and $trimmed -match "isambard|aip2") {
            $denied = $true
            $why = "SSH to Isambard stays in your terminal. The agent proposes commands only."
            break
        }
    }
    if ($denied) {
        $payload = @{
            permission    = "deny"
            user_message  = $why
            agent_message = "$why Docs: https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/"
        }
        Write-Output ($payload | ConvertTo-Json -Compress)
        exit 0
    }
    Write-Output '{"permission":"allow"}'
    exit 0
}
catch {
    Write-Output '{"permission":"allow"}'
    exit 0
}