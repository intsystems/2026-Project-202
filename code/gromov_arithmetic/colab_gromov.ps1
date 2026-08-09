<#
.SYNOPSIS
    Run the two Gromov reproductions on a Google Colab runtime from Windows.

.DESCRIPTION
    Same shape as ../prediction_improved/colab_job.ps1, with two differences that
    matter for a campaign this long:

      * the bundle is gromov_arithmetic + gromov_polynomials, which share one
        training core, so both must be pushed together or the polynomial runs
        cannot import it;
      * -Job launches a *sequence* of commands under one detached process. A
        foreground -Run holds the exec channel, and every poll and every download
        then queues behind it -- which is how a reclaimed VM takes the results with
        it. Detached + poll is the only safe shape for anything over a few minutes.

.EXAMPLE
    .\colab_gromov.ps1 -Sync
    .\colab_gromov.ps1 -Job .\jobs\sweep.json
    .\colab_gromov.ps1 -Poll
    .\colab_gromov.ps1 -Fetch .\results
#>
[CmdletBinding()]
param(
    [string]$Session = "gromov",
    [ValidateSet("T4", "L4", "A100", "none")][string]$Gpu = "T4",
    [switch]$Sync,
    [string]$Job,
    [switch]$Poll,
    [string]$Fetch,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$colab = Join-Path $repo "prediction_improved\colab.ps1"
$shared = Join-Path $repo "prediction_improved\remote"
$temp = Join-Path $env:TEMP "colab_gromov_$PID"

function Invoke-Colab { & $colab @args; if ($LASTEXITCODE -ne 0) { throw "colab $($args -join ' ') failed" } }

# "The listing does not mention the session" and "the listing failed" are different
# facts, and conflating them cost a whole campaign: a ConnectTimeout to
# colab.research.google.com produced an empty listing, this script read that as "no such
# session" and called `new`, and the failed `new` destroyed the local name-to-runtime
# mapping of a live VM -- which the CLI can only address by name. The runtime stayed
# alive, unreachable, holding the GPU. So: only a *successful* listing is allowed to
# conclude that a session is absent.
$existing = & $colab sessions 2>&1 | Out-String
$listed = ($LASTEXITCODE -eq 0)
if (-not $listed -or $existing -notmatch "\[$Session\]") {
    Start-Sleep -Seconds 5
    $existing = & $colab sessions 2>&1 | Out-String
    $listed = ($LASTEXITCODE -eq 0)
}
if (-not $listed) {
    throw ("cannot list sessions (the CLI failed, most likely a network timeout). " +
           "Refusing to create '$Session': if it already exists, creating would orphan " +
           "it. Retry when the listing works.")
}
if ($existing -notmatch "\[$Session\]") {
    Write-Host "==> creating session '$Session'" -ForegroundColor Cyan
    if ($Gpu -eq "none") { Invoke-Colab new -s $Session } else { Invoke-Colab new -s $Session --gpu $Gpu }
} else {
    Write-Host "==> reusing session '$Session'" -ForegroundColor Cyan
}
New-Item -ItemType Directory -Force -Path $temp | Out-Null

# --- 1. sync ------------------------------------------------------------------
if ($Sync) {
    Write-Host "==> bundling code" -ForegroundColor Cyan
    $stage = Join-Path $temp "bundle"
    New-Item -ItemType Directory -Force -Path "$stage\gromov_arithmetic\remote", "$stage\gromov_polynomials", "$stage\active_rank" | Out-Null
    Copy-Item "$here\*.py" "$stage\gromov_arithmetic\"
    Copy-Item "$here\remote\*.py" "$stage\gromov_arithmetic\remote\"
    Copy-Item "$repo\gromov_polynomials\*.py" "$stage\gromov_polynomials\"
    # rank.py imports the CountSketch from ../active_rank rather than reimplementing it,
    # so that folder has to travel with the bundle or the sketched runs cannot start.
    Copy-Item "$repo\active_rank\*.py" "$stage\active_rank\"

    $zip = Join-Path $temp "edm_bundle.zip"
    Compress-Archive -Path "$stage\*" -DestinationPath $zip
    Write-Host "==> uploading" -ForegroundColor Cyan
    Invoke-Colab upload -s $Session $zip /content/edm_bundle.zip
    Invoke-Colab exec -s $Session --timeout 300 -f (Join-Path $shared "unpack.py")
}

# --- 2. launch a job sequence, detached ---------------------------------------
if ($Job) {
    if (-not (Test-Path $Job)) { throw "job file not found: $Job" }
    Write-Host "==> launching $Job" -ForegroundColor Cyan
    Invoke-Colab upload -s $Session (Resolve-Path $Job).Path /content/job_seq.json

    # One argument per line; launch_detached.py reads it as the argv of the process.
    $cmd = Join-Path $temp "job_cmd.txt"
    [System.IO.File]::WriteAllText($cmd, "/content/edm/gromov_arithmetic/remote/run_sequence.py`n",
        (New-Object System.Text.UTF8Encoding $false))
    Invoke-Colab upload -s $Session $cmd /content/job_cmd.txt
    # A fresh sequence must not be blocked by the pid file of a finished one: every
    # sequence shares the same argv[0], so launch_detached.py's liveness check cannot
    # tell a recycled pid from ours.
    Invoke-Colab exec -s $Session --timeout 60 -f (Join-Path $here "remote\clear_pid.py")
    Invoke-Colab exec -s $Session --timeout 120 -f (Join-Path $shared "launch_detached.py")
}

# --- 3. poll -------------------------------------------------------------------
if ($Poll) {
    # Deliberately not Invoke-Colab: a poll must never throw. The CLI returns a
    # non-zero status for transient conditions that say nothing about the job, and a
    # poller that aborts on those is worse than one that prints a bad status line.
    & $colab exec -s $Session --timeout 120 -f (Join-Path $here "remote\poll.py")
}

# --- 4. fetch ------------------------------------------------------------------
if ($Fetch) {
    Write-Host "==> packing outputs" -ForegroundColor Cyan
    Invoke-Colab exec -s $Session --timeout 900 -f (Join-Path $shared "pack_outputs.py")
    New-Item -ItemType Directory -Force -Path $Fetch | Out-Null
    $tar = Join-Path $Fetch "outputs.tar.gz"
    Invoke-Colab download -s $Session /content/outputs.tar.gz $tar
    tar -xzf $tar -C $Fetch
    Remove-Item $tar -Force
    Write-Host "==> results in $Fetch" -ForegroundColor Green
    Get-ChildItem $Fetch -Recurse -File | Select-Object Name, Length | Format-Table -AutoSize
}

if ($Stop) {
    Write-Host "==> stopping session '$Session'" -ForegroundColor Cyan
    & $colab stop -s $Session
}
Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
