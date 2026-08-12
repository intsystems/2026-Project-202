<#
.SYNOPSIS
    Run grokking experiments on a Google Colab runtime from Windows.

.DESCRIPTION
    Wraps the four steps of a remote experiment so neither the code bundle nor the
    result files have to be shuffled by hand:

      -Sync            zip grokking_train/ + prediction_improved/, upload, unpack
      -Run a,b         execute run_probe.py for each run key, streaming output
      -Fetch <dir>     tar the outputs on the VM, download and extract them
      -Stop            terminate the session and its keep-alive daemon

    Steps combine in one call and always execute in the order above. A session is
    created automatically if it does not exist.

.PARAMETER Session
    Colab session name. Default 'gpu'.

.PARAMETER Gpu
    Accelerator for a session this script creates: T4, L4, G4, A100, H100, or
    'none' for a CPU runtime. Ignored if the session already exists.

.EXAMPLE
    # First time: create a T4 session, push the code, train both runs, bring results back
    .\colab_job.ps1 -Sync -Run mod_wd1,mod_wd0 -Fetch .\results -Stop

.EXAMPLE
    # Iterate on analysis without retraining
    .\colab_job.ps1 -Sync -Run s5_wd1
    .\colab_job.ps1 -Fetch .\results

.NOTES
    Requires the one-time setup in README.md (uv, the CLI, and colab_auth.py).
#>
[CmdletBinding()]
param(
    [string]$Session = "gpu",
    [ValidateSet("T4", "L4", "G4", "A100", "H100", "none")][string]$Gpu = "T4",
    [switch]$Sync,
    [string[]]$Run,
    [string]$Fetch,
    [switch]$Stop,
    [int]$TimeoutSeconds = 3600,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$colab = Join-Path $here "colab.ps1"
$repo = Split-Path -Parent $here
$temp = Join-Path $env:TEMP "colab_job_$PID"

function Invoke-Colab { & $colab @args; if ($LASTEXITCODE -ne 0) { throw "colab $($args -join ' ') failed" } }

# --- ensure the session exists ------------------------------------------------
$existing = & $colab sessions 2>&1 | Out-String
if ($existing -notmatch "\[$Session\]") {
    Write-Host "==> creating session '$Session'" -ForegroundColor Cyan
    if ($Gpu -eq "none") { Invoke-Colab new -s $Session }
    else { Invoke-Colab new -s $Session --gpu $Gpu }
} else {
    Write-Host "==> reusing session '$Session'" -ForegroundColor Cyan
}

New-Item -ItemType Directory -Force -Path $temp | Out-Null

# --- 1. sync -------------------------------------------------------------------
if ($Sync) {
    Write-Host "==> bundling code" -ForegroundColor Cyan
    $stage = Join-Path $temp "bundle"
    New-Item -ItemType Directory -Force -Path "$stage\grokking_train\grok", "$stage\prediction_improved", "$stage\edm_validation" | Out-Null
    Copy-Item "$repo\grokking_train\grok\*.py" "$stage\grokking_train\grok\"
    # All top-level modules, not only runs.py: train.py is the plain trainer that
    # edm_validation/phase13_dense_logging.py invokes, and test_train.py carries the
    # checks that can only run where torch is installed.
    Copy-Item "$repo\grokking_train\*.py" "$stage\grokking_train\"
    # All top-level modules: probe, run_probe, controls, verify_noninvasive, ...
    Copy-Item "$here\*.py" "$stage\prediction_improved\"
    if (Test-Path "$repo\edm_validation") {
        Copy-Item "$repo\edm_validation\*.py" "$stage\edm_validation\"
    }

    $zip = Join-Path $temp "edm_bundle.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path "$stage\*" -DestinationPath $zip

    Write-Host "==> uploading" -ForegroundColor Cyan
    Invoke-Colab upload -s $Session $zip /content/edm_bundle.zip
    Invoke-Colab exec -s $Session --timeout 300 -f (Join-Path $here "remote\unpack.py")
}

# --- 2. run --------------------------------------------------------------------
if ($Run) {
    # Generated rather than shipped: the run keys and any pass-through flags are
    # baked in, and streaming each line keeps `colab exec` from timing out on silence.
    $keys = ($Run | ForEach-Object { "'$_'" }) -join ", "
    $extra = ($ExtraArgs | ForEach-Object { "'$_'" }) -join ", "
    $driver = Join-Path $temp "drive_runs.py"
    @"
import subprocess, sys
for key in [$keys]:
    print(f"\n{'=' * 60}\n{key}\n{'=' * 60}", flush=True)
    cmd = [sys.executable, "-u", "/content/edm/prediction_improved/run_probe.py", key,
           "--outdir", "/content/out", "--force", "--progress-every", "100"] + [$extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"[{key}] exit={p.returncode}", flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
"@ | ForEach-Object {
        # UTF-8 *without* a BOM. `Set-Content -Encoding utf8` emits one on Windows
        # PowerShell 5.1, and the CLI reads -f files as plain "utf-8", so the BOM
        # would arrive as a leading ﻿ and the remote exec would SyntaxError.
        [System.IO.File]::WriteAllText($driver, $_, (New-Object System.Text.UTF8Encoding $false))
    }

    Write-Host "==> running: $($Run -join ', ')" -ForegroundColor Cyan
    Invoke-Colab exec -s $Session --timeout $TimeoutSeconds -f $driver
}

# --- 3. fetch ------------------------------------------------------------------
if ($Fetch) {
    Write-Host "==> packing outputs" -ForegroundColor Cyan
    Invoke-Colab exec -s $Session --timeout 600 -f (Join-Path $here "remote\pack_outputs.py")

    New-Item -ItemType Directory -Force -Path $Fetch | Out-Null
    $tar = Join-Path $Fetch "outputs.tar.gz"
    Invoke-Colab download -s $Session /content/outputs.tar.gz $tar
    tar -xzf $tar -C $Fetch
    Remove-Item $tar -Force
    Write-Host "==> results in $Fetch" -ForegroundColor Green
    Get-ChildItem $Fetch | Select-Object Name, Length | Format-Table -AutoSize
}

# --- 4. stop -------------------------------------------------------------------
if ($Stop) {
    Write-Host "==> stopping session '$Session'" -ForegroundColor Cyan
    & $colab stop -s $Session
}

Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
