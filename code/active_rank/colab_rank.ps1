<#
.SYNOPSIS
    Run the parameter-rank experiment on a Google Colab runtime from Windows.

.DESCRIPTION
    Same shape as ../prediction_improved/colab_job.ps1, which documents the one-time
    setup (uv, the CLI, colab_auth.py) -- this only changes what is bundled and what is
    executed. Each run key is a separate `colab exec`, so a reclaimed VM costs at most
    one run rather than the whole campaign.

.EXAMPLE
    .\colab_rank.ps1 -Sync -Run mod_wd1,mod_wd0 -Fetch .\results -Stop
#>
[CmdletBinding()]
param(
    [string]$Session = "rank",
    [ValidateSet("T4", "L4", "G4", "A100", "H100", "none")][string]$Gpu = "T4",
    [switch]$Sync,
    [string[]]$Run,
    [string]$Fetch,
    [switch]$Stop,
    [int]$TimeoutSeconds = 2400,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$colab = Join-Path $repo "prediction_improved\colab.ps1"
$temp = Join-Path $env:TEMP "colab_rank_$PID"

function Invoke-Colab { & $colab @args; if ($LASTEXITCODE -ne 0) { throw "colab $($args -join ' ') failed" } }

$existing = & $colab sessions 2>&1 | Out-String
if ($existing -notmatch "\[$Session\]") {
    Write-Host "==> creating session '$Session'" -ForegroundColor Cyan
    if ($Gpu -eq "none") { Invoke-Colab new -s $Session } else { Invoke-Colab new -s $Session --gpu $Gpu }
} else {
    Write-Host "==> reusing session '$Session'" -ForegroundColor Cyan
}
New-Item -ItemType Directory -Force -Path $temp | Out-Null

if ($Sync) {
    Write-Host "==> bundling code" -ForegroundColor Cyan
    $stage = Join-Path $temp "bundle"
    New-Item -ItemType Directory -Force -Path "$stage\grokking_train\grok", "$stage\active_rank" | Out-Null
    Copy-Item "$repo\grokking_train\grok\*.py" "$stage\grokking_train\grok\"
    Copy-Item "$repo\grokking_train\*.py" "$stage\grokking_train\"
    Copy-Item "$here\*.py" "$stage\active_rank\"

    $zip = Join-Path $temp "rank_bundle.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path "$stage\*" -DestinationPath $zip

    Write-Host "==> uploading" -ForegroundColor Cyan
    Invoke-Colab upload -s $Session $zip /content/rank_bundle.zip

    Invoke-Colab exec -s $Session --timeout 300 -f (Join-Path $here "remote_unpack.py")
}

if ($Run) {
    $extra = ($ExtraArgs | ForEach-Object { "'$_'" }) -join ", "
    foreach ($key in $Run) {
        $driver = Join-Path $temp "drive_$key.py"
        @"
import subprocess, sys
# --tag defaults to the run key so the training CSV is named '<key>_train.csv' and
# analyze_rank.py can pair it with '<key>_rank.npz' without consulting the registry.
cmd = [sys.executable, "-u", "/content/edm/active_rank/run_rank.py", "$key",
       "--outdir", "/content/out", "--force", "--progress-every", "200",
       "--tag", "$key"] + [$extra]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
print(f"[$key] exit={p.returncode}", flush=True)
raise SystemExit(p.returncode)
"@ | ForEach-Object { [System.IO.File]::WriteAllText($driver, $_, (New-Object System.Text.UTF8Encoding $false)) }
        Write-Host "==> running $key" -ForegroundColor Cyan
        Invoke-Colab exec -s $Session --timeout $TimeoutSeconds -f $driver
    }
}

if ($Fetch) {
    Write-Host "==> packing outputs" -ForegroundColor Cyan
    $pack = Join-Path $temp "pack.py"
    @"
import subprocess
print(subprocess.run(['tar','-czf','/content/rank_outputs.tar.gz','-C','/content/out','.'],
      capture_output=True, text=True).returncode)
print(subprocess.run(['ls','-la','/content/out'], capture_output=True, text=True).stdout)
"@ | ForEach-Object { [System.IO.File]::WriteAllText($pack, $_, (New-Object System.Text.UTF8Encoding $false)) }
    Invoke-Colab exec -s $Session --timeout 600 -f $pack

    New-Item -ItemType Directory -Force -Path $Fetch | Out-Null
    $tar = Join-Path $Fetch "rank_outputs.tar.gz"
    Invoke-Colab download -s $Session /content/rank_outputs.tar.gz $tar
    tar -xzf $tar -C $Fetch
    Remove-Item $tar -Force
    Write-Host "==> results in $Fetch" -ForegroundColor Green
    Get-ChildItem $Fetch | Select-Object Name, Length | Format-Table -AutoSize
}

if ($Stop) { Write-Host "==> stopping '$Session'" -ForegroundColor Cyan; & $colab stop -s $Session }
Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
