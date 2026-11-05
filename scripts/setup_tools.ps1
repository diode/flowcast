# Set up this project's tools inside the project folder (docs/setup.md):
#   tools\CARLA_0.9.16\   CARLA server + its SUMO co-simulation bridge
# and check the rest: .venv (Python, SUMO, CARLA client), ffmpeg with H.264.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1               # download + unzip + check
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1 -Zip D:\CARLA_0.9.16.zip   # already downloaded
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1 -CheckOnly
param(
    [string]$Zip = "",
    [switch]$CheckOnly,
    [switch]$KeepZip
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root "tools"
$Carla = Join-Path $Tools "CARLA_0.9.16"
$Url = "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Ok($m) { Write-Host "  ok    $m" -ForegroundColor Green }
function Bad($m) { Write-Host "  MISSING  $m" -ForegroundColor Red; $script:failed = $true }
$failed = $false

if (-not $CheckOnly -and -not (Test-Path (Join-Path $Carla "CarlaUE4.exe"))) {
    New-Item -ItemType Directory -Force $Tools | Out-Null
    if (-not $Zip) {
        $Zip = Join-Path $Tools "CARLA_0.9.16.zip"
        if (-not (Test-Path $Zip)) {
            $free = [math]::Round((Get-PSDrive (Split-Path -Qualifier $Root).TrimEnd(":")).Free / 1GB)
            Write-Host "Downloading $Url (several GB; $free GB free) ..."
            # BITS (a Windows service) first; it can stop at the end when the server answers a retry without
            # HTTP Range support. Then Windows' own curl.exe.
            # Asynchronous, so a timeout is a transient error that BITS retries instead of a failure.
            try {
                $job = Start-BitsTransfer -Source $Url -Destination $Zip -DisplayName "CARLA 0.9.16" -Priority Foreground `
                    -Asynchronous -RetryInterval 60 -RetryTimeout 7200
                $last = -1
                while ($job.JobState -notin "Transferred", "Error", "Cancelled") {
                    Start-Sleep 30
                    $gb = [math]::Round($job.BytesTransferred / 1GB, 1)
                    if ($gb -ne $last) { Write-Host "  $gb GB  ($($job.JobState))"; $last = $gb }
                    if ($job.JobState -eq "TransientError") { Resume-BitsTransfer $job -Asynchronous | Out-Null }
                }
                if ($job.JobState -ne "Transferred") {
                    $msg = $job.ErrorDescription; Remove-BitsTransfer $job; throw $msg
                }
                Complete-BitsTransfer $job
            } catch {
                Write-Host "BITS failed ($("$_".Trim())); trying curl ..."
                curl.exe -L --fail --retry 5 --retry-delay 10 -o "$Zip.part" $Url
                if ($LASTEXITCODE -ne 0) {
                    Remove-Item -ErrorAction SilentlyContinue "$Zip.part"
                    throw "download failed. Download $Url in a browser and run: setup_tools.ps1 -Zip <path to the zip>"
                }
                Move-Item "$Zip.part" $Zip
            }
        }
    }
    Write-Host "Unzipping $Zip -> $Carla ..."
    New-Item -ItemType Directory -Force $Carla | Out-Null
    tar -xf $Zip -C $Carla                       # Windows' bsdtar: much faster than Expand-Archive
    if (-not (Test-Path (Join-Path $Carla "CarlaUE4.exe"))) {   # zip with a top folder: move it up
        $inner = Get-ChildItem $Carla -Directory | Where-Object { Test-Path (Join-Path $_.FullName "CarlaUE4.exe") } | Select-Object -First 1
        if ($inner) { Get-ChildItem $inner.FullName | Move-Item -Destination $Carla; Remove-Item $inner.FullName }
    }
    if (-not $KeepZip -and $Zip -eq (Join-Path $Tools "CARLA_0.9.16.zip")) { Remove-Item $Zip }
}

Write-Host "Checking tools in $Root"
if (Test-Path (Join-Path $Carla "CarlaUE4.exe")) { Ok "CARLA server  $Carla\CarlaUE4.exe" } else { Bad "CARLA server  $Carla\CarlaUE4.exe" }
if (Test-Path (Join-Path $Carla "Co-Simulation\Sumo\run_synchronization.py")) { Ok "CARLA-SUMO bridge  Co-Simulation\Sumo" } else { Bad "CARLA-SUMO bridge  $Carla\Co-Simulation\Sumo" }
if (Test-Path $Py) {
    $v = & $Py -c "import carla, sumolib, libsumo, cv2, numpy; from importlib.metadata import version; print(version('carla'), version('eclipse-sumo'))" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "Python packages  carla / eclipse-sumo $v" } else { Bad "Python packages: .venv\Scripts\python -m pip install -e . and the CARLA 0.9.16 client (pip install carla==0.9.16)" }
} else { Bad ".venv: py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -e `".[dev]`"" }
$ff = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if ($ff -and ((& $ff -hide_banner -encoders 2>$null) -match "libx264")) { Ok "ffmpeg with H.264  $ff" } else { Bad "ffmpeg with libx264 on PATH (or recording.ffmpeg in the city config)" }

if ($failed) { Write-Host "Some tools are missing (above)."; exit 1 }
Write-Host "All tools ready. Start CARLA with:"
Write-Host "  $Carla\CarlaUE4.exe -RenderOffScreen -graphicsadapter=0"
