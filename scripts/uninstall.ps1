param(
  [switch]$Clean,
  [switch]$KeepModels
)
$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "Vocalis uninstaller - Windows" -ForegroundColor Cyan
Write-Host "Project: $root"

# 1. pip uninstall (if venv exists)
$pip = Join-Path $root ".venv\Scripts\pip.exe"
if (Test-Path $pip) {
  Write-Host "Running pip uninstall vocalis ..."
  & $pip uninstall -y vocalis 2>$null
}

# 2. remove venv / build artefacts
foreach ($p in @(".venv","build","dist","__pycache__","src\vocalis\__pycache__",".pytest_cache")) {
  $full = Join-Path $root $p
  if (Test-Path $full) {
    Write-Host "Removing $p ..."
    Remove-Item -Recurse -Force -LiteralPath $full
  }
}
# egg-info
Get-ChildItem -Path $root -Filter "*.egg-info" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  Write-Host "Removing $($_.Name) ..."
  Remove-Item -Recurse -Force -LiteralPath $_.FullName
}

# 3. desktop shortcut + Start Menu folder (install.ps1 creates Programs\Vocalis\*)
try {
  $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Vocalis.lnk"
  if (Test-Path $lnk) { Remove-Item -Force -LiteralPath $lnk; Write-Host "Removed Desktop shortcut" }
  # legacy single file (pre-0.1.0)
  $legacy = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Start Menu\Programs\Vocalis.lnk"
  if (Test-Path $legacy) { Remove-Item -Force -LiteralPath $legacy }
  # current: Programs\Vocalis folder with Vocalis.lnk + Uninstall Vocalis.lnk
  $startDir = Join-Path ([Environment]::GetFolderPath("Programs")) "Vocalis"
  if (Test-Path $startDir) { Remove-Item -Recurse -Force -LiteralPath $startDir; Write-Host "Removed Start Menu folder Vocalis" }
  # also LOCALAPPDATA variant (some machines)
  $altStart = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Start Menu\Programs\Vocalis"
  if (Test-Path $altStart) { Remove-Item -Recurse -Force -LiteralPath $altStart }
} catch {}

# 4. config / models (only with -Clean)
if ($Clean) {
  # user_config_dir / user_data_dir with APP_NAME=vocalis via platformdirs
  $cfg = Join-Path $env:APPDATA "vocalis"
  $data = Join-Path $env:LOCALAPPDATA "vocalis"
  # fallback for platformdirs on some setups
  if (-not (Test-Path $cfg) -and (Test-Path (Join-Path $env:APPDATA "Vocalis"))) { $cfg = Join-Path $env:APPDATA "Vocalis" }
  if (-not (Test-Path $data) -and (Test-Path (Join-Path $env:LOCALAPPDATA "Vocalis"))) { $data = Join-Path $env:LOCALAPPDATA "Vocalis" }

  if (Test-Path $cfg) {
    Write-Host "Removing config at $cfg ..."
    Remove-Item -Recurse -Force -LiteralPath $cfg
  } else { Write-Host "No config dir at $cfg (nothing to do)" }

  if ($KeepModels) {
    Write-Host "-KeepModels: leaving models + alarms under $data (keeps ~700 MB models and offline alarms.json)"
    # still clean corrupted extras but keep data dir
  } else {
    if (Test-Path $data) {
      Write-Host "Removing data/models/alarms at $data (includes alarms.json) ..."
      Remove-Item -Recurse -Force -LiteralPath $data
    } else { Write-Host "No data dir at $data" }
  }
} else {
  Write-Host "Keeping user config/models. Re-run with -Clean to remove them (add -KeepModels to keep ~700 MB models)."
  Write-Host "  .\scripts\uninstall.ps1 -Clean              # remove everything"
  Write-Host "  .\scripts\uninstall.ps1 -Clean -KeepModels  # keep models"
}

Write-Host "`nUninstall done. To reinstall, double-click install.bat" -ForegroundColor Green

