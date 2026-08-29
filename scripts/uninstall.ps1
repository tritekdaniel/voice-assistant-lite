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

# 3. desktop shortcut
try {
  $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Vocalis.lnk"
  if (Test-Path $lnk) { Remove-Item -Force -LiteralPath $lnk; Write-Host "Removed Desktop shortcut" }
  $appEntry = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Start Menu\Programs\Vocalis.lnk"
  if (Test-Path $appEntry) { Remove-Item -Force -LiteralPath $appEntry }
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
    Write-Host "-KeepModels: leaving models under $data"
  } else {
    if (Test-Path $data) {
      Write-Host "Removing data/models at $data ..."
      Remove-Item -Recurse -Force -LiteralPath $data
    } else { Write-Host "No data dir at $data" }
  }
} else {
  Write-Host "Keeping user config/models. Re-run with -Clean to remove them (add -KeepModels to keep ~700 MB models)."
  Write-Host "  .\scripts\uninstall.ps1 -Clean              # remove everything"
  Write-Host "  .\scripts\uninstall.ps1 -Clean -KeepModels  # keep models"
}

Write-Host "`nUninstall done. To reinstall, double-click install.bat" -ForegroundColor Green

