param([switch]$Clean)
$ErrorActionPreference="Stop"
$root=(Resolve-Path "$PSScriptRoot/..").Path
$venv="$root/.venv"
$py="$venv/Scripts/python.exe"
if(-not (Test-Path $py)){ $py="py -V:Astral\CPython3.11.15" }
Write-Host "Building standalone Vocalis binary (PyInstaller)..." -ForegroundColor Cyan
if($Clean -and (Test-Path "$root/build")){ Remove-Item -Recurse -Force "$root/build" }
if($Clean -and (Test-Path "$root/dist")){ Remove-Item -Recurse -Force "$root/dist" }
& $py -m pip install --upgrade pyinstaller 2>$null
# ensure CPU torch already installed via install.ps1; if not, install it now
& $py -m pip show torch 2>$null | Out-Null
if($LASTEXITCODE -ne 0){
  Write-Host "Installing torch CPU first..."
  & "$venv/Scripts/pip.exe" install --index-url https://download.pytorch.org/whl/cpu torch --upgrade
}
Push-Location $root
try{
  # Use python -m PyInstaller via Start-Process to avoid PowerShell turning INFO logs into errors
  $psi = Start-Process -FilePath "$venv/Scripts/python.exe" -ArgumentList "-m","PyInstaller","packaging/vocalis.spec","--noconfirm","--clean" -NoNewWindow -Wait -PassThru
  if($psi.ExitCode -ne 0){ throw "pyinstaller failed with exit $($psi.ExitCode)" }
  Write-Host "Done: dist/Vocalis/Vocalis.exe" -ForegroundColor Green
  Write-Host "Run: dist/Vocalis/Vocalis.exe  or  dist/Vocalis/Vocalis.exe --check"
} finally{ Pop-Location }
