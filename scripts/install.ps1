param(
  [switch]$NoCheck,
  [switch]$NoShortcut,
  [switch]$NoBinary,
  [switch]$WithBinary
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $root ".venv"
$py = $null

Write-Host "Vocalis installer - Windows" -ForegroundColor Cyan
Write-Host "Project: $root"

# 1. Find Python 3.11 (required: avoids 3.14 CUDA wheels, blocked by requires-python <3.14)
$candidates = @(
  "py -V:Astral\CPython3.11.15",
  "py -3.11",
  "python3.11",
  "python"
)
foreach ($c in $candidates) {
  $parts = $c -split " ",2
  $exe = $parts[0]; $args = if ($parts.Count -gt 1) { $parts[1] } else { $null }
  try {
    $cmd = if ($args) { & $exe $args --version 2>&1 } else { & $exe --version 2>&1 }
    $ver = "$cmd"
    if ($ver -match "3\.11\.") { $py = $c; Write-Host "Using $c ($ver)" -ForegroundColor Green; break }
    if ($ver -match "3\.1[12]\.") { Write-Host "Skipping $c ($ver) - need 3.11 for CPU torch" -ForegroundColor Yellow }
  } catch { }
}
if (-not $py) {
  Write-Host "Python 3.11 not found. Install it via https://www.python.org/downloads/ or 'uv python install 3.11' then re-run." -ForegroundColor Red
  Write-Host "Tried: $($candidates -join ', ')"
  exit 1
}

# 2. Create venv
if (-not (Test-Path $venv)) {
  Write-Host "Creating venv at $venv ..."
  $parts = $py -split " ",2; $exe=$parts[0]; $a=if($parts.Count -gt 1){$parts[1]}else{$null}
  if ($a) { & $exe $a -m venv $venv } else { & $exe -m venv $venv }
} else {
  Write-Host "venv already exists at $venv - reusing (delete it to force fresh install)"
}
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $pip)) { Write-Host "pip not found in venv - venv creation failed" -ForegroundColor Red; exit 1 }

# 3. Upgrade pip + install torch CPU first (keeps download small, ~300MB not 2-3GB)
# Clean up any broken ~ installs left from previous WinError 32 BEFORE any pip (pip scans site-packages on every run)
Get-ChildItem -LiteralPath (Join-Path $venv "Lib\site-packages") -Filter "~*" -Force -ErrorAction SilentlyContinue | ForEach-Object {
  Write-Host "Cleaning broken install $($_.Name) ..." -ForegroundColor Yellow
  try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
}
Write-Host "Upgrading pip ..."
& $python -m pip install --upgrade pip 2>&1 | Out-Host
# Re-clean after pip upgrade (pip may have left another ~ on failure)
Get-ChildItem -LiteralPath (Join-Path $venv "Lib\site-packages") -Filter "~*" -Force -ErrorAction SilentlyContinue | ForEach-Object {
  try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
}
Write-Host "Installing torch (CPU-only, via https://download.pytorch.org/whl/cpu) ..."
& $pip install --index-url https://download.pytorch.org/whl/cpu torch --upgrade 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Write-Host "torch install failed" -ForegroundColor Red; exit 1 }

# 4. Install vocalis (handle WinError 32 when tray app is still running)
Write-Host "Installing vocalis (pip install -e .) ..."
# If Vocalis is still in the tray (close hides to tray, not quit), its exe is locked.
$lockedExe = Join-Path $venv "Scripts\vocalis.exe"
if (Test-Path $lockedExe) {
  $procs = @()
  $procs += Get-Process -Name "Vocalis" -ErrorAction SilentlyContinue
  # PowerShell 5.1 Get-Process has no CommandLine — use CIM, and guard StrictMode
  try {
    $cands = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*vocalis*" -or $_.ExecutablePath -like "*$venv*" }
    foreach ($ci in $cands) {
      try { $pr = Get-Process -Id $ci.ProcessId -ErrorAction SilentlyContinue; if ($pr) { $procs += $pr } } catch {}
    }
  } catch {}
  # Fallback: any python whose path is inside the venv
  $procs += Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*$venv*" }
  if ($procs.Count -gt 0) {
    Write-Host "Vocalis appears to be running (tray). Closing it so files can be updated..." -ForegroundColor Yellow
    foreach ($pr in $procs) {
      try { $pr.CloseMainWindow() | Out-Null; Start-Sleep -Milliseconds 500 } catch {}
    }
    Start-Sleep -Seconds 1
    foreach ($pr in $procs) {
      try { if (-not $pr.HasExited) { $pr | Stop-Process -Force -ErrorAction SilentlyContinue } } catch {}
    }
    Start-Sleep -Seconds 1
  }
  # Also close any lingering file handle by renaming stale exe if still locked
  if (Test-Path $lockedExe) {
    try {
      $fh = [System.IO.File]::Open($lockedExe, 'Open', 'Read', 'None')
      $fh.Close(); $fh.Dispose()
    } catch {
      Write-Host "vocalis.exe still locked - will rename and retry..." -ForegroundColor Yellow
      $bak = "$lockedExe.old"
      try { if (Test-Path $bak) { Remove-Item -Force $bak -ErrorAction SilentlyContinue } } catch {}
      try { Rename-Item -LiteralPath $lockedExe -NewName "vocalis.exe.old" -Force -ErrorAction SilentlyContinue; Write-Host "Renamed locked exe to vocalis.exe.old" -ForegroundColor Yellow } catch {
        Write-Host "Could not rename locked exe (will retry pip with --force): $_" -ForegroundColor Yellow
      }
    }
  }
}
$installOk = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
  Push-Location $root
  try {
    Write-Host "pip install attempt $attempt/3 ..."
    & $pip install -e . --upgrade --no-warn-script-location 2>&1 | Out-Host
    if ($LASTEXITCODE -eq 0) { $installOk = $true; break }
    Write-Host "pip failed with exit $LASTEXITCODE (attempt $attempt)" -ForegroundColor Yellow
  } finally { Pop-Location }
  if ($attempt -lt 3) {
    Write-Host "Retrying in 2s (close Vocalis tray / antivirus may hold the file)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    # Try again to free the file
    $bak = "$lockedExe.old"
    if ((Test-Path $lockedExe) -and (Test-Path $bak)) {
      try { Remove-Item -Force $bak -ErrorAction SilentlyContinue } catch {}
    }
  }
}
if (-not $installOk) {
  Write-Host "" -ForegroundColor Red
  Write-Host "pip install -e . still failed after 3 tries. Most common cause: Vocalis is still running in the tray." -ForegroundColor Red
  Write-Host "Fix: Right-click the Vocalis tray icon -> Quit, or Task Manager -> End task Vocalis/python, then re-run install.bat" -ForegroundColor Yellow
  Write-Host "Or run with -NoBinary and skip open handles: .\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation -v" -ForegroundColor Yellow
  throw "pip install -e . failed (WinError 32 - file in use). See log above."
}
# Clean up .old if install succeeded
$bak = "$lockedExe.old"
if (Test-Path $bak) { try { Remove-Item -Force $bak -ErrorAction SilentlyContinue } catch {} }

# 5. Build standalone binary (preferred distribution) - skip with -NoBinary
$binary = Join-Path $root "dist\Vocalis\Vocalis.exe"
$buildBinary = -not $NoBinary  # default ON
if ($NoBinary) { $buildBinary = $false }
if ($WithBinary) { $buildBinary = $true }
if ($buildBinary) {
  Write-Host "`nBuilding standalone binary (PyInstaller, ~1.5 GB, a few minutes) ..." -ForegroundColor Cyan
  Write-Host "This streams PyInstaller INFO logs - they are normal, not errors. Use -NoBinary to skip." -ForegroundColor DarkGray
  try {
    & $pip install --upgrade pyinstaller 2>&1 | Out-Host
    Push-Location $root
    try {
      # Use python -m PyInstaller via cmd to avoid PowerShell turning INFO stderr into NativeCommandError
      # (PowerShell with $ErrorActionPreference=Stop treats any stderr as error)
      $pyiArgs = "packaging/vocalis.spec --noconfirm --clean"
      Write-Host "Running: $venv\Scripts\python.exe -m PyInstaller $pyiArgs" -ForegroundColor DarkGray
      $psi = Start-Process -FilePath "$venv\Scripts\python.exe" -ArgumentList "-m","PyInstaller",$pyiArgs.Split(" ") -NoNewWindow -Wait -PassThru
      $pyiExit = $psi.ExitCode
      if ($pyiExit -eq 0 -and (Test-Path $binary)) {
        Write-Host "Binary built: $binary" -ForegroundColor Green
      } else {
        Write-Host "Binary build exit $pyiExit - falling back to venv script (still usable). Run manually: .\.venv\Scripts\python.exe -m PyInstaller packaging/vocalis.spec --noconfirm --clean" -ForegroundColor Yellow
        if (Test-Path "$root\build") { Write-Host "See build/warnings for details" -ForegroundColor Yellow }
        $binary = $null
      }
    } finally { Pop-Location }
  } catch {
    Write-Host "Binary build error: $_ - continuing with venv install" -ForegroundColor Yellow
    $binary = $null
  }
} else {
  Write-Host "Skipping binary build (-NoBinary)"
  $binary = $null
}

# 6. Optional self-test
if (-not $NoCheck) {
  Write-Host "`nRunning vocalis --check (models will download on first run, ~700 MB) ..."
  $toCheck = $null
  if ($binary -and (Test-Path $binary)) { $toCheck = $binary }
  else {
    $vocalis = Join-Path $venv "Scripts\vocalis.exe"
    if (Test-Path $vocalis) { $toCheck = $vocalis }
  }
  if ($toCheck) { & $toCheck --check; if ($LASTEXITCODE -ne 0) { Write-Host "check reported failures - see above; GUI will still run (models lazy-load)" -ForegroundColor Yellow } }
  else { & $python -m vocalis --check }
}

# 7. Desktop shortcut (optional) - prefers standalone binary
if (-not $NoShortcut) {
  try {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnk = Join-Path $desktop "Vocalis.lnk"
    $target = $null; $args = ""
    if ($binary -and (Test-Path $binary)) {
      $target = $binary; $args = ""
      Write-Host "Shortcut will point to standalone binary" -ForegroundColor Green
    } else {
      $target = Join-Path $venv "Scripts\vocalis.exe"
      if (-not (Test-Path $target)) { $target = Join-Path $venv "Scripts\python.exe"; $args = "-m vocalis" }
    }
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = $target; $sc.Arguments = $args; $sc.WorkingDirectory = $root
    $sc.Description = "Vocalis - voice assistant for local LLMs"
    $iconPath = Join-Path $root "packaging\icon.ico"
    $sc.IconLocation = if (Test-Path $iconPath) { $iconPath } else { $target }
    $sc.Save()
    Write-Host "Shortcut created: $lnk -> $target" -ForegroundColor Green
    $startDir = Join-Path ([Environment]::GetFolderPath("Programs")) "Vocalis"
    New-Item -ItemType Directory -Force -Path $startDir | Out-Null
    $startLnk = Join-Path $startDir "Vocalis.lnk"
    $sc2 = $shell.CreateShortcut($startLnk)
    $sc2.TargetPath = $target; $sc2.Arguments = $args; $sc2.WorkingDirectory = $root
    $sc2.Description = "Vocalis"; $sc2.IconLocation = $sc.IconLocation; $sc2.Save()
    $unLnk = Join-Path $startDir "Uninstall Vocalis.lnk"
    $sc3 = $shell.CreateShortcut($unLnk)
    $sc3.TargetPath = "powershell.exe"
    $sc3.Arguments = ('-NoProfile -ExecutionPolicy Bypass -File "{0}\scripts\uninstall.ps1"' -f $root)
    $sc3.WorkingDirectory = $root; $sc3.Description = "Uninstall Vocalis"; $sc3.Save()
  } catch {
    Write-Host "Could not create shortcuts: $_" -ForegroundColor Yellow
  }
}

Write-Host "`nDone. Launch with:" -ForegroundColor Green
if ($binary -and (Test-Path $binary)) {
  Write-Host "  dist\Vocalis\Vocalis.exe        # standalone binary (preferred, double-click)"
}
Write-Host "  .venv\Scripts\vocalis        # GUI (also: python -m vocalis)"
Write-Host "  .venv\Scripts\vocalis --check"
Write-Host "  .venv\Scripts\vocalis --headless"
Write-Host "Uninstall: Settings - Danger Zone  or  double-click uninstall.bat  (or .\scripts\uninstall.ps1 -Clean)"
Write-Host "Tip: Settings - Danger Zone has one-click Uninstall buttons (keep models / erase everything)."

