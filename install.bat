@echo off
REM One-click installer — double-click this file.
REM Runs scripts/install.ps1 with Bypass so no admin / no policy change needed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
if errorlevel 1 pause
