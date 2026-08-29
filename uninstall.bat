@echo off
REM One-click uninstall — double-click this file.
REM By default keeps config/models; use uninstall.bat -Clean to nuke everything.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\uninstall.ps1" %*
if errorlevel 1 pause
