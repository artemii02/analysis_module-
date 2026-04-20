@echo off
setlocal
set PYTHONPATH=%~dp0src
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m interview_analysis.cli %*
) else (
  python -m interview_analysis.cli %*
)
endlocal
