@echo off
setlocal
set PYTHONPATH=%~dp0src
set ANALYSIS_LLM_MODE=hf
set ANALYSIS_JOB_STORE_BACKEND=memory
set ANALYSIS_HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
set ANALYSIS_HF_ADAPTER_PATH=training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1
set ANALYSIS_HF_DEVICE=auto
set ANALYSIS_HF_LOAD_IN_4BIT=false
set ANALYSIS_HF_MAX_NEW_TOKENS=220
set ANALYSIS_HF_BATCH_SIZE=1
set ANALYSIS_HF_RETRY_MAX_NEW_TOKENS=320
set ANALYSIS_HF_REPAIR_MAX_NEW_TOKENS=220
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m uvicorn interview_analysis.main:app --app-dir src --host 127.0.0.1 --port 8000
) else (
  python -m uvicorn interview_analysis.main:app --app-dir src --host 127.0.0.1 --port 8000
)
endlocal
