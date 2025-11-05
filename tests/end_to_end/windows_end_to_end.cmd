@echo off
setlocal enabledelayedexpansion


REM ============================================
REM Usage:
REM   windows_end_to_end.cmd ^
REM     <BRANCH/depthai-nodes-version> ^
REM     <HUBAI_API_KEY> ^
REM     <HUBAI_TEAM_SLUG> ^
REM     <DEPTHAI_VERSION> ^
REM     <LUXONIS_EXTRA_INDEX_URL> ^
REM     <ADDITIONAL_PARAMETER> ^
REM     <PLATFORM>
REM
REM Example:
REM   windows_end_to_end.cmd v0.8.0 abc123 myteam 3.0.0 https://idx.xxx rvc4 " --foo bar"
REM
REM This maps to:
REM   --env LUXONIS_EXTRA_INDEX_URL
REM   --env DEPTHAI_VERSION
REM   --env HUBAI_TEAM_SLUG
REM   --env HUBAI_API_KEY
REM   --env BRANCH
REM   --env FLAGS = "<ADDITIONAL_PARAMETER> -v <BRANCH> --platform <PLATFORM>"
REM ============================================

REM ---- Parse args
set "BRANCH=%~1"
set "HUBAI_API_KEY=%~2"
set "HUBAI_TEAM_SLUG=%~3"
set "DEPTHAI_VERSION=%~4"
set "LUXONIS_EXTRA_INDEX_URL=%~5"
set "PLATFORM=%~6"
set "ADDITIONAL_PARAMETER=%~7"


REM ---- Default branch if not provided
if "%BRANCH%"=="" (
  set "BRANCH=main"
  echo [*] No branch/version provided; defaulting to BRANCH=main
)

REM ---- Basic validation (others still required)
if "%~2"==""  ( echo [!] HUBAI_API_KEY is required & exit /b 2 )
if "%~3"==""  ( echo [!] HUBAI_TEAM_SLUG is required & exit /b 2 )
if "%~4"==""  ( echo [!] DEPTHAI_VERSION is required & exit /b 2 )
if "%~6"==""  ( echo [!] PLATFORM is required (e.g., rvc4) & exit /b 2 )

REM ---- Compose FLAGS
set "FLAGS=%ADDITIONAL_PARAMETER%"

REM ---- Export env
set "LUXONIS_EXTRA_INDEX_URL=%LUXONIS_EXTRA_INDEX_URL%"
set "DEPTHAI_VERSION=%DEPTHAI_VERSION%"
set "HUBAI_TEAM_SLUG=%HUBAI_TEAM_SLUG%"
set "HUBAI_API_KEY=%HUBAI_API_KEY%"
set "BRANCH=%BRANCH%"
set "FLAGS=%FLAGS%"
set "DEPTHAI_NODES_LEVEL=debug"

echo [*] Recreating %DEST%
rmdir /S /Q "%TEMP%\depthai-nodes" 2>nul
cd %TEMP%

echo [*] Cloning luxonis/depthai-nodes @ %BRANCH%
git clone -b "%BRANCH%" https://github.com/luxonis/depthai-nodes.git || (
  echo [!] git clone failed
  exit /b 4
)

cd depthai-nodes\

python.exe -m venv venv
call venv\Scripts\activate.bat
pip install -e .
pip install -r requirements-dev.txt
pip install --upgrade --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ --extra-index-url "%LUXONIS_EXTRA_INDEX_URL%"  depthai==%DEPTHAI_VERSION%

cd tests\end_to_end
for /f "delims=" %%a in ('python setup_camera_ips.py') do %%a
python -u main.py --platform %PLATFORM%