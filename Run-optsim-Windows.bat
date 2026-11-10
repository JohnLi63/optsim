@echo off
rem optsim one-click launcher for Windows. First run installs Python packages (needs internet, ~5 min).
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\optsim\venv"
if exist "%VENV%\Scripts\pythonw.exe" goto run

echo First run: setting up the Python environment. This needs internet and takes a few minutes...
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python 3 not found. Installing Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
%PY% -m venv "%VENV%" || goto fail
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto fail

:run
start "optsim" /D "%~dp0" "%VENV%\Scripts\pythonw.exe" -m optsim_gui
exit /b 0

:fail
echo.
echo Setup failed. Check the internet connection, then run this file again.
echo To start over, delete the folder %LOCALAPPDATA%\optsim
pause
