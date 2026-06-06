@echo off
:: file-viewer-run.bat — launch the DCProgs file viewer (dcio)
::
:: Activates the dcprogs conda environment, prepends the DLL directory so
:: numpy/scipy DLLs are found, then starts the Streamlit server.
::
:: Double-click from Explorer, or run from any terminal.

setlocal

set CONDA_ROOT=C:\ProgramData\miniconda3
set ENV_NAME=dcprogs
set ENV_DIR=%CONDA_ROOT%\envs\%ENV_NAME%

:: Change to the dcio project root (where viewer/ lives).
cd /d "%~dp0"

:: Add the dcio root to PYTHONPATH so that "from viewer.registry import …"
:: resolves correctly.  Streamlit adds the script's own directory to sys.path,
:: not the project root, so without this the viewer package is not found.
set PYTHONPATH=%~dp0;%PYTHONPATH%

:: Prepend the conda Library\bin directory so numpy/scipy DLLs are found
set PATH=%ENV_DIR%\Library\bin;%PATH%

:: Activate the conda environment
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%

:: Verify streamlit is available
where streamlit >nul 2>&1
if errorlevel 1 (
    echo ERROR: streamlit not found in the "%ENV_NAME%" environment.
    echo Install it with:  conda activate %ENV_NAME%  ^&^&  pip install streamlit
    pause
    exit /b 1
)

echo Starting DCProgs file viewer ...
echo   http://localhost:8502
echo.
streamlit run viewer/app.py --server.port 8502 --server.headless false

endlocal
