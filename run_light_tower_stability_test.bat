@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "MF=C:\Users\Admin\mambaforge"
set "TEST_ENV=env_ultralytics"
set "TEST_EXIT=2"

echo Lights-only PATLITE endurance test
echo Close the inspection application so only this test controls the tower.

if not exist "%MF%\condabin\conda.bat" (
    echo ERROR: Conda launcher not found: "%MF%\condabin\conda.bat"
    echo Update MF in this batch file if Conda has moved.
    goto :finish
)
if not exist "%~dp0light_tower_stability_test.py" (
    echo ERROR: light_tower_stability_test.py must be beside this batch file.
    goto :finish
)
if not exist "%~dp0LR6main.py" (
    echo ERROR: LR6main.py must be beside this batch file.
    goto :finish
)

call "%MF%\condabin\conda.bat" activate "%TEST_ENV%"
if errorlevel 1 (
    echo ERROR: Could not activate Conda environment "%TEST_ENV%".
    goto :finish
)
python -c "import sys; import usb.core; print('Python: ' + sys.executable)"
if errorlevel 1 (
    echo ERROR: Python and PyUSB must be available in "%TEST_ENV%".
    goto :finish
)

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not open the repository folder.
    goto :finish
)
python -u "%~dp0light_tower_stability_test.py" %*
set "TEST_EXIT=%ERRORLEVEL%"
popd

:finish
echo.
echo Test exit code: %TEST_EXIT%
echo 0 = completed without command errors; 1 = completed with command errors
echo 2 = setup error or aborted; 130 = interrupted
echo Logs: "%~dp0logs\light_tower_stability"
pause
exit /b %TEST_EXIT%
