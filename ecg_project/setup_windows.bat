@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [2/2] Installing the local ECG-Digitiser nnU-Net fork...
python -m pip install -e .\nnUNet
if errorlevel 1 goto :error

echo.
echo Setup completed successfully.
echo Put the M3 model in models\M3\ before digitizing ECG images.
pause
exit /b 0

:error
echo.
echo Setup failed. Check the error messages above.
pause
exit /b 1
