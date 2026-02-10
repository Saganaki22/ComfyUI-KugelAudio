@echo off
REM ComfyUI-KugelAudio Safe Reinstall (No Dependencies)
REM Use this to update kugelaudio-open code without risking breaking your environment
REM This is the SAFE way to reinstall when you've made code changes

echo ============================================
echo ComfyUI-KugelAudio Safe Reinstall
echo ============================================
echo.
echo This will reinstall kugelaudio-open in editable mode
echo WITHOUT touching any dependencies (safe mode).
echo.
echo Use this when:
echo  - You modified code in kugelaudio-open folder
echo  - You want changes to take effect
echo  - You don't want to risk breaking dependencies
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PACKAGE_DIR=%SCRIPT_DIR%kugelaudio-open"

REM Navigate to script directory
cd /d "%SCRIPT_DIR%"

REM Find python_embeded
set "PYTHON_PATH=%SCRIPT_DIR%..\..\..\python_embeded\python.exe"

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at: %PYTHON_PATH%
    pause
    exit /b 1
)

echo Reinstalling kugelaudio-open (editable, no deps)...
echo Python: %PYTHON_PATH%
echo Package: %PACKAGE_DIR%
echo.

"%PYTHON_PATH%" -m pip install --no-deps --force-reinstall -e "%PACKAGE_DIR%"

echo.
if %errorlevel% equ 0 (
    echo ============================================
    echo Reinstall completed successfully!
    echo ============================================
    echo.
    echo Code changes will take effect after
    echo restarting ComfyUI.
    echo.
    echo Your dependencies are safe!
    echo.
) else (
    echo ============================================
    echo Reinstall failed!
    echo ============================================
)

echo.
pause
exit /b %errorlevel%
