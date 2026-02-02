@echo off
REM ComfyUI-KugelAudio Portable Installation Script
REM Run this from ComfyUI/custom_nodes/ComfyUI-KugelAudio folder
REM This script installs the bundled kugelaudio-open package using portable Python

echo ============================================
echo ComfyUI-KugelAudio Portable Installation
echo ============================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PACKAGE_DIR=%SCRIPT_DIR%kugelaudio-open"

REM Navigate to script directory
cd /d "%SCRIPT_DIR%"

REM Check if kugelaudio-open folder exists
if not exist "%PACKAGE_DIR%" (
    echo ERROR: kugelaudio-open folder not found!
    echo Please run this script from ComfyUI/custom_nodes/ComfyUI-KugelAudio
    pause
    exit /b 1
)

REM Find python_embeded relative to this script (going up 3 directories)
set "PYTHON_PATH=%SCRIPT_DIR%..\..\..\python_embeded\python.exe"

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at: %PYTHON_PATH%
    echo.
    echo Make sure this script is in:
    echo ComfyUI/custom_nodes/ComfyUI-KugelAudio/
    echo.
    pause
    exit /b 1
)

echo Installing kugelaudio-open package...
echo Python: %PYTHON_PATH%
echo Package: %PACKAGE_DIR%
echo.

"%PYTHON_PATH%" -m pip install "%PACKAGE_DIR%"

echo.
if %errorlevel% equ 0 (
    echo ============================================
    echo Installation completed successfully!
    echo ============================================
) else (
    echo ============================================
    echo Installation failed!
    echo ============================================
)

echo.
pause
exit /b %errorlevel%
