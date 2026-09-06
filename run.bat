@echo off
setlocal
cd /d "%~dp0"
title flux_vision_3d Control Terminal

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ===============================================================================
    echo [ERROR] Python was not found in your system PATH!
    echo Please ensure Python 3.10+ is installed and added to the PATH environment variable.
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

python -X utf8 tools\cli_menu.py %*

if %errorlevel% neq 0 (
    echo.
    pause
)
endlocal
