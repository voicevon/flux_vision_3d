@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title flux_vision_3d Dashboard

rem 优先使用 PATH 中的 python；若是 WindowsApps 占位 stub 或不存在，则回退到本地 SDK
set "PYCMD=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Users\feng\python-sdk\python3.13.2\python.exe" (
        set "PYCMD=C:\Users\feng\python-sdk\python3.13.2\python.exe"
    ) else (
        echo.
        echo ===============================================================================
        echo [ERROR] Python was not found in your system PATH!
        echo Please ensure Python 3.10+ is installed and added to the PATH environment variable.
        echo ===============================================================================
        echo.
        pause
        exit /b 1
    )
)

rem 依赖预检：缺失时引导用户安装（find_spec 不真正导入，毫秒级完成）
"%PYCMD%" -c "import importlib.util,sys; mods=['cv2','numpy','yaml','serial','bleak','scipy','pyrealsense2','ultralytics','paho.mqtt']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; missing and print('缺少模块: '+', '.join(missing)); sys.exit(1 if missing else 0)"
if %errorlevel% neq 0 (
    echo.
    set /p "CHOICE=[提示] 检测到缺失的 Python 依赖包，是否立即安装 (pip install -r requirements.txt)? [Y/n]: "
    if /i "!CHOICE!"=="n" exit /b 1
    "%PYCMD%" -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] 依赖安装失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

"%PYCMD%" -X utf8 tools\gui_launcher.py %*

if %errorlevel% neq 0 (
    echo.
    pause
)
endlocal
