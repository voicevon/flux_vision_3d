# flux_vision_3d PowerShell 启动入口
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 优先使用 PATH 中的 python；若是 WindowsApps 占位 stub 或不存在，则回退到本地 SDK
$python = "python"
try {
    & python --version 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "stub" }
} catch {
    $sdk = "C:\Users\feng\python-sdk\python3.13.2\python.exe"
    if (Test-Path $sdk) {
        $python = $sdk
    } else {
        Write-Host "[ERROR] Python was not found in your system PATH!" -ForegroundColor Red
        Write-Host "Please ensure Python 3.10+ is installed and added to the PATH environment variable."
        pause
        exit 1
    }
}

# 依赖预检：缺失时引导用户安装（find_spec 不真正导入，毫秒级完成）
& $python -c "import importlib.util,sys; mods=['cv2','numpy','yaml','serial','bleak','scipy','pyrealsense2','ultralytics','paho.mqtt']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; missing and print('缺少模块: '+', '.join(missing)); sys.exit(1 if missing else 0)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    $choice = Read-Host "[提示] 检测到缺失的 Python 依赖包，是否立即安装 (pip install -r requirements.txt)? [Y/n]"
    if ($choice -eq 'n' -or $choice -eq 'N') { exit 1 }
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 依赖安装失败，请检查网络后重试。" -ForegroundColor Red
        pause
        exit 1
    }
}

& $python -X utf8 tools\gui_launcher.py $args
