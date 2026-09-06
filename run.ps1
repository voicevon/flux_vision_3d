# flux_vision_3d PowerShell 启动入口
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python was not found in your system PATH!" -ForegroundColor Red
    Write-Host "Please ensure Python 3.10+ is installed and added to the PATH environment variable."
    pause
    exit 1
}

python -X utf8 tools\cli_menu.py $args
