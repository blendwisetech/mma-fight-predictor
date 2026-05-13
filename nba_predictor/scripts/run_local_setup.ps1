# Run from PowerShell in nba_predictor:
#   powershell -ExecutionPolicy Bypass -File scripts\run_local_setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> pip install" -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "Done. Start UI with: streamlit run app/main.py" -ForegroundColor Green
