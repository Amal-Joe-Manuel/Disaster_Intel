Param(
  [string]$OutFile = (Join-Path $PSScriptRoot "models\fire_smoke_best.pt")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $PSScriptRoot "models"))) {
  New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot "models") | Out-Null
}

if (Test-Path $OutFile) {
  Write-Host "Already exists: $OutFile"
  Write-Host "Delete it if you want to re-download."
  exit 0
}

# Pretrained weights (Fire + Smoke) compatible with Ultralytics YOLO.
# Public GitHub URL (no authentication required).
$Url = "https://github.com/Nocluee100/Fire-and-Smoke-Detection-AI-v1/raw/main/weights/best.pt"

Write-Host "Downloading fire/smoke model..."
Write-Host "From: $Url"
Write-Host "To:   $OutFile"

Invoke-WebRequest -Uri $Url -OutFile $OutFile

$size = (Get-Item $OutFile).Length
if ($size -lt 1000000) {
  throw "Download looks too small ($size bytes)."
}

Write-Host "Done. File size: $size bytes"
