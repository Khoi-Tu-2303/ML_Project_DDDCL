# Start Jupyter with process-level reproducibility settings before kernels exist.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "Create the project .venv and install requirements.txt first."
}
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
Push-Location $projectRoot
try {
    & $pythonPath -m jupyterlab
    if ($LASTEXITCODE -ne 0) { throw "Jupyter exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
