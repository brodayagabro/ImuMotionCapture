param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$SpecPath = Join-Path $PSScriptRoot "pyqt_mocap_gl.spec"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $Python = Get-Command python -ErrorAction SilentlyContinue

    if ($PyLauncher) {
        & $PyLauncher.Source -3 -m venv $VenvPath
    }
    elseif ($Python) {
        & $Python.Source -m venv $VenvPath
    }
    else {
        throw "Python 3.11 or newer was not found. Install Python and run this script again."
    }
}

Push-Location $ProjectRoot
try {
    if (-not $SkipInstall) {
        & $PythonPath -m pip install --upgrade pip
        & $PythonPath -m pip install -e ".[build]"
    }

    $OldSingleFilePath = Join-Path $ProjectRoot "dist\pyqt_mocap_gl.exe"
    if (Test-Path -LiteralPath $OldSingleFilePath) {
        Remove-Item -LiteralPath $OldSingleFilePath -Force
    }

    & $PythonPath -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }

    $ExecutablePath = Join-Path $ProjectRoot "dist\pyqt_mocap_gl\pyqt_mocap_gl.exe"
    if (-not (Test-Path -LiteralPath $ExecutablePath)) {
        throw "Build completed without the expected executable: $ExecutablePath"
    }

    Write-Host "Windows OpenGL build created: $ExecutablePath"
}
finally {
    Pop-Location
}
