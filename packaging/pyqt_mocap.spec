from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve().parent


analysis = Analysis(
    [str(PROJECT_ROOT / "packaging" / "pyqt_mocap_entry.py")],
    pathex=[
        str(PROJECT_ROOT / "host" / "viz"),
        str(PROJECT_ROOT / "host"),
    ],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "runtime_hook_qt.py")],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# The build host also contains Poppler's ICU 78 on PATH. PyInstaller detects
# that unrelated icuuc.dll as a Qt dependency, but it exports version-suffixed
# symbols and makes QtCore fail with ERROR_PROC_NOT_FOUND. Qt 6 uses the ICU
# compatibility library provided by supported Windows versions instead.
incompatible_icu = {"icuuc.dll", "icudt78.dll"}
analysis.binaries = [
    entry
    for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in incompatible_icu
]

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="pyqt_mocap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pyqt_mocap",
)
