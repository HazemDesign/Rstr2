# Syncs the dev addon (D:\blender\dev\Rstr2\addon) into Blender's addon
# directory and copies the core binaries next to it. Run with Blender CLOSED.
#
#   powershell -ExecutionPolicy Bypass -File tools\update-addon.ps1
#
# The viewport status line shows the running version ("Rstr2 v0.4.x") so you
# can confirm the update took effect.

$ErrorActionPreference = "Stop"

$dev     = "D:\blender\dev\Rstr2"
$srcAdd  = Join-Path $dev "addon"
$srcBin  = Join-Path $dev "bin"
$blender = "C:\Program Files\Blender Foundation\Blender 5.2"

# Keep using the existing install if 5.2 is not where we expect.
$dstRoot = Join-Path $env:APPDATA "Blender Foundation\Blender\5.2\scripts\addons"
if (-not (Test-Path $dstRoot)) {
    $latest = Get-ChildItem (Join-Path $env:APPDATA "Blender Foundation\Blender") -Directory |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "No Blender addons directory found." }
    $dstRoot = Join-Path $latest.FullName "scripts\addons"
}

$dst = Join-Path $dstRoot "rstr2"
Write-Host "Installing addon: $srcAdd -> $dst"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Path $dst | Out-Null
Copy-Item (Join-Path $srcAdd "*.py") $dst -Force

# Core binaries live in bin/ next to the addon; the addon resolves them via
# core_proc.core_exe_path(). Mirror them there too so one sync updates both.
$coreBin = Join-Path $dst "bin"
New-Item -ItemType Directory -Path $coreBin -Force | Out-Null
foreach ($f in @("Rstr2Core.exe", "optix_kernels.ptx")) {
    $s = Join-Path $srcBin $f
    if (Test-Path $s) {
        Copy-Item $s (Join-Path $coreBin $f) -Force
        Write-Host "  copied $f"
    } else {
        Write-Warning "missing in dev bin/: $f"
    }
}

# Show what version was installed.
$blinfo = Get-Content (Join-Path $dst "__init__.py") | Select-String '"version": \(([^)]+)\)'
Write-Host "Installed addon version:" $blinfo.Matches[0].Groups[1].Value
Write-Host "Done. Restart Blender (or reload the addon) to pick it up."
