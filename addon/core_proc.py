# Rstr2 - Phase 2 native core process launcher.
#
# Locates Rstr2Core.exe (bin/Rstr2Core.exe, relative to the repo root) and
# spawns it headless. All failures are swallowed so the addon keeps working
# with its test pattern when the core is unavailable.

import os
import subprocess
from pathlib import Path

# CREATE_NO_WINDOW (Win32) - launch without a console window.
CREATE_NO_WINDOW = 0x08000000

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


def _candidate_dirs():
    """Directories that may contain Rstr2Core.exe, most specific first."""
    here = Path(__file__).resolve().parent
    out = []
    # 1. bin/ inside the installed addon package itself.
    out.append(here / "bin")
    # 2. Repo layout: <repo>/bin next to the addon folder.
    out.append(here.parent / "bin")
    # 3. Explicit override.
    env = os.environ.get("RSTR2_CORE_BIN")
    if env:
        out.append(Path(env))
    return out


def _find_core(filename):
    """Return a Path to filename, or None. Paths are returned (not str) so
    callers can use .parent on the executable location."""
    try:
        for d in _candidate_dirs():
            p = d / filename
            if p.is_file():
                return p
    except Exception:
        pass
    return None


def core_exe_path():
    """Return the path to Rstr2Core.exe, or None if it is missing."""
    return _find_core("Rstr2Core.exe")


def core_ptx_path():
    """Return the path to optix_kernels.ptx next to Rstr2Core.exe, or None."""
    return _find_core("optix_kernels.ptx")


def dxr_exe_path():
    """Return the path to Rstr2Dxr.exe (DXR backend), or None.

    The DXR backend additionally requires dxr_shade.cso next to the exe; if
    that is missing we return None so callers fall back to OptiX (or a test
    pattern) instead of a hard failure.
    """
    exe = _find_core("Rstr2Dxr.exe")
    if exe is None:
        return None
    if not (exe.parent / "dxr_shade.cso").is_file():
        return None
    return exe


def core_exe_path_for(backend="optix"):
    """Return the core executable for the requested backend.

    backend: "optix" -> Rstr2Core.exe, "dxr" -> Rstr2Dxr.exe.
    Returns None if the requested binary (and its shader blob) is missing.
    """
    if backend == "dxr":
        return dxr_exe_path()
    return core_exe_path()


class CoreProcess:
    """Manages the lifetime of a single Rstr2Core.exe instance."""

    def __init__(self):
        self._proc = None
        self._log = None

    # ------------------------------------------------------------------
    def launch(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, backend="optix"):
        """Spawn the core. No-op (returns False) if already running.

        backend: "optix" (Rstr2Core.exe, NVIDIA/OptiX) or "dxr"
        (Rstr2Dxr.exe, cross-vendor DXR 1.1). If the requested backend binary
        is missing, falls back to OptiX; if that is also missing, returns
        False (the addon then uses its test pattern).

        Returns True if a new process was started, False otherwise. The
        child's stdout/stderr are redirected to bin/<exe>.log so launch
        and runtime diagnostics are visible even though it runs windowless.
        """
        if self.poll_alive():
            return False

        exe = core_exe_path_for(backend)
        if exe is None and backend == "dxr":
            # Requested DXR but the binary/shader is missing: try OptiX.
            exe = core_exe_path()
            if exe is not None:
                backend = "optix"
        if exe is None:
            return False

        log_path = exe.parent / (exe.stem + ".log")
        try:
            self._log = open(str(log_path), "wb", buffering=0)
        except Exception:
            self._log = None

        try:
            self._proc = subprocess.Popen(
                [str(exe), "--width", str(width), "--height", str(height)],
                cwd=str(exe.parent),
                creationflags=CREATE_NO_WINDOW,
                stdout=self._log,
                stderr=self._log,
            )
            return True
        except Exception:
            if self._log is not None:
                try:
                    self._log.close()
                except Exception:
                    pass
                self._log = None
            self._proc = None
            return False

    # ------------------------------------------------------------------
    def poll_alive(self):
        if self._proc is None:
            return False
        return self._proc.poll() is None

    # ------------------------------------------------------------------
    def exit_code(self):
        """Return the child's exit code if it has terminated, else None."""
        if self._proc is None:
            return None
        return self._proc.poll()

    # ------------------------------------------------------------------
    def terminate(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None
            if self._log is not None:
                try:
                    self._log.close()
                except Exception:
                    pass
                self._log = None
