# Rstr2 - Phase 2 native core process launcher.
#
# Locates Rstr2Core.exe (bin/Rstr2Core.exe, relative to the repo root) and
# spawns it headless. All failures are swallowed so the addon keeps working
# with its test pattern when the core is unavailable.

import subprocess
from pathlib import Path

# CREATE_NO_WINDOW (Win32) - launch without a console window.
CREATE_NO_WINDOW = 0x08000000

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


def core_exe_path():
    """Return the path to Rstr2Core.exe, or None if it is missing."""
    try:
        p = Path(__file__).resolve().parents[1] / "bin" / "Rstr2Core.exe"
        if p.is_file():
            return p
    except Exception:
        pass
    return None


class CoreProcess:
    """Manages the lifetime of a single Rstr2Core.exe instance."""

    def __init__(self):
        self._proc = None

    # ------------------------------------------------------------------
    def launch(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """Spawn the core. No-op (returns False) if already running.

        Returns True if a new process was started, False otherwise.
        """
        if self.poll_alive():
            return False

        exe = core_exe_path()
        if exe is None:
            return False

        try:
            self._proc = subprocess.Popen(
                [str(exe), "--width", str(width), "--height", str(height)],
                creationflags=CREATE_NO_WINDOW,
            )
            return True
        except Exception:
            self._proc = None
            return False

    # ------------------------------------------------------------------
    def poll_alive(self):
        if self._proc is None:
            return False
        return self._proc.poll() is None

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
