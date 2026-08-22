# Rstr2 - Phase 2 shared-memory frame bridge (reader side).
#
# Reads frames produced by the native Rstr2Core.exe via a Win32 named
# file mapping "Local\\Rstr2Frame_v1".
#
# Protocol (little-endian, header packed @0):
#   u32 magic       = 0x32525352
#   u32 version     = 1
#   i32 width
#   i32 height
#   u32 frame_index
#   u32 state       (1 = ready)
#   u64 pixel_offset = 256
#   u64 pixel_size
# Pixels: RGBA32F, linear, row-major, FIRST ROW = TOP, alpha = 1.
#
# The reader NEVER raises on a missing or malformed mapping: every entry
# point returns None / False so a broken or absent core cannot disturb the
# Blender addon.

import ctypes
import struct

import numpy as np

# --- Win32 bindings -------------------------------------------------------
_kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

FILE_MAP_READ = 0x0004
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_OpenFileMappingW = _kernel32.OpenFileMappingW
_OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
_OpenFileMappingW.restype = ctypes.c_void_p

_MapViewOfFile = _kernel32.MapViewOfFile
_MapViewOfFile.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_size_t,
]
_MapViewOfFile.restype = ctypes.c_void_p

_UnmapViewOfFile = _kernel32.UnmapViewOfFile
_UnmapViewOfFile.argtypes = [ctypes.c_void_p]
_UnmapViewOfFile.restype = ctypes.c_int

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [ctypes.c_void_p]
_CloseHandle.restype = ctypes.c_int

MAPPING_NAME = "Local\\Rstr2Frame_v1"
MAGIC = 0x32525352
VERSION = 1
HEADER_SIZE = 40  # packed size of the 8 header fields
MAX_PIXEL_BYTES = 64 * 1024 * 1024  # 64 MB safety cap


class CoreFrameHeader(ctypes.Structure):
    """Mirror of the on-wire header (packed, no padding)."""

    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("frame_index", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("pixel_offset", ctypes.c_uint64),
        ("pixel_size", ctypes.c_uint64),
    ]


class CoreFrameReader:
    """Opens the named mapping and copies out individual frames.

    Usage:
        reader = CoreFrameReader()
        if reader.open():
            frame = reader.read_frame()   # (frame_index, ndarray) or None
        reader.close()
    """

    def __init__(self):
        self._handle = None
        self._addr = None  # integer base address of the mapped view

    # ------------------------------------------------------------------
    def open(self):
        """Attach to the shared mapping. Returns False if it is absent."""
        self.close()
        try:
            handle = _OpenFileMappingW(FILE_MAP_READ, 0, MAPPING_NAME)
            if not handle or handle == INVALID_HANDLE_VALUE:
                return False
            view = _MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0)
            if not view:
                _CloseHandle(handle)
                return False
            self._handle = handle
            self._addr = view
            return True
        except Exception:
            self.close()
            return False

    # ------------------------------------------------------------------
    def is_open(self):
        return self._addr is not None

    # ------------------------------------------------------------------
    def read_frame(self):
        """Return (frame_index, ndarray(h, w, 4) float32) or None.

        Uses the read/copy/re-read consistency trick: read the frame index,
        copy the pixels, then re-read the index. If it changed we retry up to
        3 times; if it is still changing we give up this tick and return None.
        """
        if self._addr is None:
            return None

        for _ in range(4):  # initial attempt + up to 3 retries
            try:
                hdr = self._read_header()
                if hdr is None:
                    return None

                w = hdr.width
                h = hdr.height
                if w <= 0 or h <= 0:
                    return None

                pix_bytes = w * h * 4 * 4  # RGBA32F
                if pix_bytes > MAX_PIXEL_BYTES:
                    return None

                offset = hdr.pixel_offset
                if offset < HEADER_SIZE:
                    return None

                idx1 = hdr.frame_index

                # Copy the pixel block out of the mapping into fresh bytes.
                raw = ctypes.string_at(self._addr + offset, pix_bytes)
                if len(raw) != pix_bytes:
                    return None

                # Re-read the index to detect a mid-copy update.
                hdr2 = self._read_header()
                if hdr2 is None:
                    return None
                if idx1 == hdr2.frame_index:
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, 4)
                    # Return a true copy so callers never alias the mapping.
                    return (idx1, np.array(arr, dtype=np.float32))
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    def _read_header(self):
        try:
            raw = ctypes.string_at(self._addr, HEADER_SIZE)
            if len(raw) != HEADER_SIZE:
                return None
            hdr = CoreFrameHeader.from_buffer_copy(raw)
            if hdr.magic != MAGIC or hdr.version != VERSION:
                return None
            return hdr
        except Exception:
            return None

    # ------------------------------------------------------------------
    def close(self):
        if self._addr is not None:
            try:
                _UnmapViewOfFile(self._addr)
            except Exception:
                pass
            self._addr = None
        if self._handle is not None:
            try:
                _CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None
