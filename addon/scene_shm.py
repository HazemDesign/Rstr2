# Rstr2 - Phase 3 scene shared-memory bridge (writer side, addon -> core).
#
# Creates the named mapping "Local\\Rstr2Scene_v1" and publishes the Blender
# scene (world-space triangle soup + camera basis) that the native core
# ray-traces. Protocol matches core/src/shared_mem.h exactly (version 2):
#
#   Header (256 bytes, little-endian, _pack_=4):
#     u32 magic        = 0x32525353
#     u32 version      = 2
#     u32 epoch        (incremented LAST on each update)
#     u32 ready        (1 = valid)
#     u32 writing      (1 while we are mid-update)
#   u32 vertex_count (xyz triples)
#   u32 index_count  (uint32 indices)
#   u32 light_count  (typed lights, 16 floats each)
#   u32 flags        (bit0 = TAA enabled)
#   float exposure, taa_history
#   float cam_origin[3], cam_right[3], cam_up[3], cam_forward[3]
#   float cam_tan_half_fov_y
#   (padded to 256)
#   offset 256: vertices  (vertex_count * 3 * float32, world space)
#   then      : indices   (index_count * uint32)
#   then      : lights    (light_count * 16 * float32 - see below)
#   then      : albedos   (vertex_count * 3 * float32, linear RGB per vertex)
#
# Typed-light row layout (16 floats / 64 bytes):
#   [px,py,pz, type,
#    dx,dy,dz, intensity,
#    cr,cg,cb, size_x, size_y,
#    ax,ay,az]
#   type: 0 point, 1 sun, 2 spot, 3 area.
#   spot: size_x = cos(outer half angle), size_y = cos(inner half angle).
#   area: extents along axis (ax..az) and cross(dir, axis).

import ctypes
import struct

import numpy as np

# --- Win32 bindings -------------------------------------------------------
_kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0x000F001F
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_CreateFileMappingW = _kernel32.CreateFileMappingW
_CreateFileMappingW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_wchar_p]
_CreateFileMappingW.restype = ctypes.c_void_p

_MapViewOfFile = _kernel32.MapViewOfFile
_MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
_MapViewOfFile.restype = ctypes.c_void_p

_UnmapViewOfFile = _kernel32.UnmapViewOfFile
_UnmapViewOfFile.argtypes = [ctypes.c_void_p]
_UnmapViewOfFile.restype = ctypes.c_int

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [ctypes.c_void_p]
_CloseHandle.restype = ctypes.c_int

MAPPING_NAME = "Local\\Rstr2Scene_v1"
MAGIC = 0x32525353
VERSION = 2
HEADER_SIZE = 256
LIGHT_FLOATS = 16
FLAG_TAA = 1
FLAG_FILM_TRANSPARENT = 2
MAX_SCENE_BYTES = 64 * 1024 * 1024  # 64 MB cap


class _SceneHeader(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("epoch", ctypes.c_uint32),
        ("ready", ctypes.c_uint32),
        ("writing", ctypes.c_uint32),
        ("vertex_count", ctypes.c_uint32),
        ("index_count", ctypes.c_uint32),
        ("light_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("exposure", ctypes.c_float),
        ("taa_history", ctypes.c_float),
        ("cam_origin", ctypes.c_float * 3),
        ("cam_right", ctypes.c_float * 3),
        ("cam_up", ctypes.c_float * 3),
        ("cam_forward", ctypes.c_float * 3),
        ("cam_tan_half_fov_y", ctypes.c_float),
        ("reserved", ctypes.c_uint8 * (HEADER_SIZE - (9 * 4 + 15 * 4))),
    ]


class SceneWriter:
    """Creates the scene mapping and publishes scene updates. Never raises on
    allocation failure; write() simply returns False so the addon keeps its
    test-pattern fallback."""

    def __init__(self, name=MAPPING_NAME, max_bytes=MAX_SCENE_BYTES):
        self._handle = None
        self._addr = None
        self._max_bytes = max_bytes
        self._epoch = 0
        self._last_sig = None

        self._create(name)
    # ------------------------------------------------------------------
    def _create(self, name):
        try:
            size = HEADER_SIZE + self._max_bytes
            handle = _CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE,
                                         0, ctypes.c_uint32(size), name)
            if not handle or handle == INVALID_HANDLE_VALUE:
                return
            view = _MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size)
            if not view:
                _CloseHandle(handle)
                return
            self._handle = handle
            self._addr = view
            # Initialize header.
            hdr = _SceneHeader.from_buffer_copy(
                ctypes.string_at(view, HEADER_SIZE))
            hdr.magic = MAGIC
            hdr.version = VERSION
            hdr.epoch = 0
            hdr.ready = 0
            hdr.writing = 0
            hdr.vertex_count = 0
            hdr.index_count = 0
            ctypes.memmove(view, ctypes.byref(hdr), HEADER_SIZE)
        except Exception:
            self.close()

    # ------------------------------------------------------------------
    def is_open(self):
        return self._addr is not None

    # ------------------------------------------------------------------
    def write(self, vertices, indices, camera, lights=None, albedos=None,
              settings=None):
        """vertices: (n,3) float32 world space; indices: (m,) uint32;
        camera: dict with origin/right/up/forward (each (3,) float) and
        tan_half_fov_y float;
        lights: optional (L,16) float32 typed-light rows
        (px,py,pz,type,dx,dy,dz,intensity,cr,cg,cb,size_x,size_y,ax,ay,az);
        albedos: optional (n,3) float32 linear RGB per vertex;
        settings: optional dict {flags:int, exposure:float, history:float,
        world:[r,g,b,strength] (uniform env light, 0 strength disables)}.
        Returns True if published."""
        if self._addr is None:
            return False
        try:
            vcount = int(vertices.shape[0]) if vertices.size else 0
            icount = int(indices.shape[0]) if indices.size else 0
            if vcount == 0 or icount == 0 or (icount % 3) != 0:
                return False

            lcount = 0
            lbuf = b""
            lbytes = 0
            if lights is not None and lights.size:
                lrows = np.ascontiguousarray(lights, dtype=np.float32).reshape(-1, LIGHT_FLOATS)
                lcount = int(lrows.shape[0])
                lbuf = lrows.ravel().tobytes()
                lbytes = len(lbuf)

            abuf = b""
            abytes = 0
            # NOTE: the core unconditionally expects one rgb triple per vertex
            # after the lights, so we ALWAYS publish the block - substituting
            # the kernel's default albedo when the caller gave none.
            if albedos is not None and albedos.size:
                arows = np.ascontiguousarray(albedos, dtype=np.float32).reshape(-1, 3)
                if int(arows.shape[0]) != vcount:
                    arows = np.full((vcount, 3), (0.8, 0.8, 0.82), dtype=np.float32)
            else:
                arows = np.full((vcount, 3), (0.8, 0.8, 0.82), dtype=np.float32)
            abuf = arows.ravel().tobytes()
            abytes = len(abuf)

            sflags = FLAG_TAA
            sexposure = 1.0
            shistory = 20.0
            sworld = (0.0, 0.0, 0.0, 0.0)
            if settings:
                sflags = int(settings.get("flags", sflags))
                sexposure = float(settings.get("exposure", sexposure))
                shistory = float(settings.get("history", shistory))
                wv = settings.get("world")
                if wv is not None and len(wv) == 4:
                    sworld = tuple(float(x) for x in wv)

            vbytes = vcount * 3 * 4
            ibytes = icount * 4
            if vbytes + ibytes + lbytes + abytes > self._max_bytes:
                return False

            hdr = _SceneHeader.from_buffer_copy(ctypes.string_at(self._addr, HEADER_SIZE))
            # Mark writing so the reader skips a torn update.
            hdr.writing = 1
            hdr.ready = 0
            ctypes.memmove(self._addr, ctypes.byref(hdr), HEADER_SIZE)

            # Copy vertices, indices, lights, albedos into the view.
            vbuf = np.ascontiguousarray(vertices, dtype=np.float32).ravel().tobytes()
            ibuf = np.ascontiguousarray(indices, dtype=np.uint32).ravel().tobytes()
            cam_key = (
                tuple(float(x) for x in camera["origin"]),
                tuple(float(x) for x in camera["right"]),
                tuple(float(x) for x in camera["up"]),
                tuple(float(x) for x in camera["forward"]),
                float(camera["tan_half_fov_y"]),
            )
            sig = (vbuf, ibuf, lbuf, abuf, cam_key,
                   sflags, sexposure, shistory, sworld)
            if sig == self._last_sig:
                return True
            self._last_sig = sig

            vdst = self._addr + HEADER_SIZE
            idst = self._addr + HEADER_SIZE + vbytes
            ldst = idst + ibytes
            adst = ldst + lbytes
            ctypes.memmove(vdst, vbuf, vbytes)
            ctypes.memmove(idst, ibuf, ibytes)
            if lbytes:
                ctypes.memmove(ldst, lbuf, lbytes)
            if abytes:
                ctypes.memmove(adst, abuf, abytes)

            # Fill header + camera, bump epoch LAST, clear writing.
            hdr.vertex_count = ctypes.c_uint32(vcount)
            hdr.index_count = ctypes.c_uint32(icount)
            hdr.light_count = ctypes.c_uint32(lcount)
            hdr.flags = ctypes.c_uint32(sflags)
            hdr.exposure = ctypes.c_float(sexposure)
            hdr.taa_history = ctypes.c_float(shistory)
            # Reserved-area addendum: world color + strength (4 floats).
            struct.pack_into("<4f", hdr.reserved, 0,
                             sworld[0], sworld[1], sworld[2], sworld[3])
            for i in range(3):
                hdr.cam_origin[i] = camera["origin"][i]
                hdr.cam_right[i] = camera["right"][i]
                hdr.cam_up[i] = camera["up"][i]
                hdr.cam_forward[i] = camera["forward"][i]
            hdr.cam_tan_half_fov_y = camera["tan_half_fov_y"]
            self._epoch += 1
            hdr.epoch = ctypes.c_uint32(self._epoch)
            hdr.writing = 0
            hdr.ready = 1
            ctypes.memmove(self._addr, ctypes.byref(hdr), HEADER_SIZE)
            return True
        except Exception:
            return False

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
