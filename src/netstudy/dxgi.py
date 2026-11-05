"""Windows' DXGI adapters in EnumAdapters1 order: what Unreal's (CARLA's) -graphicsadapter=N indexes. The order
can change between boots (it has, on a two-GPU machine), so CARLA's adapter is
looked up by name each time (docs/gpu-tests.md), never fixed."""
import ctypes
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]


class LUID(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]


class DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID), ("Flags", ctypes.c_uint)]


IID_IDXGIFactory1 = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))


def vcall(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def adapters() -> list[dict]:
    """[{index, name, mib}] in EnumAdapters1 order."""
    factory = ctypes.c_void_p()
    ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
    enum1 = vcall(factory, 12, ctypes.HRESULT, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))   # EnumAdapters1
    out, i = [], 0
    while True:
        ad = ctypes.c_void_p()
        try:
            enum1(factory, i, ctypes.byref(ad))
        except OSError:
            break
        d = DESC1()
        vcall(ad, 10, ctypes.HRESULT, ctypes.POINTER(DESC1))(ad, ctypes.byref(d))                # GetDesc1
        out.append({"index": i, "name": d.Description, "mib": d.DedicatedVideoMemory // 2**20,
                    "luid": f"{d.AdapterLuid.high:x}:{d.AdapterLuid.low:x}"})
        i += 1
    return out


def adapter_index(name: str) -> int:
    """The -graphicsadapter index of the card whose name contains `name` (part of the name Windows lists)."""
    found = [a for a in adapters() if name.lower() in a["name"].lower()]
    if len(found) != 1:
        raise SystemExit(f"graphics adapter {name!r}: {len(found)} matches in "
                         f"{[a['name'] for a in adapters()]} (set recording.carla_gpu)")
    return found[0]["index"]
