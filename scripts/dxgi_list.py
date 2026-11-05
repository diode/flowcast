"""List DXGI adapters in EnumAdapters1 order (what Unreal's -graphicsadapter=N indexes)."""
from netstudy.dxgi import adapters

for a in adapters():
    print(a["index"], a["name"], f"{a['mib']} MiB", f"luid={a['luid']}")
