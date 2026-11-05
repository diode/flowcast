"""CARLA's own SUMO co-simulation bridge (tools/CARLA_0.9.16/Co-Simulation/Sumo), used unchanged: SUMO moves
the vehicles, the bridge mirrors each one into CARLA every step. The helpers here only adapt it:

  import_cosim     the bridge's classes, from the CARLA folder (tools.carla_root)
  use_blueprints   draw our vehicle types with chosen CARLA models (the bridge cannot map 'bus' itself)
  connect          the bridge's CarlaSimulation with a longer client timeout (its own 2 s is too short for a
                   first get_world())
  detach           take CARLA out of the run (its vehicles, sync mode); SUMO keeps going
"""
from __future__ import annotations

import importlib
import logging
import sys
import types

import numpy as np

from .network import ensure_sumo
from .tools import carla_root

log = logging.getLogger(__name__)

# SUMO vehicle type (= our category) -> CARLA models it is drawn as.
BLUEPRINTS = {
    "car": ["vehicle.audi.a2", "vehicle.nissan.micra", "vehicle.citroen.c3", "vehicle.toyota.prius",
            "vehicle.seat.leon", "vehicle.mini.cooper_s", "vehicle.tesla.model3", "vehicle.chevrolet.impala",
            "vehicle.nissan.patrol", "vehicle.lincoln.mkz_2020"],
    "two_wheeler": ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider",
                    "vehicle.vespa.zx125"],
    "auto_rickshaw": ["vehicle.micro.microlino"],             # CARLA has no auto-rickshaw; closest in size
    "bus": ["vehicle.mitsubishi.fusorosa"],
    "truck": ["vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv"],
}


def import_cosim():
    """(carla, SimulationSynchronization, SumoSimulation, CarlaSimulation, BridgeHelper)."""
    ensure_sumo()
    cosim = carla_root() / "Co-Simulation" / "Sumo"
    if str(cosim) not in sys.path:
        sys.path.insert(0, str(cosim))
    import carla

    sync = importlib.import_module("run_synchronization")
    from sumo_integration.bridge_helper import BridgeHelper
    from sumo_integration.carla_simulation import CarlaSimulation
    from sumo_integration.sumo_simulation import SumoSimulation

    if not getattr(SumoSimulation, "_netstudy_patched", False):
        # A vehicle that departs and arrives in the same step is in both lists; the bridge would subscribe to it
        # after SUMO removed it, and SUMO quits on the error ("Vehicle 't3166' is not known").
        tick = SumoSimulation.tick

        def safe_tick(self):
            tick(self)
            self.spawned_actors -= self.destroyed_actors

        SumoSimulation.tick, SumoSimulation._netstudy_patched = safe_tick, True
    return carla, sync.SimulationSynchronization, SumoSimulation, CarlaSimulation, BridgeHelper


def use_blueprints(BridgeHelper, rng: np.random.Generator):
    original = BridgeHelper.__dict__["get_carla_blueprint"].__func__

    def get_carla_blueprint(sumo_actor, sync_color=False):
        ids = BLUEPRINTS.get(sumo_actor.type_id)
        if ids:
            sumo_actor = sumo_actor._replace(type_id=ids[rng.integers(len(ids))])
        return original(sumo_actor, sync_color)

    BridgeHelper.get_carla_blueprint = staticmethod(get_carla_blueprint)


def connect(CarlaSimulation, host: str, port: int, step: float, min_timeout: float = 60.0):
    mod = sys.modules[CarlaSimulation.__module__]
    real = mod.carla

    class _Client:
        def __init__(self, *args):
            self._c = real.Client(*args)

        def set_timeout(self, seconds):
            self._c.set_timeout(max(float(seconds), min_timeout))

        def __getattr__(self, name):
            return getattr(self._c, name)

    mod.carla = types.SimpleNamespace(**{**vars(real), "Client": _Client})
    try:
        return CarlaSimulation(host, port, step)
    finally:
        mod.carla = real


def detach(sync, traci):
    c = sync.carla
    try:
        settings = c.world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        c.world.apply_settings(settings)
        on_road = set(traci.vehicle.getIDList())
        for sumo_id, carla_id in sync.sumo2carla_ids.items():
            c.destroy_actor(carla_id)
            if sumo_id in on_road:
                traci.vehicle.unsubscribe(sumo_id)
        c.close()
    except Exception:  # noqa: BLE001 - best effort cleanup of the CARLA world
        log.exception("CARLA cleanup failed")
