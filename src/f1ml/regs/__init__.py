from f1ml.regs.base import RegulationModule, build_plugins, register, registered_plugins

# Importing the modules registers them.
from f1ml.regs import active_aero_energy, drs  # noqa: F401,E402

__all__ = ["RegulationModule", "build_plugins", "register", "registered_plugins"]
