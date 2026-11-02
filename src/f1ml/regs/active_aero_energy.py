"""2026+ era: active aero (Straight Mode zones), no DRS, high electrical share.

Buscombe's 2026 observations this module will eventually encode:
- deployment differs by zone (e.g. 350 kW vs 250 kW zones at Mexico) and cars
  "clip" (de-rate) late on long straights when the battery empties;
- a tow saves battery as well as drag (Monza);
- energy-starved tracks favour some PUs (Mercedes "great on straights and energy-starved tracks").
"""

from __future__ import annotations

from f1ml.regs.base import RegulationModule, register


@register("active_aero_energy")
class ActiveAeroEnergyModule(RegulationModule):
    # Speed-trap vs finish-line speeds are the public proxy for end-of-straight clipping.
    required_lap_features = ("speed_st", "speed_fl", "speed_i1", "speed_i2")
