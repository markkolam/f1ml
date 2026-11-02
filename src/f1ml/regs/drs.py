"""DRS era (2011-2025): a drag-reduction flap available within 1 s at detection points."""

from __future__ import annotations

from f1ml.regs.base import RegulationModule, register


@register("drs")
class DRSModule(RegulationModule):
    # Speed-trap speeds let later phases estimate the straight-line advantage of DRS.
    required_lap_features = ("speed_st", "speed_fl")
