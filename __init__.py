from .nodes import (
    SeedVR2Analysis,
    SeedVR2EquivalenceAnalysis,
    SeedVR2FourWayAnalysis,
    SeedVR2DOVERFourWay,
)

NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
    "SeedVR2FourWayAnalysis": SeedVR2FourWayAnalysis,
    "SeedVR2DOVERFourWay": SeedVR2DOVERFourWay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
    "SeedVR2FourWayAnalysis": "SeedVR2 Four-Way Analysis (Rev 3: ref/floor/numz/native)",
    "SeedVR2DOVERFourWay": "SeedVR2 Four-Way Analysis (legacy class alias)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
