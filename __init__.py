from .nodes import (
    SeedVR2Analysis,
    SeedVR2EquivalenceAnalysis,
    SeedVR2WorstFrameFidelityAnalysis,
)

NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
    "SeedVR2WorstFrameFidelityAnalysis": SeedVR2WorstFrameFidelityAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
    "SeedVR2WorstFrameFidelityAnalysis": "SeedVR2 Worst-Frame Fidelity Analysis",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
