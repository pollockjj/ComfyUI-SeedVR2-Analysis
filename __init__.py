from .nodes import SeedVR2Analysis, SeedVR2EquivalenceAnalysis

NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
