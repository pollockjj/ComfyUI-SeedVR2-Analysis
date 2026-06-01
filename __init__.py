from .nodes import (
    SeedVR2Analysis,
    SeedVR2EquivalenceAnalysis,
    SeedVR2NativeLatentToNumzDiT,
    SeedVR2NativeRawDiTProbe,
    SeedVR2NumzPreparedConditioningToNative,
    SeedVR2NumzUpscaledLatentFileToNative,
    SeedVR2WorstFrameFidelityAnalysis,
)

NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
    "SeedVR2WorstFrameFidelityAnalysis": SeedVR2WorstFrameFidelityAnalysis,
    "SeedVR2NumzPreparedConditioningToNative": SeedVR2NumzPreparedConditioningToNative,
    "SeedVR2NumzUpscaledLatentFileToNative": SeedVR2NumzUpscaledLatentFileToNative,
    "SeedVR2NativeLatentToNumzDiT": SeedVR2NativeLatentToNumzDiT,
    "SeedVR2NativeRawDiTProbe": SeedVR2NativeRawDiTProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
    "SeedVR2WorstFrameFidelityAnalysis": "SeedVR2 Worst-Frame Fidelity Analysis",
    "SeedVR2NumzPreparedConditioningToNative": "SeedVR2 Numz Prepared Conditioning -> Native",
    "SeedVR2NumzUpscaledLatentFileToNative": "SeedVR2 Numz Upscaled Latent File -> Native",
    "SeedVR2NativeLatentToNumzDiT": "SeedVR2 Native Latent -> Numz DiT",
    "SeedVR2NativeRawDiTProbe": "SeedVR2 Native Raw DiT Probe",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
