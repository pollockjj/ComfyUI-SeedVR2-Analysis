from .nodes import (
    SeedVR2Analysis,
    SeedVR2EquivalenceAnalysis,
    SeedVR2ImageComparisonAnalysis,
    SeedVR2NativeLatentToNumzDiT,
    SeedVR2NativeRawDiTProbe,
    SeedVR2NumzRawDiTFromNativeProbe,
    SeedVR2NumzPreparedConditioningToNative,
    SeedVR2NumzUpscaledLatentFileToNative,
    SeedVR2WorstFrameFidelityAnalysis,
)

NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2ImageComparisonAnalysis": SeedVR2ImageComparisonAnalysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
    "SeedVR2WorstFrameFidelityAnalysis": SeedVR2WorstFrameFidelityAnalysis,
    "SeedVR2NumzPreparedConditioningToNative": SeedVR2NumzPreparedConditioningToNative,
    "SeedVR2NumzUpscaledLatentFileToNative": SeedVR2NumzUpscaledLatentFileToNative,
    "SeedVR2NativeLatentToNumzDiT": SeedVR2NativeLatentToNumzDiT,
    "SeedVR2NativeRawDiTProbe": SeedVR2NativeRawDiTProbe,
    "SeedVR2NumzRawDiTFromNativeProbe": SeedVR2NumzRawDiTFromNativeProbe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2ImageComparisonAnalysis": "SeedVR2 Image Comparison Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
    "SeedVR2WorstFrameFidelityAnalysis": "SeedVR2 Worst-Frame Fidelity Analysis",
    "SeedVR2NumzPreparedConditioningToNative": "SeedVR2 Numz Prepared Conditioning -> Native",
    "SeedVR2NumzUpscaledLatentFileToNative": "SeedVR2 Numz Upscaled Latent File -> Native",
    "SeedVR2NativeLatentToNumzDiT": "SeedVR2 Native Latent -> Numz DiT",
    "SeedVR2NativeRawDiTProbe": "SeedVR2 Native Raw DiT Probe",
    "SeedVR2NumzRawDiTFromNativeProbe": "SeedVR2 Numz Raw DiT From Native Probe",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
