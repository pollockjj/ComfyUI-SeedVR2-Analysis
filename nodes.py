"""SeedVR2-flavored video analysis node.

Consumes one VIDEO (NR mode) or two VIDEOs (FR mode) and emits a
schema-versioned JSON of per-frame-aggregated quality metrics. Inference-
agnostic — the model that produced ``output_video`` is out of scope.

Metrics:

    NR (always):  NIQE, MUSIQ, CLIP-IQA via pyiqa; DOVER fused via the
                  vendored VQAssessment/DOVER subprocess.
    FR (iff a shape-matching reference_video is supplied):
                  PSNR, SSIM, LPIPS (alex / vgg backbone selectable),
                  DISTS via pyiqa.

The node accepts ComfyUI ``VIDEO`` objects (``VideoFromFile`` /
``VideoFromComponents``) and bare string paths interchangeably. Inside
pyisolate-sealed runtimes the comfy adapter sends the VIDEO across as
``VideoFromComponents`` (images tensor + frame_rate), so the node
extracts frames via ``video.get_components()`` and works with the tensor
directly. For DOVER (which requires a video file on disk) the frames
are re-encoded to a temporary mp4 via an ffmpeg subprocess; the temp
file is removed after DOVER completes.

Each per-frame value is computed by feeding the frame tensor to a pyiqa
metric and aggregating across frames as mean / std / min / max plus the
full per-frame list. No fixture backends, no ``static-smoke`` placeholder
values.

Fail-loud paths:
    - missing pyiqa weights / DOVER weights / DOVER script  → raises.
    - shape mismatch on (frame_count, frame_rate, width, height) when
      reference_video is supplied → raises with the offending field name.
    - any pyiqa or DOVER backend exception → raises.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from fractions import Fraction
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DOVER_ROOT = REPO_ROOT / "vendor" / "DOVER"

NR_METRIC_NAMES = ("niqe", "musiq", "clip_iqa", "dover_fused")
FR_METRIC_NAMES = ("psnr", "ssim", "lpips", "dists")
VIDEO_ALIGNMENT_KEYS = ("frame_count", "frame_rate", "width", "height")

SCHEMA_VERSION = "1.0"

# Source of truth: scripts/bootstrap_vendor.sh. Keep in sync.
DOVER_REPO_URL = "https://github.com/VQAssessment/DOVER.git"
DOVER_PIN_SHA = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"
DOVER_WEIGHT_URL = "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"
DOVER_WEIGHT_SHA256 = "f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838"


def _bootstrap_dover_repo(dover_root: Path) -> None:
    """Clone VQAssessment/DOVER at the pinned SHA into dover_root."""
    dover_root.parent.mkdir(parents=True, exist_ok=True)
    if not (dover_root / ".git").exists():
        subprocess.run(
            ["git", "clone", DOVER_REPO_URL, str(dover_root)],
            check=True,
        )
    subprocess.run(["git", "-C", str(dover_root), "fetch", "origin"], check=True)
    subprocess.run(
        ["git", "-C", str(dover_root), "checkout", DOVER_PIN_SHA],
        check=True,
    )


def _download_dover_weights(weight_path: Path) -> None:
    """Download DOVER.pth to weight_path and verify sha256."""
    import urllib.request

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weight_path.with_suffix(weight_path.suffix + ".partial")
    with urllib.request.urlopen(DOVER_WEIGHT_URL) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    observed = sha256(tmp_path.read_bytes()).hexdigest()
    if observed != DOVER_WEIGHT_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"DOVER.pth sha256 mismatch: expected {DOVER_WEIGHT_SHA256}, "
            f"observed {observed}"
        )
    tmp_path.replace(weight_path)


def _ensure_dover_vendor(
    dover_root: Path,
    script_path: Path,
    weight_path: Path,
) -> None:
    """Idempotently bootstrap the vendored DOVER tree if missing."""
    if not script_path.is_file():
        _bootstrap_dover_repo(dover_root)
    if not weight_path.is_file():
        _download_dover_weights(weight_path)


def _resolve_ffmpeg() -> str:
    configured = os.environ.get("SEEDVR2_ANALYSIS_FFMPEG")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(Path(discovered))

    if os.name == "nt":
        tools_root = Path("C:/Tools")
        if tools_root.is_dir():
            candidates.extend(tools_root.glob("**/ffmpeg.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    path_value = os.environ.get("PATH", "")
    raise FileNotFoundError(
        "ffmpeg executable not found. Set SEEDVR2_ANALYSIS_FFMPEG to the "
        f"absolute ffmpeg path. PATH={path_value!r}"
    )


def _analysis_temp_dir() -> Path:
    env_dir = os.environ.get("MYSOLATE_ARTIFACTS_DIR")
    if env_dir:
        root = Path(env_dir)
    else:
        try:
            import folder_paths

            root = Path(folder_paths.get_temp_directory())
        except Exception:
            root = REPO_ROOT / "outputs"
    temp_dir = root / "seedvr2_analysis_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _aggregate(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        raise ValueError("metric returned zero frames")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": min(values),
        "max": max(values),
        "per_frame": values,
    }


def _is_video_object(obj: Any) -> bool:
    """ComfyUI VIDEO duck-test: has callable ``get_components`` returning
    a ``VideoComponents``-like with ``images`` and ``frame_rate``."""
    return callable(getattr(obj, "get_components", None))


def _frames_from_video(video_obj_or_path: Any) -> tuple[Any, Fraction]:
    """Return ``(images_tensor_NHWC_in_[0,1], frame_rate_Fraction)`` for
    either a ComfyUI VIDEO object or a string path. Tensor stays on CPU
    here; the caller permutes/moves to device per metric.
    """
    import torch  # noqa: F401  (cheap; deferred to keep ComfyUI startup fast)

    if _is_video_object(video_obj_or_path):
        components = video_obj_or_path.get_components()
        images = components.images  # (N, H, W, C) float in [0, 1]
        frame_rate = components.frame_rate
        if not isinstance(frame_rate, Fraction):
            frame_rate = Fraction(frame_rate)
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"VIDEO components.images must be (N, H, W, 3); got shape "
                f"{tuple(images.shape)}"
            )
        return images.detach().contiguous(), frame_rate

    # Fallback — bare string / Path. PyAV (decord replacement, py313-compatible).
    import av
    import numpy as np
    import torch as _torch

    path_str = str(video_obj_or_path)
    if not Path(path_str).is_file():
        raise FileNotFoundError(f"video path does not exist: {path_str}")
    container = av.open(path_str)
    try:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path_str}")
        stream = container.streams.video[0]
        avg_rate = stream.average_rate
        if avg_rate is None:
            raise ValueError(
                f"video has no average frame rate metadata: {path_str}"
            )
        fr = Fraction(avg_rate.numerator, avg_rate.denominator)
        frames_np: list = []
        for frame in container.decode(video=0):
            frames_np.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if not frames_np:
        raise ValueError(f"video has zero frames: {path_str}")
    images_u8 = np.stack(frames_np, axis=0)  # (N, H, W, 3) uint8
    images = (
        _torch.from_numpy(images_u8).to(dtype=_torch.float32).div_(255.0).contiguous()
    )
    return images, fr


def _to_pyiqa_input(images_nhwc):
    """Permute (N, H, W, 3) → (N, 3, H, W) and return contiguous tensor."""
    return images_nhwc.permute(0, 3, 1, 2).contiguous()


class SeedVR2MetricBackend:
    """Real metric backend — per-frame pyiqa over the VIDEO's frames tensor
    plus a DOVER subprocess against a temporary mp4 re-encoded from the
    same frames."""

    DOVER_WEIGHTS_PATH = DOVER_ROOT / "pretrained_weights" / "DOVER.pth"
    DOVER_SCRIPT_PATH = DOVER_ROOT / "evaluate_one_video.py"

    def __init__(self, lpips_backbone: str = "alex"):
        if lpips_backbone not in ("alex", "vgg"):
            raise ValueError(
                f"lpips_backbone must be 'alex' or 'vgg'; got {lpips_backbone!r}"
            )
        self._lpips_backbone = lpips_backbone

    @classmethod
    def _require_dover(cls) -> None:
        _ensure_dover_vendor(DOVER_ROOT, cls.DOVER_SCRIPT_PATH, cls.DOVER_WEIGHTS_PATH)
        if not cls.DOVER_SCRIPT_PATH.is_file():
            raise FileNotFoundError(
                f"DOVER script does not exist after bootstrap: {cls.DOVER_SCRIPT_PATH}"
            )
        if not cls.DOVER_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"DOVER weights do not exist after bootstrap: {cls.DOVER_WEIGHTS_PATH}"
            )

    @staticmethod
    def _select_device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run_pyiqa_per_frame(
        self,
        metric_name: str,
        frames_a_chw,
        frames_b_chw=None,
        **kwargs: Any,
    ) -> list[float]:
        import pyiqa
        device = self._select_device()
        metric_fn = pyiqa.create_metric(metric_name, device=device, **kwargs)
        n = frames_a_chw.shape[0]
        per_frame: list[float] = []
        for i in range(n):
            a = frames_a_chw[i : i + 1].to(device)
            if frames_b_chw is None:
                value = metric_fn(a)
            else:
                b = frames_b_chw[i : i + 1].to(device)
                value = metric_fn(a, b)
            per_frame.append(float(value.detach().cpu().item()))
        return per_frame

    @staticmethod
    def _encode_frames_to_temp_mp4(
        frames_nhwc, frame_rate: Fraction
    ) -> Path:
        """Re-encode frames tensor to a temporary mp4 via ffmpeg pipe.
        Caller must remove the file after use.
        """
        import numpy as np

        if frames_nhwc.dim() != 4 or frames_nhwc.shape[-1] != 3:
            raise ValueError(
                f"_encode_frames_to_temp_mp4 expects (N, H, W, 3); got "
                f"{tuple(frames_nhwc.shape)}"
            )
        n, h, w, _ = frames_nhwc.shape
        # Convert to uint8 NHWC bytes.
        arr_u8 = (
            (frames_nhwc.detach().cpu().clamp_(0.0, 1.0) * 255.0)
            .to(dtype=__import__("torch").uint8)
            .contiguous()
            .numpy()
        )
        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=".mp4",
            prefix="seedvr2_analysis_",
            dir=str(_analysis_temp_dir()),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        cmd = [
            _resolve_ffmpeg(),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", f"{frame_rate.numerator}/{frame_rate.denominator}",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-an",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, input=arr_u8.tobytes(), check=True, capture_output=True)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg produced empty mp4: {tmp_path}")
        return tmp_path

    def _run_dover_subprocess(self, video_path: Path) -> float:
        cmd = [
            sys.executable,
            str(self.DOVER_SCRIPT_PATH),
            "-v",
            str(video_path),
            "-f",
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(DOVER_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(
            r"Normalized fused overall score \(scale in \[0,1\]\):\s+([0-9.eE+-]+)",
            completed.stdout,
        )
        if match is None:
            raise RuntimeError(
                "could not parse DOVER fused score from subprocess stdout: "
                + repr(completed.stdout[-2000:])
            )
        return float(match.group(1))

    def compute_nr_metrics(
        self, frames_nhwc, frame_rate: Fraction
    ) -> dict[str, Any]:
        self._require_dover()
        frames_chw = _to_pyiqa_input(frames_nhwc)

        results: dict[str, Any] = {}
        results["niqe"] = _aggregate(self._run_pyiqa_per_frame("niqe", frames_chw))
        results["musiq"] = _aggregate(self._run_pyiqa_per_frame("musiq", frames_chw))
        results["clip_iqa"] = _aggregate(
            self._run_pyiqa_per_frame("clipiqa", frames_chw)
        )

        tmp_video = self._encode_frames_to_temp_mp4(frames_nhwc, frame_rate)
        try:
            results["dover_fused"] = self._run_dover_subprocess(tmp_video)
        finally:
            tmp_video.unlink(missing_ok=True)
        return results

    def compute_fr_metrics(
        self,
        out_frames_nhwc,
        ref_frames_nhwc,
    ) -> dict[str, Any]:
        if out_frames_nhwc.shape != ref_frames_nhwc.shape:
            raise ValueError(
                f"FR pair tensor shape mismatch: "
                f"output={tuple(out_frames_nhwc.shape)} "
                f"reference={tuple(ref_frames_nhwc.shape)}"
            )
        out_chw = _to_pyiqa_input(out_frames_nhwc)
        ref_chw = _to_pyiqa_input(ref_frames_nhwc)

        results: dict[str, Any] = {}
        results["psnr"] = _aggregate(
            self._run_pyiqa_per_frame("psnr", out_chw, ref_chw)
        )
        results["ssim"] = _aggregate(
            self._run_pyiqa_per_frame("ssim", out_chw, ref_chw)
        )
        lpips_per_frame = self._run_pyiqa_per_frame(
            "lpips", out_chw, ref_chw, net=self._lpips_backbone
        )
        lpips_block = _aggregate(lpips_per_frame)
        lpips_block["backbone"] = self._lpips_backbone
        results["lpips"] = lpips_block
        results["dists"] = _aggregate(
            self._run_pyiqa_per_frame("dists", out_chw, ref_chw)
        )
        return results

    @property
    def tool_provenance(self) -> dict[str, Any]:
        self._require_dover()

        def _pkg_version(name: str) -> str:
            try:
                return importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                raise RuntimeError(f"required package not installed: {name}")

        import torch

        ffmpeg_path = _resolve_ffmpeg()
        try:
            ff = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, check=True
            )
            ffmpeg_first_line = ff.stdout.splitlines()[0] if ff.stdout else ""
        except Exception:
            ffmpeg_first_line = "unknown"

        dover_repo_sha = "unknown"
        if (DOVER_ROOT / ".git").exists():
            try:
                rs = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(DOVER_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                dover_repo_sha = rs.stdout.strip()
            except Exception:
                pass

        return {
            "pyiqa": {
                "name": "pyiqa",
                "version": _pkg_version("pyiqa"),
            },
            "av": {
                "name": "av",
                "version": _pkg_version("av"),
            },
            "torch": {
                "name": "torch",
                "version": torch.__version__,
                "cuda": torch.version.cuda or "",
            },
            "ffmpeg": {
                "name": "ffmpeg",
                "version": ffmpeg_first_line,
            },
            "dover": {
                "name": "VQAssessment/DOVER",
                "repo_sha": dover_repo_sha,
                "script_path": str(self.DOVER_SCRIPT_PATH),
            },
            "dover_weights": {
                "name": "DOVER.pth",
                "path": str(self.DOVER_WEIGHTS_PATH),
                "sha256": _file_sha256(self.DOVER_WEIGHTS_PATH),
            },
        }


class SeedVR2Analysis:
    def __init__(self, metric_backend=None):
        self._metric_backend = metric_backend

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_video": ("VIDEO",),
            },
            "optional": {
                "reference_video": ("VIDEO",),
                "lpips_backbone": (["alex", "vgg"], {"default": "alex"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metrics_artifact",)
    FUNCTION = "analyze"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True

    @staticmethod
    def _artifact_dir() -> Path:
        try:
            import folder_paths
            return Path(folder_paths.get_output_directory()) / "seedvr2_analysis"
        except Exception:
            return REPO_ROOT / "outputs" / "seedvr2_analysis"

    @staticmethod
    def _alignment_metadata(frames_nhwc, frame_rate: Fraction) -> dict[str, Any]:
        n, h, w, _ = frames_nhwc.shape
        return {
            "frame_count": int(n),
            "frame_rate": str(frame_rate),
            "width": int(w),
            "height": int(h),
        }

    @classmethod
    def _assert_reference_alignment(
        cls,
        out_meta: dict[str, Any],
        ref_meta: dict[str, Any],
    ) -> dict[str, Any]:
        mismatches: list[str] = []
        for key in VIDEO_ALIGNMENT_KEYS:
            a = out_meta[key]
            b = ref_meta[key]
            if key == "frame_rate":
                a = Fraction(a)
                b = Fraction(b)
            if a != b:
                mismatches.append(key)
        if mismatches:
            raise ValueError(
                f"reference_video metadata mismatch: {', '.join(mismatches)}"
            )
        return {
            "matched": True,
            "checked": list(VIDEO_ALIGNMENT_KEYS),
            "output": out_meta,
            "reference": ref_meta,
            "mismatches": [],
        }

    def analyze(
        self,
        output_video,
        reference_video=None,
        lpips_backbone: str = "alex",
    ):
        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"seedvr2_analysis_{uuid.uuid4().hex}.json"

        out_frames, out_fps = _frames_from_video(output_video)
        out_meta = self._alignment_metadata(out_frames, out_fps)

        ref_frames = None
        ref_fps = None
        ref_meta = None
        alignment: dict[str, Any] = {
            "matched": False,
            "checked": [],
            "output": None,
            "reference": None,
            "mismatches": [],
        }
        if reference_video is not None:
            ref_frames, ref_fps = _frames_from_video(reference_video)
            ref_meta = self._alignment_metadata(ref_frames, ref_fps)
            alignment = self._assert_reference_alignment(out_meta, ref_meta)

        backend = self._metric_backend or SeedVR2MetricBackend(
            lpips_backbone=lpips_backbone
        )

        nr_metrics = backend.compute_nr_metrics(out_frames, out_fps)
        fr_metrics = (
            backend.compute_fr_metrics(out_frames, ref_frames)
            if ref_frames is not None
            else None
        )

        metrics_doc = {
            "schema_version": SCHEMA_VERSION,
            "inputs": {
                "output_video": (
                    repr(type(output_video).__name__) if _is_video_object(output_video)
                    else str(output_video)
                ),
                "reference_video": (
                    None
                    if reference_video is None
                    else (
                        repr(type(reference_video).__name__)
                        if _is_video_object(reference_video)
                        else str(reference_video)
                    )
                ),
            },
            "videos": {
                "output": out_meta,
                "reference": ref_meta,
            },
            "alignment": alignment,
            "metrics": {
                "nr": nr_metrics,
                "fr": fr_metrics,
            },
            "tool_provenance": backend.tool_provenance,
        }

        artifact_path.write_text(
            json.dumps(metrics_doc, indent=2, sort_keys=True), encoding="utf-8"
        )
        return (str(artifact_path),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
}
