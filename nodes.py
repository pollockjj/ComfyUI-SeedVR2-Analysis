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
import math
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
COLOR_METRIC_NAMES = (
    "deltae76",
    "deltae00",
    "lab_l_mae",
    "lab_a_mae",
    "lab_b_mae",
    "chroma_mae",
    "hue_mae_deg",
    "lab_hist_w1",
)
FR_METRIC_NAMES = ("psnr", "ssim", "lpips", "dists") + COLOR_METRIC_NAMES
VIDEO_ALIGNMENT_KEYS = ("frame_count", "frame_rate", "width", "height")
DOVER_SAMPLE_SEED = 5770521
WEBP_DEFAULT_FRAME_RATE = Fraction(25, 1)

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


def _frames_from_webp(source: Any) -> tuple[Any, Fraction]:
    import numpy as np
    import torch
    from PIL import Image, ImageSequence

    if callable(getattr(source, "seek", None)):
        source.seek(0)
    im = Image.open(source)
    frames_np = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(im)]
    if not frames_np:
        raise ValueError(f"WebP has zero frames: {source}")
    images_u8 = np.stack(frames_np, axis=0)
    images = torch.from_numpy(images_u8).to(dtype=torch.float32).div_(255.0).contiguous()
    return images, WEBP_DEFAULT_FRAME_RATE


def _webp_source_from_video_object(video_obj: Any) -> Any | None:
    getter = getattr(video_obj, "get_stream_source", None)
    if callable(getter):
        source = getter()
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".webp" and path.is_file():
                return path
        if callable(getattr(source, "seek", None)):
            name = getattr(source, "name", "")
            if str(name).lower().endswith(".webp"):
                return source

    for attr in ("_VideoFromFile__file", "file", "path", "_path"):
        source = getattr(video_obj, attr, None)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".webp" and path.is_file():
                return path
        if callable(getattr(source, "seek", None)):
            name = getattr(source, "name", "")
            if str(name).lower().endswith(".webp"):
                return source
    return None


def _frames_from_video(video_obj_or_path: Any) -> tuple[Any, Fraction]:
    """Return ``(images_tensor_NHWC_in_[0,1], frame_rate_Fraction)`` for
    either a ComfyUI VIDEO object or a string path. Tensor stays on CPU
    here; the caller permutes/moves to device per metric.
    """
    import torch  # noqa: F401  (cheap; deferred to keep ComfyUI startup fast)

    if _is_video_object(video_obj_or_path):
        webp_source = _webp_source_from_video_object(video_obj_or_path)
        if webp_source is not None:
            return _frames_from_webp(webp_source)
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
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"video path does not exist: {path_str}")
    if path.suffix.lower() == ".webp":
        return _frames_from_webp(path)
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
        self,
        frames_nhwc,
        frame_rate: Fraction,
        enabled: set[str] | None = None,
    ) -> dict[str, Any]:
        if enabled is None:
            enabled = set(NR_METRIC_NAMES)
        results: dict[str, Any] = {}
        if not enabled:
            return results
        frames_chw = _to_pyiqa_input(frames_nhwc)
        if "niqe" in enabled:
            results["niqe"] = _aggregate(self._run_pyiqa_per_frame("niqe", frames_chw))
        if "musiq" in enabled:
            results["musiq"] = _aggregate(self._run_pyiqa_per_frame("musiq", frames_chw))
        if "clip_iqa" in enabled:
            results["clip_iqa"] = _aggregate(
                self._run_pyiqa_per_frame("clipiqa", frames_chw)
            )
        if "dover_fused" in enabled:
            self._require_dover()
            tmp_video = self._encode_frames_to_temp_mp4(frames_nhwc, frame_rate)
            try:
                results["dover_fused"] = self._run_dover_subprocess(tmp_video)
            finally:
                tmp_video.unlink(missing_ok=True)
        return results

    @staticmethod
    def _lab_histogram_w1(lab_a, lab_b) -> float:
        import numpy as np

        ranges = ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))
        distances = []
        for channel, value_range in enumerate(ranges):
            hist_a, edges = np.histogram(
                lab_a[..., channel],
                bins=256,
                range=value_range,
                density=False,
            )
            hist_b, _ = np.histogram(
                lab_b[..., channel],
                bins=256,
                range=value_range,
                density=False,
            )
            hist_a = hist_a.astype(np.float64)
            hist_b = hist_b.astype(np.float64)
            hist_a /= hist_a.sum()
            hist_b /= hist_b.sum()
            bin_width = float(edges[1] - edges[0])
            distances.append(float(np.abs(np.cumsum(hist_a) - np.cumsum(hist_b)).sum() * bin_width))
        return float(np.mean(distances))

    @staticmethod
    def _run_color_per_frame(out_frames_nhwc, ref_frames_nhwc, enabled: set[str]) -> dict[str, list[float]]:
        import numpy as np
        from skimage.color import deltaE_ciede2000, rgb2lab

        out: dict[str, list[float]] = {name: [] for name in sorted(enabled & set(COLOR_METRIC_NAMES))}
        if not out:
            return out
        n = out_frames_nhwc.shape[0]
        out_np = out_frames_nhwc.detach().cpu().clamp(0.0, 1.0).numpy()
        ref_np = ref_frames_nhwc.detach().cpu().clamp(0.0, 1.0).numpy()
        for i in range(n):
            lab_a = rgb2lab(out_np[i])
            lab_b = rgb2lab(ref_np[i])
            diff = lab_a - lab_b
            if "deltae76" in out:
                out["deltae76"].append(float(np.linalg.norm(diff, axis=-1).mean()))
            if "deltae00" in out:
                out["deltae00"].append(float(deltaE_ciede2000(lab_a, lab_b).mean()))
            if "lab_l_mae" in out:
                out["lab_l_mae"].append(float(np.abs(diff[..., 0]).mean()))
            if "lab_a_mae" in out:
                out["lab_a_mae"].append(float(np.abs(diff[..., 1]).mean()))
            if "lab_b_mae" in out:
                out["lab_b_mae"].append(float(np.abs(diff[..., 2]).mean()))
            chroma_a = np.hypot(lab_a[..., 1], lab_a[..., 2])
            chroma_b = np.hypot(lab_b[..., 1], lab_b[..., 2])
            if "chroma_mae" in out:
                out["chroma_mae"].append(float(np.abs(chroma_a - chroma_b).mean()))
            if "hue_mae_deg" in out:
                hue_a = np.degrees(np.arctan2(lab_a[..., 2], lab_a[..., 1])) % 360.0
                hue_b = np.degrees(np.arctan2(lab_b[..., 2], lab_b[..., 1])) % 360.0
                hue_diff = np.abs(((hue_a - hue_b + 180.0) % 360.0) - 180.0)
                chroma_mask = (chroma_a > 1e-6) | (chroma_b > 1e-6)
                out["hue_mae_deg"].append(float(hue_diff[chroma_mask].mean()) if chroma_mask.any() else 0.0)
            if "lab_hist_w1" in out:
                out["lab_hist_w1"].append(SeedVR2MetricBackend._lab_histogram_w1(lab_a, lab_b))
        return out

    def compute_fr_metrics(
        self,
        out_frames_nhwc,
        ref_frames_nhwc,
        enabled: set[str] | None = None,
    ) -> dict[str, Any]:
        if enabled is None:
            enabled = set(FR_METRIC_NAMES)
        results: dict[str, Any] = {}
        if not enabled:
            return results
        if out_frames_nhwc.shape != ref_frames_nhwc.shape:
            raise ValueError(
                f"FR pair tensor shape mismatch: "
                f"output={tuple(out_frames_nhwc.shape)} "
                f"reference={tuple(ref_frames_nhwc.shape)}"
            )
        out_chw = _to_pyiqa_input(out_frames_nhwc)
        ref_chw = _to_pyiqa_input(ref_frames_nhwc)
        if "psnr" in enabled:
            results["psnr"] = _aggregate(
                self._run_pyiqa_per_frame("psnr", out_chw, ref_chw)
            )
        if "ssim" in enabled:
            results["ssim"] = _aggregate(
                self._run_pyiqa_per_frame("ssim", out_chw, ref_chw)
            )
        if "lpips" in enabled:
            lpips_per_frame = self._run_pyiqa_per_frame(
                "lpips", out_chw, ref_chw, net=self._lpips_backbone
            )
            lpips_block = _aggregate(lpips_per_frame)
            lpips_block["backbone"] = self._lpips_backbone
            results["lpips"] = lpips_block
        if "dists" in enabled:
            results["dists"] = _aggregate(
                self._run_pyiqa_per_frame("dists", out_chw, ref_chw)
            )
        for name, values in self._run_color_per_frame(
            out_frames_nhwc,
            ref_frames_nhwc,
            enabled,
        ).items():
            results[name] = _aggregate(values)
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
            "skimage": {
                "name": "scikit-image",
                "version": _pkg_version("scikit-image"),
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
                # FR metric toggles (require reference_video)
                "enable_psnr": ("BOOLEAN", {"default": True}),
                "enable_ssim": ("BOOLEAN", {"default": True}),
                "enable_lpips": ("BOOLEAN", {"default": True}),
                "enable_dists": ("BOOLEAN", {"default": True}),
                "enable_color_metrics": ("BOOLEAN", {"default": False}),
                # NR metric toggles
                "enable_niqe": ("BOOLEAN", {"default": True}),
                "enable_musiq": ("BOOLEAN", {"default": True}),
                "enable_clip_iqa": ("BOOLEAN", {"default": True}),
                "enable_dover": ("BOOLEAN", {"default": True}),
                # Output destination overrides (empty = default ComfyUI/output/seedvr2_analysis/seedvr2_analysis_<uuid>.json)
                "output_directory": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
                # Embed the metrics JSON into the output_video file's container metadata
                "embed_in_source": ("BOOLEAN", {"default": False}),
            },
        }

    # metrics_json: the full JSON string (so a downstream save-text node can chain)
    # artifact_path: where it was written on disk
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metrics_json", "artifact_path")
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

    @staticmethod
    def _resolve_source_path(video_obj) -> Path | None:
        """Best-effort: return the on-disk file path backing a VIDEO object,
        or None if the source is not a file (BytesIO, VideoFromComponents, ...)."""
        if video_obj is None:
            return None
        getter = getattr(video_obj, "get_stream_source", None)
        if callable(getter):
            try:
                src = getter()
            except Exception:
                return None
            if isinstance(src, (str, Path)):
                p = Path(src)
                if p.is_file():
                    return p
        # Fallback: name-mangled private attr on VideoFromFile
        for attr in ("_VideoFromFile__file", "file", "path", "_path"):
            v = getattr(video_obj, attr, None)
            if isinstance(v, (str, Path)):
                p = Path(v)
                if p.is_file():
                    return p
        return None

    @staticmethod
    def _embed_metrics_into_mp4(source_path: Path, metrics_json: str) -> dict[str, Any]:
        """Remux source_path adding a `analysis_metrics` format-level tag containing
        metrics_json. Preserves existing format tags (e.g. ComfyUI's `prompt`).
        Returns a result dict describing what happened."""
        ffmpeg = _resolve_ffmpeg()
        # Use the same extension as the source so ffmpeg can infer the muxer;
        # pick a unique name to avoid collisions.
        tmp_out = source_path.with_name(
            f".{source_path.stem}.analysis_tmp.{uuid.uuid4().hex[:8]}{source_path.suffix}"
        )
        cmd = [
            ffmpeg, "-y",
            "-i", str(source_path),
            "-c", "copy",
            "-map_metadata", "0",
            "-metadata", f"analysis_metrics={metrics_json}",
            "-movflags", "+use_metadata_tags",
            str(tmp_out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            tmp_out.unlink(missing_ok=True)
            return {
                "attempted": True,
                "succeeded": False,
                "target": str(source_path),
                "error": e.stderr[-2000:] if e.stderr else str(e),
            }
        tmp_out.replace(source_path)
        return {
            "attempted": True,
            "succeeded": True,
            "target": str(source_path),
            "tag": "analysis_metrics",
        }

    def analyze(
        self,
        output_video,
        reference_video=None,
        lpips_backbone: str = "alex",
        enable_psnr: bool = True,
        enable_ssim: bool = True,
        enable_lpips: bool = True,
        enable_dists: bool = True,
        enable_color_metrics: bool = False,
        enable_niqe: bool = True,
        enable_musiq: bool = True,
        enable_clip_iqa: bool = True,
        enable_dover: bool = True,
        output_directory: str = "",
        output_filename: str = "",
        embed_in_source: bool = False,
    ):
        # Resolve enabled metric sets from user toggles
        enabled_fr: set[str] = set()
        if enable_psnr:  enabled_fr.add("psnr")
        if enable_ssim:  enabled_fr.add("ssim")
        if enable_lpips: enabled_fr.add("lpips")
        if enable_dists: enabled_fr.add("dists")
        if enable_color_metrics: enabled_fr.update(COLOR_METRIC_NAMES)
        enabled_nr: set[str] = set()
        if enable_niqe:     enabled_nr.add("niqe")
        if enable_musiq:    enabled_nr.add("musiq")
        if enable_clip_iqa: enabled_nr.add("clip_iqa")
        if enable_dover:    enabled_nr.add("dover_fused")

        # Resolve output destination
        if output_directory.strip():
            artifact_dir = Path(output_directory).expanduser().resolve()
        else:
            artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if output_filename.strip():
            fname = output_filename.strip()
            if not fname.lower().endswith(".json"):
                fname = fname + ".json"
        else:
            fname = f"seedvr2_analysis_{uuid.uuid4().hex}.json"
        artifact_path = artifact_dir / fname

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
        # FR alignment only runs if reference is provided AND at least one FR metric is on
        run_fr = reference_video is not None and bool(enabled_fr)
        if run_fr:
            ref_frames, ref_fps = _frames_from_video(reference_video)
            ref_meta = self._alignment_metadata(ref_frames, ref_fps)
            alignment = self._assert_reference_alignment(out_meta, ref_meta)
        elif reference_video is not None:
            # Reference supplied but no FR metric enabled — capture metadata, skip alignment assert
            ref_frames, ref_fps = _frames_from_video(reference_video)
            ref_meta = self._alignment_metadata(ref_frames, ref_fps)

        backend = self._metric_backend or SeedVR2MetricBackend(
            lpips_backbone=lpips_backbone
        )

        nr_metrics = backend.compute_nr_metrics(out_frames, out_fps, enabled=enabled_nr) if enabled_nr else {}
        fr_metrics = (
            backend.compute_fr_metrics(out_frames, ref_frames, enabled=enabled_fr)
            if run_fr
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
            "enabled_metrics": {
                "fr": sorted(enabled_fr),
                "nr": sorted(enabled_nr),
            },
            "metrics": {
                "nr": nr_metrics,
                "fr": fr_metrics,
            },
            "tool_provenance": backend.tool_provenance,
        }

        metrics_json = json.dumps(metrics_doc, indent=2, sort_keys=True)
        artifact_path.write_text(metrics_json, encoding="utf-8")

        if embed_in_source:
            source_path = self._resolve_source_path(output_video)
            if source_path is None:
                metrics_doc["embed_in_source"] = {
                    "attempted": True,
                    "succeeded": False,
                    "target": None,
                    "error": "output_video is not backed by a resolvable file path",
                }
            else:
                metrics_doc["embed_in_source"] = self._embed_metrics_into_mp4(
                    source_path, metrics_json
                )
            # Re-serialize so the JSON record reflects the embed attempt
            metrics_json = json.dumps(metrics_doc, indent=2, sort_keys=True)
            artifact_path.write_text(metrics_json, encoding="utf-8")

        return (metrics_json, str(artifact_path))


EQUIVALENCE_SCHEMA_VERSION = "2.2"
WORST_FRAME_SCHEMA_VERSION = "1.0"
IMAGE_COMPARISON_SCHEMA_VERSION = "1.0"
FAST_VISUAL_FIDELITY_METRICS = ("psnr", "ssim", "lpips", "dists")

# Direction: True iff higher metric value = better quality.
# video1 is treated as the "candidate" (e.g. native); video2 is the "reference impl" (e.g. numz).
# "worse" for the candidate means: lower on a higher-is-better metric, OR higher on a lower-is-better metric.
METRIC_HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "musiq": True,
    "clip_iqa": True,
    "dover_fused": True,
    "lpips": False,
    "dists": False,
    "deltae76": False,
    "deltae00": False,
    "lab_l_mae": False,
    "lab_a_mae": False,
    "lab_b_mae": False,
    "chroma_mae": False,
    "hue_mae_deg": False,
    "lab_hist_w1": False,
    "niqe": False,
    "alpha_mae": False,
    "alpha_psnr": True,
}


class SeedVR2ImageComparisonAnalysis:
    """Full-reference comparison for SeedVR2 still-image validation sets."""

    def __init__(self, metric_backend=None):
        self._metric_backend = metric_backend

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "floor": ("IMAGE",),
                "input1": ("IMAGE",),
                "input2": ("IMAGE",),
            },
            "optional": {
                "input1_label": ("STRING", {"default": "input1"}),
                "input2_label": ("STRING", {"default": "input2"}),
                "lpips_backbone": (["alex", "vgg"], {"default": "alex"}),
                "enable_psnr": ("BOOLEAN", {"default": True}),
                "enable_ssim": ("BOOLEAN", {"default": True}),
                "enable_lpips": ("BOOLEAN", {"default": True}),
                "enable_dists": ("BOOLEAN", {"default": True}),
                "enable_color_metrics": ("BOOLEAN", {"default": False}),
                "output_directory": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metrics_json", "artifact_path", "summary")
    FUNCTION = "analyze"
    CATEGORY = "image/analysis"
    OUTPUT_NODE = True

    @staticmethod
    def _artifact_dir() -> Path:
        try:
            import folder_paths
            return Path(folder_paths.get_output_directory()) / "seedvr2_analysis"
        except Exception:
            return REPO_ROOT / "outputs" / "seedvr2_analysis"

    @staticmethod
    def _image_label(value) -> str:
        shape = getattr(value, "shape", None)
        return f"IMAGE{tuple(shape)}" if shape is not None else repr(type(value).__name__)

    @staticmethod
    def _split_rgb_alpha(image, name: str):
        import torch

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"{name} must be a torch IMAGE tensor")
        if image.dim() != 4:
            raise ValueError(f"{name} must be (N, H, W, C); got shape {tuple(image.shape)}")
        channels = image.shape[-1]
        if channels not in (1, 3, 4):
            raise ValueError(f"{name} must have 1, 3, or 4 channels; got {channels}")
        image = image.detach().to(dtype=torch.float32).clamp(0.0, 1.0).contiguous()
        if channels == 1:
            return image.expand(-1, -1, -1, 3).contiguous(), None
        if channels == 3:
            return image, None
        return image[..., :3].contiguous(), image[..., 3].contiguous()

    @staticmethod
    def _broadcast_batch(tensor, count: int, name: str):
        if tensor is None:
            return None
        if tensor.shape[0] == count:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(count, *tensor.shape[1:]).contiguous()
        raise ValueError(f"{name} batch count {tensor.shape[0]} cannot broadcast to {count}")

    @classmethod
    def _normalize_inputs(cls, images: dict[str, Any]):
        split = {name: cls._split_rgb_alpha(image, name) for name, image in images.items()}
        batch_count = max(rgb.shape[0] for rgb, _ in split.values())
        out = {}
        for name, (rgb, alpha) in split.items():
            out[name] = (
                cls._broadcast_batch(rgb, batch_count, f"{name}.rgb"),
                cls._broadcast_batch(alpha, batch_count, f"{name}.alpha"),
            )
        return out

    @staticmethod
    def _resize_nhwc(images, height: int, width: int):
        import torch.nn.functional as F

        if images.shape[1] == height and images.shape[2] == width:
            return images
        channels_first = images.permute(0, 3, 1, 2)
        resized = F.interpolate(
            channels_first,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _resize_alpha(alpha, height: int, width: int):
        import torch.nn.functional as F

        if alpha.shape[1] == height and alpha.shape[2] == width:
            return alpha
        resized = F.interpolate(
            alpha.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.squeeze(1).contiguous()

    @classmethod
    def _score_images(cls, backend, candidate_rgb, candidate_alpha, reference_rgb, reference_alpha, enabled: set[str]):
        import torch

        height, width = reference_rgb.shape[1:3]
        candidate_rgb = cls._resize_nhwc(candidate_rgb, height, width).clamp(0.0, 1.0)
        if reference_alpha is not None:
            if candidate_alpha is None:
                candidate_alpha = torch.ones_like(reference_alpha)
            else:
                candidate_alpha = cls._resize_alpha(candidate_alpha, height, width).clamp(0.0, 1.0)
            candidate_fr = candidate_rgb * candidate_alpha.unsqueeze(-1)
            reference_fr = reference_rgb * reference_alpha.unsqueeze(-1)
        else:
            candidate_fr = candidate_rgb
            reference_fr = reference_rgb

        metrics = backend.compute_fr_metrics(candidate_fr, reference_fr, enabled=enabled)
        if reference_alpha is not None:
            alpha_delta = candidate_alpha - reference_alpha
            alpha_mae = alpha_delta.abs().mean(dim=(1, 2))
            alpha_mse = alpha_delta.square().mean(dim=(1, 2))
            alpha_psnr = [
                99.0 if float(mse) <= 1e-12 else float(10.0 * math.log10(1.0 / float(mse)))
                for mse in alpha_mse
            ]
            metrics["alpha_mae"] = _aggregate([float(v) for v in alpha_mae])
            metrics["alpha_psnr"] = _aggregate(alpha_psnr)
        return metrics

    @staticmethod
    def _enabled_metrics(enable_psnr, enable_ssim, enable_lpips, enable_dists, enable_color_metrics) -> set[str]:
        enabled: set[str] = set()
        if enable_psnr:
            enabled.add("psnr")
        if enable_ssim:
            enabled.add("ssim")
        if enable_lpips:
            enabled.add("lpips")
        if enable_dists:
            enabled.add("dists")
        if enable_color_metrics:
            enabled.update(COLOR_METRIC_NAMES)
        return enabled

    @staticmethod
    def _winner(value1: float, value2: float, label1: str, label2: str, higher_is_better: bool) -> str:
        if value1 == value2:
            return "tie"
        if higher_is_better:
            return label1 if value1 > value2 else label2
        return label1 if value1 < value2 else label2

    @staticmethod
    def _beats_floor(value: float, floor_value: float, higher_is_better: bool) -> bool:
        if value == floor_value:
            return False
        return value > floor_value if higher_is_better else value < floor_value

    def analyze(
        self,
        reference,
        floor,
        input1,
        input2,
        input1_label: str = "input1",
        input2_label: str = "input2",
        lpips_backbone: str = "alex",
        enable_psnr: bool = True,
        enable_ssim: bool = True,
        enable_lpips: bool = True,
        enable_dists: bool = True,
        enable_color_metrics: bool = False,
        output_directory: str = "",
        output_filename: str = "",
    ):
        label1 = input1_label.strip() or "input1"
        label2 = input2_label.strip() or "input2"
        if label1 == label2 or label1 == "floor" or label2 == "floor":
            raise ValueError("input labels must be non-empty and distinct from each other and 'floor'")

        if output_directory.strip():
            artifact_dir = Path(output_directory).expanduser().resolve()
        else:
            artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if output_filename.strip():
            fname = output_filename.strip()
            if not fname.lower().endswith(".json"):
                fname = fname + ".json"
        else:
            fname = f"seedvr2_image_comparison_{uuid.uuid4().hex}.json"
        artifact_path = artifact_dir / fname

        normalized = self._normalize_inputs({
            "reference": reference,
            "floor": floor,
            label1: input1,
            label2: input2,
        })
        reference_rgb, reference_alpha = normalized["reference"]
        height, width = reference_rgb.shape[1:3]
        batch_count = reference_rgb.shape[0]
        enabled = self._enabled_metrics(
            enable_psnr,
            enable_ssim,
            enable_lpips,
            enable_dists,
            enable_color_metrics,
        )
        backend = self._metric_backend or SeedVR2MetricBackend(lpips_backbone=lpips_backbone)
        leg_metrics = {}
        for name in ("floor", label1, label2):
            rgb, alpha = normalized[name]
            leg_metrics[name] = self._score_images(
                backend,
                rgb,
                alpha,
                reference_rgb,
                reference_alpha,
                enabled,
            )

        metric_names = [m for m in FR_METRIC_NAMES if m in leg_metrics[label1]]
        if reference_alpha is not None:
            metric_names.extend([m for m in ("alpha_mae", "alpha_psnr") if m in leg_metrics[label1]])

        rows: list[dict[str, Any]] = []
        wins = {label1: 0, label2: 0}
        for metric_name in metric_names:
            higher_is_better = METRIC_HIGHER_IS_BETTER[metric_name]
            v1 = float(leg_metrics[label1][metric_name]["mean"])
            v2 = float(leg_metrics[label2][metric_name]["mean"])
            vf = float(leg_metrics["floor"][metric_name]["mean"])
            winner = self._winner(v1, v2, label1, label2, higher_is_better)
            if winner in wins:
                wins[winner] += 1
            rows.append({
                "metric": metric_name,
                "direction": "higher" if higher_is_better else "lower",
                label1: v1,
                label2: v2,
                "floor": vf,
                "winner": winner,
                f"{label1}_beats_floor": self._beats_floor(v1, vf, higher_is_better),
                f"{label2}_beats_floor": self._beats_floor(v2, vf, higher_is_better),
            })

        verdict = label1 if wins[label1] > wins[label2] else (label2 if wins[label2] > wins[label1] else "tie")
        summary = f"metric wins: {label1}={wins[label1]} {label2}={wins[label2]} verdict={verdict}"
        metrics_doc = {
            "schema_version": IMAGE_COMPARISON_SCHEMA_VERSION,
            "node": "SeedVR2ImageComparisonAnalysis",
            "inputs": {
                "reference": self._image_label(reference),
                "floor": self._image_label(floor),
                label1: self._image_label(input1),
                label2: self._image_label(input2),
            },
            "images": {
                "batch_count": int(batch_count),
                "width": int(width),
                "height": int(height),
                "reference_has_alpha": reference_alpha is not None,
            },
            "metrics": rows,
            "raw_metrics": leg_metrics,
            "metric_wins": wins,
            "verdict": verdict,
            "tool_provenance": SeedVR2WorstFrameFidelityAnalysis._tool_provenance(),
        }
        metrics_json = json.dumps(metrics_doc, indent=2)
        artifact_path.write_text(metrics_json, encoding="utf-8")
        return (metrics_json, str(artifact_path), summary)


def _bayes_normal_posterior_decision(
    diffs: list[float],
    rope: float,
    cred: float,
) -> dict[str, Any]:
    """Closed-form Bayesian-Normal posterior on the mean of paired differences.

    Model: d_i ~ Normal(mu, sigma^2) with non-informative reference prior
    p(mu, sigma) ~ 1/sigma. The marginal posterior on mu is then
    Student-t(loc=mean(d), scale=std(d)/sqrt(n), df=n-1) — the conjugate
    Bayesian counterpart of the frequentist paired t-test.

    Decision rule (Kruschke):
      - EQUIVALENT       : HDI ⊂ ROPE
      - NOT_EQUIVALENT   : HDI ∩ ROPE = ∅
      - UNDECIDED        : HDI partially in ROPE
    """
    import math
    from statistics import fmean, stdev
    from scipy import stats as scipy_stats

    n = len(diffs)
    if n < 2:
        return {
            "n_frames": n,
            "mean_diff": diffs[0] if diffs else None,
            "std_diff": None,
            "se_diff": None,
            "hdi_lo": None,
            "hdi_hi": None,
            "rope_lo": -abs(rope),
            "rope_hi": abs(rope),
            "p_in_rope": None,
            "decision": "INSUFFICIENT_DATA",
            "model": "Bayesian-Normal posterior on mu_diff (closed-form Student-t)",
        }
    m = fmean(diffs)
    s = stdev(diffs)
    se = s / math.sqrt(n)
    df = n - 1
    rope_lo = -abs(rope)
    rope_hi = abs(rope)
    if se == 0.0:
        # Degenerate posterior: point mass at m. Decide directly.
        in_rope = rope_lo <= m <= rope_hi
        decision = "EQUIVALENT" if in_rope else "NOT_EQUIVALENT"
        return {
            "n_frames": n,
            "mean_diff": m,
            "std_diff": s,
            "se_diff": se,
            "hdi_lo": m,
            "hdi_hi": m,
            "rope_lo": rope_lo,
            "rope_hi": rope_hi,
            "p_in_rope": 1.0 if in_rope else 0.0,
            "decision": decision,
            "model": "Bayesian-Normal posterior on mu_diff (degenerate: zero variance)",
        }
    alpha = (1.0 - cred) / 2.0
    # Student-t equal-tail interval; for symmetric Student-t this equals the HDI.
    hdi_lo = scipy_stats.t.ppf(alpha, df=df, loc=m, scale=se)
    hdi_hi = scipy_stats.t.ppf(1.0 - alpha, df=df, loc=m, scale=se)
    p_lo = scipy_stats.t.cdf(rope_lo, df=df, loc=m, scale=se)
    p_hi = scipy_stats.t.cdf(rope_hi, df=df, loc=m, scale=se)
    p_in_rope = float(p_hi - p_lo)
    if hdi_lo >= rope_lo and hdi_hi <= rope_hi:
        decision = "EQUIVALENT"
    elif hdi_hi < rope_lo or hdi_lo > rope_hi:
        decision = "NOT_EQUIVALENT"
    else:
        decision = "UNDECIDED"
    return {
        "n_frames": n,
        "mean_diff": m,
        "std_diff": s,
        "se_diff": se,
        "hdi_lo": float(hdi_lo),
        "hdi_hi": float(hdi_hi),
        "rope_lo": rope_lo,
        "rope_hi": rope_hi,
        "p_in_rope": p_in_rope,
        "decision": decision,
        "model": "Bayesian-Normal posterior on mu_diff (closed-form Student-t)",
    }


def _per_metric_non_inferiority(
    eq_result: dict[str, Any],
    rope: float,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Direction-aware non-inferiority computation built on top of the per-metric
    Bayesian-Normal posterior. video1 (candidate) is non-inferior to video2
    (reference impl) on this metric iff its worse-direction excess is bounded
    by rope.

    Returns:
        worse_excess_mean: posterior-mean shittiness in raw units (0 if better-or-equal).
        worse_excess_in_ropes: same, in ROPE-half-width units.
        p_non_inferior: P(candidate not worse than reference by more than rope).
        ni_decision: NON_INFERIOR / INFERIOR / UNDECIDED.
    """
    import math
    from scipy import stats as scipy_stats

    if eq_result.get("decision") in ("NOT_PAIRABLE", "INSUFFICIENT_DATA"):
        return {
            "worse_excess_mean": None,
            "worse_excess_in_ropes": None,
            "p_non_inferior": None,
            "ni_decision": eq_result.get("decision", "UNDECIDED"),
        }

    m = eq_result["mean_diff"]
    se = eq_result.get("se_diff")
    n = eq_result["n_frames"]
    # Worse-direction sign convention: define w = mean_diff in the direction
    # where positive == candidate is worse than reference.
    #   higher-is-better metric: worse when video1 < video2 -> w = -(v1 - v2) = -mean_diff
    #   lower-is-better metric:  worse when video1 > video2 -> w = +(v1 - v2) = +mean_diff
    w = -m if higher_is_better else m
    rope_pos = abs(rope)
    # Posterior-mean worse-excess: max(0, w) clipped (we only count how far candidate
    # is worse than reference; better-or-equal contributes zero).
    worse_excess_mean = float(max(0.0, w))
    worse_excess_in_ropes = float(worse_excess_mean / rope_pos) if rope_pos > 0 else None

    # P(non-inferior) = P(w <= rope) under the Student-t posterior on the mean diff.
    # The posterior on w is Student-t with center = w, scale = se_diff, df = n-1
    # (sign-flip is a shift of mean; scale unchanged).
    if se is None or se == 0.0 or n < 2:
        # Degenerate posterior at the point estimate. Decide by comparing to ROPE.
        if worse_excess_mean <= rope_pos:
            p_ni = 1.0
            ni_decision = "NON_INFERIOR"
        else:
            p_ni = 0.0
            ni_decision = "INFERIOR"
        return {
            "worse_excess_mean": worse_excess_mean,
            "worse_excess_in_ropes": worse_excess_in_ropes,
            "p_non_inferior": p_ni,
            "ni_decision": ni_decision,
            "model": "degenerate posterior (zero variance)",
        }
    df = n - 1
    # P(w <= rope_pos) where w has Student-t(loc=w, scale=se, df=df)
    p_ni = float(scipy_stats.t.cdf(rope_pos, df=df, loc=w, scale=se))
    if p_ni >= 0.95:
        ni_decision = "NON_INFERIOR"
    elif p_ni <= 0.05:
        ni_decision = "INFERIOR"
    else:
        ni_decision = "UNDECIDED"
    return {
        "worse_excess_mean": worse_excess_mean,
        "worse_excess_in_ropes": worse_excess_in_ropes,
        "p_non_inferior": p_ni,
        "ni_decision": ni_decision,
        "model": "Student-t posterior on worse-direction mu_diff",
    }


def _compute_joint_non_inferiority(
    per_metric_eq: dict[str, dict[str, Any]],
    ropes: dict[str, float],
) -> dict[str, Any]:
    """Combine per-metric non-inferiority into a joint cross-metric verdict.

    Under independence (caveat noted), joint P(all metrics non-inferior) = product
    of per-metric P_NI. Cumulative shittiness sums per-metric worse-excess in
    ROPE-units across metrics -- captures Boss's 'all a little shittier ->
    movie is a lot shittier' intuition.

    Aggregate verdict mapping (loose, designed for the "is native at least as
    good as numz" question):
        NOT_INFERIOR : every per-metric ni_decision is NON_INFERIOR
                       AND joint_p_non_inferior >= 0.50
                       AND cumulative_worse_excess_in_ropes <= 1.0
        INFERIOR     : any per-metric ni_decision == INFERIOR
                       OR cumulative_worse_excess_in_ropes >= 3.0
        UNDECIDED    : otherwise
    """
    import math

    per_metric_ni: dict[str, dict[str, Any]] = {}
    log_p_sum = 0.0
    log_p_valid = True
    cumulative_we = 0.0
    cumulative_count = 0
    any_inferior = False
    all_non_inferior = True
    for metric, eq in per_metric_eq.items():
        direction = METRIC_HIGHER_IS_BETTER.get(metric)
        if direction is None:
            continue
        rope = ropes.get(metric)
        if rope is None:
            continue
        ni = _per_metric_non_inferiority(eq, rope, direction)
        per_metric_ni[metric] = ni
        if ni.get("ni_decision") == "INFERIOR":
            any_inferior = True
            all_non_inferior = False
        elif ni.get("ni_decision") != "NON_INFERIOR":
            all_non_inferior = False
        p = ni.get("p_non_inferior")
        if p is None:
            log_p_valid = False
        else:
            # Floor to avoid log(0) blowing up the product on a single hard fail.
            log_p_sum += math.log(max(p, 1e-12))
        we = ni.get("worse_excess_in_ropes")
        if we is not None:
            cumulative_we += we
            cumulative_count += 1
    joint_p_ni = math.exp(log_p_sum) if log_p_valid else None

    if any_inferior or cumulative_we >= 3.0:
        verdict = "INFERIOR"
    elif all_non_inferior and (joint_p_ni is None or joint_p_ni >= 0.50) and cumulative_we <= 1.0:
        verdict = "NOT_INFERIOR"
    else:
        verdict = "UNDECIDED"

    return {
        "method": (
            "Direction-aware per-metric non-inferiority on the worse-side Student-t "
            "posterior tail; joint product under independence assumption "
            "(caveat: metric posteriors are not actually independent — joint p is a "
            "conservative anchor, not a calibrated joint probability)."
        ),
        "per_metric": per_metric_ni,
        "joint_p_non_inferior": joint_p_ni,
        "joint_log_p_non_inferior": log_p_sum if log_p_valid else None,
        "cumulative_worse_excess_in_ropes": cumulative_we,
        "cumulative_worse_excess_metric_count": cumulative_count,
        "aggregate_verdict": verdict,
    }


def _render_equivalence_text_table(metrics_doc: dict) -> str:
    inputs = metrics_doc.get("inputs", {}) or {}
    videos = metrics_doc.get("videos", {}) or {}
    eq = metrics_doc.get("equivalence", {}) or {}
    joint = metrics_doc.get("joint_non_inferiority", {}) or {}
    ropes = metrics_doc.get("rope_half_widths", {}) or {}
    metrics = metrics_doc.get("metrics", {}) or {}
    v1m = metrics.get("video1", {}) or {}
    v2m = metrics.get("video2", {}) or {}

    def _video_label(side: str) -> str:
        v = videos.get(side, {}) or {}
        fc = v.get("frame_count")
        fr = v.get("frame_rate")
        w = v.get("width")
        h = v.get("height")
        return f"{w}x{h}@{fr}fps, {fc} frames" if fc else "(unknown)"

    def _mean_from(side_block: dict, family: str, name: str):
        fam = side_block.get(family, {}) or {}
        m = fam.get(name, {}) or {}
        return m.get("mean")

    metric_rows = [
        ("psnr", "fr", "higher_is_better"),
        ("ssim", "fr", "higher_is_better"),
        ("lpips", "fr", "lower_is_better"),
        ("dists", "fr", "lower_is_better"),
        ("deltae76", "fr", "lower_is_better"),
        ("deltae00", "fr", "lower_is_better"),
        ("lab_l_mae", "fr", "lower_is_better"),
        ("lab_a_mae", "fr", "lower_is_better"),
        ("lab_b_mae", "fr", "lower_is_better"),
        ("chroma_mae", "fr", "lower_is_better"),
        ("hue_mae_deg", "fr", "lower_is_better"),
        ("lab_hist_w1", "fr", "lower_is_better"),
        ("niqe", "nr", "lower_is_better"),
        ("musiq", "nr", "higher_is_better"),
        ("clip_iqa", "nr", "higher_is_better"),
        ("dover_fused", "nr", "higher_is_better"),
    ]

    lines = []
    lines.append("=" * 96)
    lines.append("SeedVR2 Equivalence Analysis  —  video1 vs video2 (worse-direction non-inferiority)")
    lines.append("=" * 96)
    lines.append(f"video1 (subject):     {inputs.get('video1', '?')}   [{_video_label('video1')}]")
    lines.append(f"video2 (reference):   {inputs.get('video2', '?')}   [{_video_label('video2')}]")
    lines.append(f"reference (HQ):       {inputs.get('reference', '?')}   [{_video_label('reference')}]")
    lines.append(f"HDI credibility:      {eq.get('hdi_credibility', '?')}")
    lines.append(f"Aggregate (joint):    {joint.get('aggregate_verdict', '?')}    "
                 f"cum worse-excess: {joint.get('cumulative_worse_excess_in_ropes', float('nan')):.3f} ROPE-units "
                 f"(n_metrics={joint.get('cumulative_worse_excess_metric_count', 0)})")
    lines.append(f"BEST overall:         {eq.get('overall_decision', '?')}")
    lines.append("-" * 96)
    header = f"{'metric':<12} {'NI verdict':<18} {'wer (ROPE)':>10} {'rope_hw':>10} {'v1_mean':>14} {'v2_mean':>14} {'v1 - v2':>14}"
    lines.append(header)
    lines.append("-" * 96)

    per_metric_ni = joint.get("per_metric", {}) or {}
    for name, family, _ in metric_rows:
        pm = per_metric_ni.get(name, {}) or {}
        ni = pm.get("ni_decision", "—")
        wer = pm.get("worse_excess_in_ropes")
        rope_hw = ropes.get(name)
        v1_mean = _mean_from(v1m, family, name)
        v2_mean = _mean_from(v2m, family, name)
        def _fmt(x, w=14, p=4):
            if x is None:
                return f"{'—':>{w}}"
            try:
                return f"{float(x):>{w}.{p}f}"
            except (TypeError, ValueError):
                return f"{str(x):>{w}}"
        diff = None
        if v1_mean is not None and v2_mean is not None:
            try:
                diff = float(v1_mean) - float(v2_mean)
            except (TypeError, ValueError):
                diff = None
        lines.append(
            f"{name:<12} {ni:<18} {_fmt(wer, 10, 3)} {_fmt(rope_hw, 10, 4)} "
            f"{_fmt(v1_mean, 14, 4)} {_fmt(v2_mean, 14, 4)} {_fmt(diff, 14, 4)}"
        )

    lines.append("=" * 96)
    return "\n".join(lines) + "\n"


def _render_rev3_text_table(metrics_doc: dict) -> str:
    block = metrics_doc.get("rev3", {}) or {}
    videos = block.get("videos", {}) or {}
    scores = block.get("scores", {}) or {}
    perf = block.get("perf_delta", {}) or {}

    def _video_label(side: str) -> str:
        v = videos.get(side, {}) or {}
        fc = v.get("frame_count")
        fr = v.get("frame_rate")
        w = v.get("width")
        h = v.get("height")
        return f"{w}x{h}@{fr}fps, {fc} frames" if fc else "(unknown)"

    def _fmt(value, width=14, precision=4):
        if value is None:
            return f"{'--':>{width}}"
        try:
            return f"{float(value):>{width}.{precision}f}"
        except (TypeError, ValueError):
            return f"{'--':>{width}}"

    def _fmt_percent(value, width=16, precision=1):
        if value is None:
            return f"{'--':>{width}}"
        try:
            return f"{float(value) * 100.0:>{width - 1}.{precision}f}%"
        except (TypeError, ValueError):
            return f"{'--':>{width}}"

    def _metric_label(name: str) -> str:
        direction = "higher" if METRIC_HIGHER_IS_BETTER[name] else "lower"
        return f"{name} ({direction})"

    lines = []
    lines.append("=" * 96)
    lines.append("SeedVR2 Rev3 Anchor Analysis  -  reference / numz / native / floor")
    lines.append("=" * 96)
    lines.append(f"reference (HQ):       {_video_label('reference')}")
    lines.append(f"numz:                 {_video_label('numz')}")
    lines.append(f"native:               {_video_label('native')}")
    lines.append(f"floor (ESRGAN):       {_video_label('floor')}")
    lines.append("-" * 96)
    lines.append(
        f"{'metric':<20} {'reference':>14} {'numz':>14} "
        f"{'native':>14} {'floor':>14} {'perf_delta_numz':>16} {'perf_delta_native':>18}"
    )
    lines.append("-" * 96)
    for name in ("psnr", "ssim", "lpips", "dists", *COLOR_METRIC_NAMES, "niqe", "musiq", "clip_iqa", "dover_fused"):
        metric_scores = scores.get(name)
        if not metric_scores:
            continue
        metric_perf = perf.get(name, {}) or {}
        reference_score = None if name in set(FR_METRIC_NAMES) else metric_scores.get("reference")
        lines.append(
            f"{_metric_label(name):<20} {_fmt(reference_score)} "
            f"{_fmt(metric_scores.get('numz'))} {_fmt(metric_scores.get('native'))} "
            f"{_fmt(metric_scores.get('floor'))} "
            f"{_fmt_percent(metric_perf.get('numz'))} {_fmt_percent(metric_perf.get('native'), 18)}"
        )
    lines.append("=" * 96)
    return "\n".join(lines) + "\n"


class SeedVR2EquivalenceAnalysis:
    """Three-video paired analysis with BEST-style ROPE equivalence testing.

    Inputs:
      - reference (optional): if supplied, FR metrics (PSNR/SSIM/LPIPS/DISTS)
        are computed against (reference, video1) and (reference, video2)
        and per-frame paired differences feed the equivalence test.
      - video1, video2 (required): NR metrics (NIQE/MUSIQ/CLIP-IQA)
        are computed on each, paired across frames.
    Per metric, paired per-frame differences (m1[i] - m2[i]) drive a
    Bayesian-Normal posterior on the mean difference; HDI vs ROPE decides
    EQUIVALENT / NOT_EQUIVALENT / UNDECIDED per Kruschke.
    """

    def __init__(self, metric_backend=None):
        self._metric_backend = metric_backend

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("VIDEO",),
                "video2": ("VIDEO",),
            },
            "optional": {
                "reference": ("VIDEO",),
                "floor_video": ("VIDEO",),
                "lpips_backbone": (["alex", "vgg"], {"default": "alex"}),
                # Metric toggles
                "enable_psnr": ("BOOLEAN", {"default": True}),
                "enable_ssim": ("BOOLEAN", {"default": True}),
                "enable_lpips": ("BOOLEAN", {"default": True}),
                "enable_dists": ("BOOLEAN", {"default": True}),
                "enable_color_metrics": ("BOOLEAN", {"default": False}),
                "enable_niqe": ("BOOLEAN", {"default": True}),
                "enable_musiq": ("BOOLEAN", {"default": True}),
                "enable_clip_iqa": ("BOOLEAN", {"default": True}),
                "enable_dover": ("BOOLEAN", {"default": True}),
                # ROPE half-widths (operator must justify these on prior grounds)
                "rope_psnr": ("FLOAT", {"default": 0.10, "min": 0.0, "step": 0.01}),
                "rope_ssim": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
                "rope_lpips": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
                "rope_dists": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
                "rope_color": ("FLOAT", {"default": 0.10, "min": 0.0, "step": 0.01}),
                "rope_niqe": ("FLOAT", {"default": 0.10, "min": 0.0, "step": 0.01}),
                "rope_musiq": ("FLOAT", {"default": 1.00, "min": 0.0, "step": 0.1}),
                "rope_clip_iqa": ("FLOAT", {"default": 0.02, "min": 0.0, "step": 0.001}),
                "rope_dover": ("FLOAT", {"default": 0.02, "min": 0.0, "step": 0.001}),
                "hdi_credibility": ("FLOAT", {"default": 0.95, "min": 0.50, "max": 0.999, "step": 0.01}),
                "output_directory": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metrics_json", "artifact_path", "overall_decision")
    FUNCTION = "analyze"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True
    _dover_evaluator = None
    _dover_evaluator_device = None
    _dover_evaluator_opt = None
    _dover_tensor_registry: dict[str, Any] = {}
    _dover_reader_patched = False

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
    def _assert_alignment_pair(cls, a_meta, b_meta, label_a, label_b):
        mismatches = []
        for key in VIDEO_ALIGNMENT_KEYS:
            a = a_meta[key]
            b = b_meta[key]
            if key == "frame_rate":
                a = Fraction(a)
                b = Fraction(b)
            if a != b:
                mismatches.append(key)
        if mismatches:
            raise ValueError(
                f"alignment mismatch ({label_a} vs {label_b}): {', '.join(mismatches)}"
            )

    def _per_frame_nr(self, backend, frames_nhwc, frame_rate, enabled_nr) -> dict[str, list[float]]:
        """Compute per-frame NR metric values (no aggregation)."""
        out: dict[str, list[float]] = {}
        if not enabled_nr:
            return out
        frames_chw = _to_pyiqa_input(frames_nhwc)
        if "niqe" in enabled_nr:
            out["niqe"] = list(map(float, backend._run_pyiqa_per_frame("niqe", frames_chw)))
        if "musiq" in enabled_nr:
            out["musiq"] = list(map(float, backend._run_pyiqa_per_frame("musiq", frames_chw)))
        if "clip_iqa" in enabled_nr:
            out["clip_iqa"] = list(map(float, backend._run_pyiqa_per_frame("clipiqa", frames_chw)))
        if "dover_fused" in enabled_nr:
            backend._require_dover()
            tmp_video = backend._encode_frames_to_temp_mp4(frames_nhwc, frame_rate)
            try:
                out["dover_fused"] = [float(backend._run_dover_subprocess(tmp_video))]
            finally:
                tmp_video.unlink(missing_ok=True)
        return out

    def _per_frame_fr(self, backend, out_frames_nhwc, ref_frames_nhwc, enabled_fr) -> dict[str, list[float]]:
        """Compute per-frame FR metric values (no aggregation)."""
        out: dict[str, list[float]] = {}
        if not enabled_fr:
            return out
        if out_frames_nhwc.shape != ref_frames_nhwc.shape:
            raise ValueError(
                f"FR pair tensor shape mismatch: out={tuple(out_frames_nhwc.shape)} "
                f"ref={tuple(ref_frames_nhwc.shape)}"
            )
        out_chw = _to_pyiqa_input(out_frames_nhwc)
        ref_chw = _to_pyiqa_input(ref_frames_nhwc)
        if "psnr" in enabled_fr:
            out["psnr"] = list(map(float, backend._run_pyiqa_per_frame("psnr", out_chw, ref_chw)))
        if "ssim" in enabled_fr:
            out["ssim"] = list(map(float, backend._run_pyiqa_per_frame("ssim", out_chw, ref_chw)))
        if "lpips" in enabled_fr:
            out["lpips"] = list(map(float, backend._run_pyiqa_per_frame(
                "lpips", out_chw, ref_chw, net=backend._lpips_backbone
            )))
        if "dists" in enabled_fr:
            out["dists"] = list(map(float, backend._run_pyiqa_per_frame("dists", out_chw, ref_chw)))
        for name, values in backend._run_color_per_frame(
            out_frames_nhwc,
            ref_frames_nhwc,
            enabled_fr,
        ).items():
            out[name] = list(map(float, values))
        return out

    @staticmethod
    def _aggregate_block(per_frame: list[float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        block = _aggregate(per_frame)
        if extra:
            block.update(extra)
        return block

    @classmethod
    def _ensure_dover_imported(cls):
        if str(DOVER_ROOT) not in sys.path:
            sys.path.insert(0, str(DOVER_ROOT))

    @classmethod
    def _ensure_dover_video_reader_patched(cls):
        if cls._dover_reader_patched:
            return
        cls._ensure_dover_imported()
        from dover.datasets import dover_datasets as _dd  # type: ignore

        original_reader = _dd.VideoReader
        registry = cls._dover_tensor_registry

        class _TensorReader:
            def __init__(self, sentinel_path):
                key = str(sentinel_path)[len("tensor://"):]
                self._frames = registry[key]

            def __len__(self):
                return int(self._frames.shape[0])

            def __getitem__(self, idx):
                return self._frames[int(idx)]

        class _WebPReader:
            def __init__(self, webp_path):
                from PIL import Image, ImageSequence
                import numpy as np
                import torch

                im = Image.open(str(webp_path))
                frames_np = [
                    np.asarray(frame.convert("RGB"))
                    for frame in ImageSequence.Iterator(im)
                ]
                if not frames_np:
                    raise ValueError(f"WebP has zero frames: {webp_path}")
                self._frames = torch.from_numpy(np.stack(frames_np, axis=0)).to(
                    dtype=torch.uint8
                ).contiguous()

            def __len__(self):
                return int(self._frames.shape[0])

            def __getitem__(self, idx):
                return self._frames[int(idx)]

        def _dispatch(path, *args, **kwargs):
            path_str = str(path)
            if path_str.startswith("tensor://"):
                return _TensorReader(path_str)
            if path_str.lower().endswith(".webp"):
                return _WebPReader(path_str)
            return original_reader(path, *args, **kwargs)

        _dd.VideoReader = _dispatch
        cls._dover_reader_patched = True

    @classmethod
    def _ensure_dover_evaluator(cls, device: str):
        if cls._dover_evaluator is not None and cls._dover_evaluator_device == device:
            return cls._dover_evaluator, cls._dover_evaluator_opt
        import torch
        import yaml

        cls._ensure_dover_imported()
        from dover.models import DOVER  # type: ignore

        with open(DOVER_ROOT / "dover.yml", "r") as fp:
            opt = yaml.safe_load(fp)
        weights_path = (DOVER_ROOT / opt["test_load_path"]).resolve()
        evaluator = DOVER(**opt["model"]["args"]).to(device)
        evaluator.load_state_dict(torch.load(str(weights_path), map_location=device))
        evaluator.eval()
        cls._dover_evaluator = evaluator
        cls._dover_evaluator_device = device
        cls._dover_evaluator_opt = opt
        return evaluator, opt

    @staticmethod
    def _seed_dover_sampler():
        import random
        import numpy as np
        import torch

        random.seed(DOVER_SAMPLE_SEED)
        np.random.seed(DOVER_SAMPLE_SEED)
        torch.manual_seed(DOVER_SAMPLE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(DOVER_SAMPLE_SEED)

    @classmethod
    def _score_dover_source(cls, source_path: str, device: str) -> float:
        import math
        import torch

        cls._ensure_dover_video_reader_patched()
        evaluator, opt = cls._ensure_dover_evaluator(device)
        cls._ensure_dover_imported()
        from dover.datasets import dover_datasets as _dd  # type: ignore
        from dover.datasets import UnifiedFrameSampler  # type: ignore

        dopt = opt["data"]["val-l1080p"]["args"]
        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
        cls._seed_dover_sampler()
        views, _ = _dd.spatial_temporal_view_decomposition(
            source_path, dopt["sample_types"], temporal_samplers
        )
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )
        with torch.no_grad():
            results = [r.mean().item() for r in evaluator(views)]
        x = (
            (results[0] - 0.1107) / 0.07355 * 0.6104
            + (results[1] + 0.08285) / 0.03774 * 0.3896
        )
        return float(1.0 / (1.0 + math.exp(-x)))

    @classmethod
    def _score_dover_frames(cls, frames_nhwc_float01, device: str) -> float:
        import torch

        frames_u8 = (
            frames_nhwc_float01.detach().cpu().clamp(0.0, 1.0) * 255.0
        ).to(dtype=torch.uint8).contiguous()
        key = uuid.uuid4().hex
        cls._dover_tensor_registry[key] = frames_u8
        try:
            return cls._score_dover_source(f"tensor://{key}", device)
        finally:
            cls._dover_tensor_registry.pop(key, None)

    @classmethod
    def _score_dover_input(cls, source, frames_nhwc_float01, device: str) -> float:
        if isinstance(source, (str, Path)):
            return cls._score_dover_source(str(source), device)
        return cls._score_dover_frames(frames_nhwc_float01, device)

    @staticmethod
    def _rev3_anchor_block(
        ref_score: float,
        numz_score: float,
        native_score: float,
        floor_score: float,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        import math

        if not all(math.isfinite(v) for v in (ref_score, numz_score, native_score, floor_score)):
            return (
                {
                    "reference": None,
                    "numz": None,
                    "native": None,
                    "floor": None,
                },
                {"numz": None, "native": None},
                True,
            )
        reference_normal = abs(ref_score - floor_score)
        numz_normal = abs(numz_score - floor_score)
        native_normal = abs(native_score - floor_score)
        normalized = {
            "reference": reference_normal,
            "numz": numz_normal,
            "native": native_normal,
            "floor": 0.0,
        }
        if reference_normal < 1e-9:
            return normalized, {"numz": None, "native": None}, True
        return (
            normalized,
            {
                "numz": 100.0 * numz_normal / reference_normal,
                "native": 100.0 * native_normal / reference_normal,
            },
            False,
        )

    @staticmethod
    def _rev3_perf_delta(
        metric_name: str,
        ref_score: float,
        numz_score: float,
        native_score: float,
        floor_score: float,
    ) -> tuple[dict[str, Any], bool]:
        import math

        if not all(math.isfinite(v) for v in (ref_score, numz_score, native_score, floor_score)):
            return {"numz": None, "native": None}, True
        if metric_name in {"psnr", "ssim"}:
            return {
                "numz": numz_score - floor_score,
                "native": native_score - floor_score,
            }, False
        if metric_name in {"lpips", "dists", *COLOR_METRIC_NAMES}:
            return {
                "numz": floor_score - numz_score,
                "native": floor_score - native_score,
            }, False
        denom = ref_score - floor_score
        if abs(denom) < 1e-9:
            return {"numz": None, "native": None}, True
        return {
            "numz": (numz_score - floor_score) / denom,
            "native": (native_score - floor_score) / denom,
        }, False

    def _rev3_pyiqa_nr_score(
        self,
        backend,
        frames_nhwc,
        frame_rate,
        metric_name: str,
    ) -> float:
        values = self._per_frame_nr(backend, frames_nhwc, frame_rate, {metric_name}).get(
            metric_name,
            [],
        )
        if not values:
            raise ValueError(f"Rev3 metric produced no values: {metric_name}")
        return float(sum(values) / len(values))

    def _rev3_pyiqa_fr_score(
        self,
        backend,
        frames_nhwc,
        reference_frames_nhwc,
        metric_name: str,
    ) -> float:
        values = self._per_frame_fr(
            backend,
            frames_nhwc,
            reference_frames_nhwc,
            {metric_name},
        ).get(metric_name, [])
        if not values:
            raise ValueError(f"Rev3 metric produced no values: {metric_name}")
        return float(sum(values) / len(values))

    def analyze(
        self,
        video1,
        video2,
        reference=None,
        floor_video=None,
        lpips_backbone: str = "alex",
        enable_psnr: bool = True,
        enable_ssim: bool = True,
        enable_lpips: bool = True,
        enable_dists: bool = True,
        enable_color_metrics: bool = False,
        enable_niqe: bool = True,
        enable_musiq: bool = True,
        enable_clip_iqa: bool = True,
        enable_dover: bool = True,
        rope_psnr: float = 0.10,
        rope_ssim: float = 0.005,
        rope_lpips: float = 0.005,
        rope_dists: float = 0.005,
        rope_color: float = 0.10,
        rope_niqe: float = 0.10,
        rope_musiq: float = 1.00,
        rope_clip_iqa: float = 0.02,
        rope_dover: float = 0.02,
        hdi_credibility: float = 0.95,
        output_directory: str = "",
        output_filename: str = "",
    ):
        # Resolve enabled metric sets
        enabled_fr: set[str] = set()
        if enable_psnr:  enabled_fr.add("psnr")
        if enable_ssim:  enabled_fr.add("ssim")
        if enable_lpips: enabled_fr.add("lpips")
        if enable_dists: enabled_fr.add("dists")
        if enable_color_metrics: enabled_fr.update(COLOR_METRIC_NAMES)
        enabled_nr: set[str] = set()
        if enable_niqe:     enabled_nr.add("niqe")
        if enable_musiq:    enabled_nr.add("musiq")
        if enable_clip_iqa: enabled_nr.add("clip_iqa")
        rev3_enabled_fr: list[str] = []
        if enable_psnr:  rev3_enabled_fr.append("psnr")
        if enable_ssim:  rev3_enabled_fr.append("ssim")
        if enable_lpips: rev3_enabled_fr.append("lpips")
        if enable_dists: rev3_enabled_fr.append("dists")
        if enable_color_metrics: rev3_enabled_fr.extend(COLOR_METRIC_NAMES)
        rev3_enabled_nr: list[str] = []
        if enable_niqe:     rev3_enabled_nr.append("niqe")
        if enable_musiq:    rev3_enabled_nr.append("musiq")
        if enable_clip_iqa: rev3_enabled_nr.append("clip_iqa")
        if enable_dover:    rev3_enabled_nr.append("dover_fused")
        rev3_enabled_metrics = rev3_enabled_fr + rev3_enabled_nr
        run_rev3 = floor_video is not None and reference is not None and bool(rev3_enabled_metrics)
        if floor_video is not None and reference is None:
            raise ValueError("floor_video requires reference for Rev3 analysis")

        ropes = {
            "psnr": rope_psnr, "ssim": rope_ssim, "lpips": rope_lpips, "dists": rope_dists,
            "niqe": rope_niqe, "musiq": rope_musiq, "clip_iqa": rope_clip_iqa, "dover_fused": rope_dover,
        }
        ropes.update({name: rope_color for name in COLOR_METRIC_NAMES})

        # Output destination
        if output_directory.strip():
            artifact_dir = Path(output_directory).expanduser().resolve()
        else:
            artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if output_filename.strip():
            fname = output_filename.strip()
            if not fname.lower().endswith(".json"):
                fname = fname + ".json"
        else:
            fname = f"seedvr2_equivalence_{uuid.uuid4().hex}.json"
        artifact_path = artifact_dir / fname

        # Load all videos + alignment
        v1_frames, v1_fps = _frames_from_video(video1)
        v2_frames, v2_fps = _frames_from_video(video2)
        v1_meta = self._alignment_metadata(v1_frames, v1_fps)
        v2_meta = self._alignment_metadata(v2_frames, v2_fps)
        # NR pairing requires identical frame_count + frame_rate (paired across frames)
        self._assert_alignment_pair(v1_meta, v2_meta, "video1", "video2")
        ref_frames = ref_fps = ref_meta = None
        floor_frames = floor_fps = floor_meta = None
        run_fr = reference is not None and bool(enabled_fr)
        if reference is not None:
            ref_frames, ref_fps = _frames_from_video(reference)
            ref_meta = self._alignment_metadata(ref_frames, ref_fps)
            if run_fr:
                self._assert_alignment_pair(ref_meta, v1_meta, "reference", "video1")
                self._assert_alignment_pair(ref_meta, v2_meta, "reference", "video2")
        if floor_video is not None:
            floor_frames, floor_fps = _frames_from_video(floor_video)
            floor_meta = self._alignment_metadata(floor_frames, floor_fps)
            self._assert_alignment_pair(ref_meta, floor_meta, "reference", "floor_video")
            self._assert_alignment_pair(floor_meta, v1_meta, "floor_video", "video1")
            self._assert_alignment_pair(floor_meta, v2_meta, "floor_video", "video2")

        backend = self._metric_backend or SeedVR2MetricBackend(lpips_backbone=lpips_backbone)

        # Per-frame metric collection
        v1_nr = self._per_frame_nr(backend, v1_frames, v1_fps, enabled_nr)
        v2_nr = self._per_frame_nr(backend, v2_frames, v2_fps, enabled_nr)
        v1_fr: dict[str, list[float]] = {}
        v2_fr: dict[str, list[float]] = {}
        if run_fr:
            v1_fr = self._per_frame_fr(backend, v1_frames, ref_frames, enabled_fr)
            v2_fr = self._per_frame_fr(backend, v2_frames, ref_frames, enabled_fr)

        # Equivalence per metric (pair v1 vs v2 across frames)
        eq_results: dict[str, Any] = {}
        for k, vals1 in {**v1_fr, **v1_nr}.items():
            vals2 = v2_fr.get(k) if k in v1_fr else v2_nr.get(k)
            if vals2 is None or len(vals1) != len(vals2):
                eq_results[k] = {
                    "decision": "NOT_PAIRABLE",
                    "n_v1": len(vals1),
                    "n_v2": 0 if vals2 is None else len(vals2),
                    "rope_lo": -abs(ropes.get(k, 0.0)),
                    "rope_hi": abs(ropes.get(k, 0.0)),
                }
                continue
            diffs = [a - b for a, b in zip(vals1, vals2)]
            eq_results[k] = _bayes_normal_posterior_decision(diffs, ropes.get(k, 0.0), hdi_credibility)

        # Overall (bidirectional) equivalence verdict
        decisions = [r.get("decision") for r in eq_results.values()]
        if not decisions:
            overall = "NO_METRICS"
        elif all(d == "EQUIVALENT" for d in decisions):
            overall = "EQUIVALENT"
        elif any(d == "NOT_EQUIVALENT" for d in decisions):
            overall = "NOT_EQUIVALENT"
        else:
            overall = "UNDECIDED"

        # Joint non-inferiority verdict (video1 candidate vs video2 reference impl)
        joint_ni = _compute_joint_non_inferiority(eq_results, ropes)

        # Aggregated per-video stat blocks (mean/std/min/max/full per-frame)
        v1_block = {
            "fr": {k: self._aggregate_block(v) for k, v in v1_fr.items()},
            "nr": {k: self._aggregate_block(v) for k, v in v1_nr.items()},
        }
        v2_block = {
            "fr": {k: self._aggregate_block(v) for k, v in v2_fr.items()},
            "nr": {k: self._aggregate_block(v) for k, v in v2_nr.items()},
        }
        if "lpips" in v1_block["fr"]:
            v1_block["fr"]["lpips"]["backbone"] = backend._lpips_backbone
        if "lpips" in v2_block["fr"]:
            v2_block["fr"]["lpips"]["backbone"] = backend._lpips_backbone

        metrics_doc = {
            "schema_version": EQUIVALENCE_SCHEMA_VERSION,
            "node": "SeedVR2EquivalenceAnalysis",
            "inputs": {
                "video1": (repr(type(video1).__name__) if _is_video_object(video1) else str(video1)),
                "video2": (repr(type(video2).__name__) if _is_video_object(video2) else str(video2)),
                "reference": (
                    None if reference is None
                    else (repr(type(reference).__name__) if _is_video_object(reference) else str(reference))
                ),
            },
            "videos": {"video1": v1_meta, "video2": v2_meta, "reference": ref_meta},
            "enabled_metrics": {
                "fr": sorted(enabled_fr) if run_fr else [],
                "nr": sorted(enabled_nr),
            },
            "rope_half_widths": {k: ropes[k] for k in sorted(ropes.keys())},
            "metrics": {"video1": v1_block, "video2": v2_block},
            "equivalence": {
                "method": "Bayesian-Normal posterior on mu_diff (closed-form Student-t); HDI vs ROPE per Kruschke",
                "hdi_credibility": hdi_credibility,
                "results": eq_results,
                "overall_decision": overall,
            },
            "joint_non_inferiority": joint_ni,
            "tool_provenance": backend.tool_provenance,
        }

        if run_rev3:
            prev_cwd = os.getcwd()
            rev3_scores: dict[str, Any] = {}
            rev3_normalized: dict[str, Any] = {}
            rev3_raw_percent: dict[str, Any] = {}
            rev3_perf_delta: dict[str, Any] = {}
            rev3_denominator_unstable: dict[str, bool] = {}
            for metric_name in rev3_enabled_metrics:
                if metric_name == "dover_fused":
                    try:
                        os.chdir(str(DOVER_ROOT))
                        print("[SeedVR2EquivalenceAnalysis] Rev3 dover_fused scoring reference", flush=True)
                        ref_score = self._score_dover_input(reference, ref_frames, "cuda")
                        print("[SeedVR2EquivalenceAnalysis] Rev3 dover_fused scoring numz", flush=True)
                        numz_score = self._score_dover_input(video1, v1_frames, "cuda")
                        print("[SeedVR2EquivalenceAnalysis] Rev3 dover_fused scoring native", flush=True)
                        native_score = self._score_dover_input(video2, v2_frames, "cuda")
                        print("[SeedVR2EquivalenceAnalysis] Rev3 dover_fused scoring floor", flush=True)
                        floor_score = self._score_dover_input(floor_video, floor_frames, "cuda")
                    finally:
                        os.chdir(prev_cwd)
                elif metric_name in set(FR_METRIC_NAMES):
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring reference", flush=True)
                    ref_score = self._rev3_pyiqa_fr_score(backend, ref_frames, ref_frames, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring numz", flush=True)
                    numz_score = self._rev3_pyiqa_fr_score(backend, v1_frames, ref_frames, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring native", flush=True)
                    native_score = self._rev3_pyiqa_fr_score(backend, v2_frames, ref_frames, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring floor", flush=True)
                    floor_score = self._rev3_pyiqa_fr_score(backend, floor_frames, ref_frames, metric_name)
                else:
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring reference", flush=True)
                    ref_score = self._rev3_pyiqa_nr_score(backend, ref_frames, ref_fps, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring numz", flush=True)
                    numz_score = self._rev3_pyiqa_nr_score(backend, v1_frames, v1_fps, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring native", flush=True)
                    native_score = self._rev3_pyiqa_nr_score(backend, v2_frames, v2_fps, metric_name)
                    print(f"[SeedVR2EquivalenceAnalysis] Rev3 {metric_name} scoring floor", flush=True)
                    floor_score = self._rev3_pyiqa_nr_score(backend, floor_frames, floor_fps, metric_name)
                normalized, raw_percent, denom_unstable = self._rev3_anchor_block(
                    ref_score,
                    numz_score,
                    native_score,
                    floor_score,
                )
                perf_delta, perf_delta_unstable = self._rev3_perf_delta(
                    metric_name,
                    ref_score,
                    numz_score,
                    native_score,
                    floor_score,
                )
                rev3_scores[metric_name] = {
                    "reference": ref_score,
                    "numz": numz_score,
                    "native": native_score,
                    "floor": floor_score,
                }
                rev3_normalized[metric_name] = normalized
                rev3_raw_percent[metric_name] = raw_percent
                rev3_perf_delta[metric_name] = perf_delta
                rev3_denominator_unstable[metric_name] = denom_unstable or perf_delta_unstable
            metrics_doc["rev3"] = {
                "formula": "raw_percent_score = 100 * abs(candidate - floor) / abs(reference - floor)",
                "perf_delta_formula": {
                    "psnr": "actual - floor",
                    "ssim": "actual - floor",
                    "lpips": "floor - actual",
                    "dists": "floor - actual",
                    "deltae76": "floor - actual",
                    "deltae00": "floor - actual",
                    "lab_l_mae": "floor - actual",
                    "lab_a_mae": "floor - actual",
                    "lab_b_mae": "floor - actual",
                    "chroma_mae": "floor - actual",
                    "hue_mae_deg": "floor - actual",
                    "lab_hist_w1": "floor - actual",
                    "niqe": "(actual - floor) / (reference - floor)",
                    "musiq": "(actual - floor) / (reference - floor)",
                    "clip_iqa": "(actual - floor) / (reference - floor)",
                    "dover_fused": "(actual - floor) / (reference - floor)",
                },
                "videos": {
                    "reference": ref_meta,
                    "numz": v1_meta,
                    "native": v2_meta,
                    "floor": floor_meta,
                },
                "scores": rev3_scores,
                "normalized_delta_from_floor": rev3_normalized,
                "raw_percent_score": rev3_raw_percent,
                "perf_delta": rev3_perf_delta,
                "denominator_unstable": rev3_denominator_unstable,
            }
            metrics_json = json.dumps(metrics_doc, indent=2)
            artifact_path.write_text(metrics_json, encoding="utf-8")
            print(_render_rev3_text_table(metrics_doc), flush=True)
        else:
            metrics_json = json.dumps(metrics_doc, indent=2)
            artifact_path.write_text(metrics_json, encoding="utf-8")
        # Combined verdict surfaced as the third return for downstream nodes:
        # "{equivalence_overall}|{aggregate_non_inferiority}"
        combined = f"{overall}|{joint_ni.get('aggregate_verdict', 'UNDECIDED')}"
        return (metrics_json, str(artifact_path), combined)


class SeedVR2WorstFrameFidelityAnalysis:
    """Fast full-reference visual-fidelity check for chunk-boundary regressions."""

    def __init__(self, metric_backend=None):
        self._metric_backend = metric_backend

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("VIDEO",),
                "full_chunk": ("VIDEO",),
                "two_chunk": ("VIDEO",),
            },
            "optional": {
                "lpips_backbone": (["alex", "vgg"], {"default": "alex"}),
                "output_directory": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metrics_json", "artifact_path", "worst_summary")
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
    def _input_label(value) -> str:
        return repr(type(value).__name__) if _is_video_object(value) else str(value)

    @staticmethod
    def _per_frame_fr(backend, frames_nhwc, reference_frames_nhwc) -> dict[str, list[float]]:
        if frames_nhwc.shape != reference_frames_nhwc.shape:
            raise ValueError(
                f"FR pair tensor shape mismatch: output={tuple(frames_nhwc.shape)} "
                f"reference={tuple(reference_frames_nhwc.shape)}"
            )
        out_chw = _to_pyiqa_input(frames_nhwc)
        ref_chw = _to_pyiqa_input(reference_frames_nhwc)
        return {
            "psnr": list(map(float, backend._run_pyiqa_per_frame("psnr", out_chw, ref_chw))),
            "ssim": list(map(float, backend._run_pyiqa_per_frame("ssim", out_chw, ref_chw))),
            "lpips": list(map(float, backend._run_pyiqa_per_frame(
                "lpips", out_chw, ref_chw, net=backend._lpips_backbone
            ))),
            "dists": list(map(float, backend._run_pyiqa_per_frame("dists", out_chw, ref_chw))),
        }

    @staticmethod
    def _worse_label(full_value: float, two_value: float, higher_is_better: bool) -> str:
        if full_value == two_value:
            return "tie"
        if higher_is_better:
            return "full_chunk" if full_value < two_value else "two_chunk"
        return "full_chunk" if full_value > two_value else "two_chunk"

    @staticmethod
    def _worse_score(full_value: float, two_value: float, higher_is_better: bool) -> float:
        return min(full_value, two_value) if higher_is_better else max(full_value, two_value)

    @classmethod
    def _worst_frame(cls, rows: list[dict[str, Any]], higher_is_better: bool) -> dict[str, Any]:
        if not rows:
            raise ValueError("metric produced zero frames")
        key = lambda row: row["worst_score"]
        return min(rows, key=key) if higher_is_better else max(rows, key=key)

    @staticmethod
    def _tool_provenance() -> dict[str, Any]:
        def _pkg_version(name: str) -> str:
            try:
                return importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                raise RuntimeError(f"required package not installed: {name}")

        import torch

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
        }

    def analyze(
        self,
        reference,
        full_chunk,
        two_chunk,
        lpips_backbone: str = "alex",
        output_directory: str = "",
        output_filename: str = "",
    ):
        if output_directory.strip():
            artifact_dir = Path(output_directory).expanduser().resolve()
        else:
            artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if output_filename.strip():
            fname = output_filename.strip()
            if not fname.lower().endswith(".json"):
                fname = fname + ".json"
        else:
            fname = f"seedvr2_worst_frame_fidelity_{uuid.uuid4().hex}.json"
        artifact_path = artifact_dir / fname

        reference_frames, reference_fps = _frames_from_video(reference)
        full_frames, full_fps = _frames_from_video(full_chunk)
        two_frames, two_fps = _frames_from_video(two_chunk)

        reference_meta = SeedVR2EquivalenceAnalysis._alignment_metadata(reference_frames, reference_fps)
        full_meta = SeedVR2EquivalenceAnalysis._alignment_metadata(full_frames, full_fps)
        two_meta = SeedVR2EquivalenceAnalysis._alignment_metadata(two_frames, two_fps)
        SeedVR2EquivalenceAnalysis._assert_alignment_pair(reference_meta, full_meta, "reference", "full_chunk")
        SeedVR2EquivalenceAnalysis._assert_alignment_pair(reference_meta, two_meta, "reference", "two_chunk")

        backend = self._metric_backend or SeedVR2MetricBackend(lpips_backbone=lpips_backbone)
        full_metrics = self._per_frame_fr(backend, full_frames, reference_frames)
        two_metrics = self._per_frame_fr(backend, two_frames, reference_frames)

        metrics: dict[str, Any] = {}
        summary_lines: list[str] = []
        for metric_name in FAST_VISUAL_FIDELITY_METRICS:
            higher_is_better = METRIC_HIGHER_IS_BETTER[metric_name]
            full_values = full_metrics[metric_name]
            two_values = two_metrics[metric_name]
            if len(full_values) != len(two_values):
                raise ValueError(
                    f"metric frame count mismatch for {metric_name}: "
                    f"full_chunk={len(full_values)} two_chunk={len(two_values)}"
                )
            rows: list[dict[str, Any]] = []
            for i, (full_value, two_value) in enumerate(zip(full_values, two_values)):
                rows.append({
                    "frame_index": i,
                    "frame_number": i + 1,
                    "full_chunk": full_value,
                    "two_chunk": two_value,
                    "worst_score": self._worse_score(full_value, two_value, higher_is_better),
                    "worst_video": self._worse_label(full_value, two_value, higher_is_better),
                })
            worst = self._worst_frame(rows, higher_is_better)
            metrics[metric_name] = {
                "direction": "higher" if higher_is_better else "lower",
                "full_chunk": _aggregate(full_values),
                "two_chunk": _aggregate(two_values),
                "per_frame_worst": rows,
                "worst_frame": worst,
            }
            summary_lines.append(
                f"{metric_name}: frame {worst['frame_number']} "
                f"{worst['worst_video']} score={worst['worst_score']:.6f}"
            )

        metrics_doc = {
            "schema_version": WORST_FRAME_SCHEMA_VERSION,
            "node": "SeedVR2WorstFrameFidelityAnalysis",
            "inputs": {
                "reference": self._input_label(reference),
                "full_chunk": self._input_label(full_chunk),
                "two_chunk": self._input_label(two_chunk),
            },
            "videos": {
                "reference": reference_meta,
                "full_chunk": full_meta,
                "two_chunk": two_meta,
            },
            "metrics": metrics,
            "tool_provenance": self._tool_provenance(),
        }
        metrics_json = json.dumps(metrics_doc, indent=2)
        artifact_path.write_text(metrics_json, encoding="utf-8")
        worst_summary = " | ".join(summary_lines)
        return (metrics_json, str(artifact_path), worst_summary)


def _seedvr2_numz_root() -> Path:
    return REPO_ROOT.parent / "ComfyUI-SeedVR2_VideoUpscaler"


def _ensure_numz_import_path() -> None:
    root = str(_seedvr2_numz_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _numz_latent_to_native_5d(latent: Any) -> Any:
    import torch

    if not torch.is_tensor(latent):
        raise TypeError(f"expected tensor latent, got {type(latent).__name__}")
    latent = latent.detach()
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5 or latent.shape[-1] != 16:
        raise ValueError(
            "expected numz latent shape (T,H,W,16) or (B,T,H,W,16); "
            f"got {tuple(latent.shape)}"
        )
    return latent.movedim(-1, 1).contiguous()


def _native_5d_to_collapsed_samples(channel_first: Any) -> dict[str, Any]:
    b, c, t, h, w = channel_first.shape
    collapsed = channel_first.reshape(b, c * t, h, w).contiguous()
    return {"samples": collapsed}


def _numz_latent_to_native_samples(latent: Any) -> dict[str, Any]:
    return _native_5d_to_collapsed_samples(_numz_latent_to_native_5d(latent))


def _native_latent_to_numz_latent(latent: dict[str, Any]) -> Any:
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("native LATENT input is missing samples")
    if samples.ndim == 4:
        b, ct, h, w = samples.shape
        if ct % 16 != 0:
            raise ValueError(f"native collapsed latent channel count is not divisible by 16: {tuple(samples.shape)}")
        t = ct // 16
        samples = samples.reshape(b, 16, t, h, w)
    elif samples.ndim != 5:
        raise ValueError(f"expected native latent shape 4D or 5D, got {tuple(samples.shape)}")
    if samples.shape[1] != 16:
        raise ValueError(f"expected native channel-first latent with 16 channels, got {tuple(samples.shape)}")
    if samples.shape[0] != 1:
        raise ValueError(f"bridge expects batch size 1 for this probe, got {tuple(samples.shape)}")
    return samples[0].movedim(0, -1).contiguous()


def _debug_tensor_stats(tensor: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor, got {type(tensor).__name__}")
    t = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _debug_tensor_probe(tensor: Any, sample_count: int = 4096) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor, got {type(tensor).__name__}")
    flat = tensor.detach().float().flatten()
    sample = flat[: min(sample_count, flat.numel())].cpu()
    return {
        "stats": _debug_tensor_stats(tensor),
        "sample": sample,
    }


class SeedVR2NativeRawDiTProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 5770521, "min": 0, "max": 2**32 - 1, "step": 1}),
                "output_path": (
                    "STRING",
                    {
                        "default": "/home/johnj/dev_master/mydevelopment/github_issues/283/scratch/divergence_probe/native_raw_dit_probe.pt",
                    },
                ),
            }
            ,
            "optional": {
                "capture_blocks": ("BOOLEAN", {"default": True}),
                "norm_override": (["native", "fp32"], {"default": "native"}),
                "rope_override": (["native", "legacy"], {"default": "native"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2/debug"
    OUTPUT_NODE = True

    def execute(self, model, positive, latent_image, seed, output_path, capture_blocks=True, norm_override="native", rope_override="native"):
        import torch
        import comfy.model_management
        import comfy.sample
        import comfy.ldm.seedvr.model as seedvr_model

        path = Path(output_path)
        if "/scratch/" not in str(path):
            raise ValueError(f"SeedVR2NativeRawDiTProbe output_path must be under a scratch directory: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        if not positive:
            raise ValueError("SeedVR2NativeRawDiTProbe requires non-empty positive conditioning")
        context = positive[0][0]
        cond_meta = positive[0][1]
        if "condition" not in cond_meta:
            raise ValueError("SeedVR2NativeRawDiTProbe positive conditioning is missing 'condition'")
        condition = cond_meta["condition"]

        samples = latent_image["samples"]
        samples = comfy.sample.fix_empty_latent_channels(
            model,
            samples,
            latent_image.get("downscale_ratio_spacial", None),
            latent_image.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent_image.get("batch_index", None))

        comfy.model_management.load_models_gpu([model], force_full_load=True)
        device = model.load_device
        x_t = noise.to(device=device)
        sigma = torch.ones([x_t.shape[0]], device=device, dtype=torch.float32)
        context = context.to(device=device)
        condition = condition.to(device=device)

        block_probes: list[dict[str, Any]] = []
        submodule_probes: list[dict[str, Any]] = []
        hook_handles = []
        transformer_options = {"cond_or_uncond": [0]}
        if capture_blocks:
            diffusion_model = model.model.diffusion_model
            block0 = diffusion_model.blocks[0]

            def add_submodule_probe(name, module):
                def hook(mod, inputs, kwargs, output):
                    stage = name
                    if name.endswith(".ada"):
                        stage = f"{name}.{kwargs.get('layer', 'unknown')}.{kwargs.get('mode', 'unknown')}"
                    vid_out = output[0] if isinstance(output, tuple) else output
                    submodule_probes.append({"stage": stage, "vid": _debug_tensor_probe(vid_out)})
                hook_handles.append(module.register_forward_hook(hook, with_kwargs=True))

            add_submodule_probe("block_0.attn_norm", block0.attn_norm)
            add_submodule_probe("block_0.ada", block0.ada)
            add_submodule_probe("block_0.attn", block0.attn)
            add_submodule_probe("block_0.attn.proj_qkv", block0.attn.proj_qkv)
            add_submodule_probe("block_0.attn.norm_q", block0.attn.norm_q)
            add_submodule_probe("block_0.attn.norm_k", block0.attn.norm_k)
            if block0.attn.rope is not None:
                add_submodule_probe("block_0.attn.rope", block0.attn.rope)
            add_submodule_probe("block_0.attn.proj_out", block0.attn.proj_out)
            add_submodule_probe("block_0.mlp_norm", block0.mlp_norm)
            add_submodule_probe("block_0.mlp", block0.mlp)

            patches_replace = {"dit": {}}

            for block_idx in range(len(diffusion_model.blocks)):
                def make_block_probe(i):
                    def block_probe(args, extra):
                        if i == 0:
                            block_probes.append({"stage": "block_0_in", "vid": _debug_tensor_probe(args["vid"])})
                        out = extra["original_block"](args)
                        block_probes.append({"stage": f"block_{i}_out", "vid": _debug_tensor_probe(out["vid"])})
                        return out
                    return block_probe

                patches_replace["dit"][("block", block_idx)] = make_block_probe(block_idx)
            transformer_options["patches_replace"] = patches_replace

        original_norm_forward = None
        original_var_attention = None
        original_rope_forward = None
        original_mm_rope_forward = None
        if norm_override == "fp32":
            original_norm_forward = seedvr_model.CustomRMSNorm.forward

            def fp32_norm_forward(self, input):
                dims = tuple(range(-len(self.normalized_shape), 0))
                work = input.float()
                variance = work.pow(2).mean(dim=dims, keepdim=True)
                normalized = work / torch.sqrt(variance + self.eps)
                if self.elementwise_affine:
                    return normalized * self.weight.to(normalized.dtype)
                return normalized

            seedvr_model.CustomRMSNorm.forward = fp32_norm_forward
        if rope_override == "legacy":
            from rotary_embedding_torch import apply_rotary_emb as legacy_apply_rotary_emb

            original_rope_forward = seedvr_model.NaRotaryEmbedding3d.forward
            original_mm_rope_forward = seedvr_model.NaMMRotaryEmbedding3d.forward

            def legacy_rope_forward(self, q, k, shape, cache):
                freqs = cache("rope_freqs_3d_legacy", lambda: self.get_legacy_freqs(shape))
                freqs = freqs.to(device=q.device, dtype=q.dtype)
                q = seedvr_model.rearrange(q, "L h d -> h L d")
                k = seedvr_model.rearrange(k, "L h d -> h L d")
                q = legacy_apply_rotary_emb(freqs, q.float()).to(q.dtype)
                k = legacy_apply_rotary_emb(freqs, k.float()).to(k.dtype)
                q = seedvr_model.rearrange(q, "h L d -> L h d")
                k = seedvr_model.rearrange(k, "h L d -> L h d")
                return q, k

            def get_legacy_freqs(self, shape):
                plain_rope = seedvr_model.RotaryEmbedding(
                    dim=self.rope.freqs.numel() * 2,
                    freqs_for="pixel",
                    max_freq=seedvr_model.BYTEDANCE_ROPE_MAX_FREQ,
                )
                plain_rope = plain_rope.to(self.rope.dummy.device)
                freq_list = []
                for f, h, w in shape.tolist():
                    freqs = plain_rope.get_axial_freqs(f, h, w)
                    freq_list.append(freqs.view(-1, freqs.size(-1)))
                return torch.cat(freq_list, dim=0)

            seedvr_model.NaRotaryEmbedding3d.forward = legacy_rope_forward
            seedvr_model.NaRotaryEmbedding3d.get_legacy_freqs = get_legacy_freqs

            def legacy_mm_rope_forward(self, vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache):
                freqs = cache("rope_freqs_3d_legacy_plain_video", lambda: self.get_legacy_freqs(vid_shape))
                freqs = freqs.to(device=vid_q.device, dtype=vid_q.dtype)
                vid_q = seedvr_model.rearrange(vid_q, "L h d -> h L d")
                vid_k = seedvr_model.rearrange(vid_k, "L h d -> h L d")
                vid_q = legacy_apply_rotary_emb(freqs, vid_q.float()).to(vid_q.dtype)
                vid_k = legacy_apply_rotary_emb(freqs, vid_k.float()).to(vid_k.dtype)
                vid_q = seedvr_model.rearrange(vid_q, "h L d -> L h d")
                vid_k = seedvr_model.rearrange(vid_k, "h L d -> L h d")
                return vid_q, vid_k, txt_q, txt_k

            seedvr_model.NaMMRotaryEmbedding3d.forward = legacy_mm_rope_forward
        if capture_blocks:
            original_var_attention = seedvr_model.optimized_var_attention
            captured_var_attention = {"done": False}

            def probe_var_attention(*args, **kwargs):
                q = kwargs.get("q", args[0] if len(args) > 0 else None)
                k = kwargs.get("k", args[1] if len(args) > 1 else None)
                v = kwargs.get("v", args[2] if len(args) > 2 else None)
                if not captured_var_attention["done"]:
                    if q is not None:
                        submodule_probes.append({"stage": "block_0.attn.var.q", "vid": _debug_tensor_probe(q)})
                    if k is not None:
                        submodule_probes.append({"stage": "block_0.attn.var.k", "vid": _debug_tensor_probe(k)})
                    if v is not None:
                        submodule_probes.append({"stage": "block_0.attn.var.v", "vid": _debug_tensor_probe(v)})
                out = original_var_attention(*args, **kwargs)
                if not captured_var_attention["done"]:
                    submodule_probes.append({"stage": "block_0.attn.var.out", "vid": _debug_tensor_probe(out)})
                    captured_var_attention["done"] = True
                return out

            seedvr_model.optimized_var_attention = probe_var_attention

        with torch.no_grad():
            try:
                denoised = model.model.apply_model(
                    x_t,
                    sigma,
                    c_crossattn=context,
                    transformer_options=transformer_options,
                    condition=condition,
                )
            finally:
                if original_norm_forward is not None:
                    seedvr_model.CustomRMSNorm.forward = original_norm_forward
                if original_var_attention is not None:
                    seedvr_model.optimized_var_attention = original_var_attention
                if original_rope_forward is not None:
                    seedvr_model.NaRotaryEmbedding3d.forward = original_rope_forward
                if original_mm_rope_forward is not None:
                    seedvr_model.NaMMRotaryEmbedding3d.forward = original_mm_rope_forward
                for handle in hook_handles:
                    handle.remove()
        raw_pred = x_t.float() - denoised.float()

        artifact = {
            "schema": "seedvr2_native_raw_dit_probe.v1",
            "seed": seed,
            "norm_override": norm_override,
            "rope_override": rope_override,
            "block0_rope": {
                "class": type(diffusion_model.blocks[0].attn.rope).__name__ if capture_blocks else None,
                "mm": getattr(diffusion_model.blocks[0].attn.rope, "mm", None) if capture_blocks else None,
                "freqs_for": getattr(getattr(diffusion_model.blocks[0].attn.rope, "rope", None), "freqs_for", None) if capture_blocks else None,
                "block_version_7b": getattr(diffusion_model.blocks[0], "version", None) if capture_blocks else None,
                "model_7b_version": getattr(diffusion_model, "_7b_version", None),
            },
            "sampler_edge": {
                "sigma": [float(v) for v in sigma.detach().cpu().tolist()],
                "scheduler": "simple",
                "sampler": "euler",
                "steps": 1,
                "raw_pred_reconstruction": "x_t - denoised at sigma=1",
            },
            "stats": {
                "x_t": _debug_tensor_stats(x_t),
                "condition": _debug_tensor_stats(condition),
                "context": _debug_tensor_stats(context),
                "denoised": _debug_tensor_stats(denoised),
                "raw_pred": _debug_tensor_stats(raw_pred),
            },
            "block_probes": block_probes,
            "submodule_probes": submodule_probes,
            "tensors": {
                "x_t": x_t.detach().cpu(),
                "condition": condition.detach().cpu(),
                "context": context.detach().cpu(),
                "denoised": denoised.detach().cpu(),
                "raw_pred": raw_pred.detach().cpu(),
            },
        }
        torch.save(artifact, path)

        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "schema": artifact["schema"],
                    "seed": seed,
                    "stats": artifact["stats"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return (str(sidecar),)


class SeedVR2NumzPreparedConditioningToNative:
    @classmethod
    def INPUT_TYPES(cls):
        _ensure_numz_import_path()
        from src.optimization.memory_manager import get_device_list

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("SEEDVR2_VAE",),
                "seed": ("INT", {"default": 5770521, "min": 0, "max": 2**32 - 1, "step": 1}),
                "resolution": ("INT", {"default": 1314, "min": 16, "max": 16384, "step": 2}),
                "max_resolution": ("INT", {"default": 4096, "min": 0, "max": 16384, "step": 2}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1}),
                "uniform_batch_size": ("BOOLEAN", {"default": False}),
                "temporal_overlap": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "prepend_frames": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab"}),
                "input_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "offload_device": (get_device_list(include_none=True, include_cpu=True), {"default": "cpu"}),
                "enable_debug": ("BOOLEAN", {"default": False}),
                "capture_blocks": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "original_image", "upscaled_shorter_edge")
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2/debug"

    def execute(
        self,
        image,
        model,
        vae,
        seed,
        resolution=1314,
        max_resolution=4096,
        batch_size=1,
        uniform_batch_size=False,
        temporal_overlap=0,
        prepend_frames=0,
        color_correction="lab",
        input_noise_scale=0.0,
        offload_device="cpu",
        enable_debug=False,
    ):
        _ensure_numz_import_path()
        from src.interfaces.latent_nodes import SeedVR2VAEEncode
        from comfy_extras.nodes_seedvr import SeedVR2Conditioning

        encoded = SeedVR2VAEEncode.execute(
            image=image,
            vae=vae,
            seed=seed,
            resolution=resolution,
            max_resolution=max_resolution,
            batch_size=batch_size,
            uniform_batch_size=uniform_batch_size,
            temporal_overlap=temporal_overlap,
            prepend_frames=prepend_frames,
            color_correction=color_correction,
            input_noise_scale=input_noise_scale,
            offload_device=offload_device,
            enable_debug=enable_debug,
        )[0]
        latents = encoded.get("all_latents")
        if not latents:
            raise ValueError("numz VAE encode returned no latents")
        native_5d = _numz_latent_to_native_5d(latents[0])
        conditioned = SeedVR2Conditioning.execute(model, {"samples": native_5d})
        return (
            conditioned[0],
            conditioned[1],
            conditioned[2],
            conditioned[3],
            image,
            int(resolution),
        )


class SeedVR2NumzUpscaledLatentFileToNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": (
                    "STRING",
                    {
                        "default": str(
                            Path("/home/johnj/dev_master/mydevelopment")
                            / "github_issues/283/scratch/numz_upscaled_latent_sharp_fp8_ee8ecf65.pt"
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2/debug"

    def execute(self, latent_path):
        import torch

        path = Path(latent_path)
        if not path.is_file():
            raise FileNotFoundError(f"numz upscaled latent file not found: {path}")
        latent = torch.load(path, map_location="cpu", weights_only=False)
        return (_numz_latent_to_native_samples(latent),)


class SeedVR2NativeLatentToNumzDiT:
    @classmethod
    def INPUT_TYPES(cls):
        _ensure_numz_import_path()
        from src.optimization.memory_manager import get_device_list

        return {
            "required": {
                "image": ("IMAGE",),
                "native_latent": ("LATENT",),
                "dit": ("SEEDVR2_DIT",),
                "vae": ("SEEDVR2_VAE",),
                "seed": ("INT", {"default": 5770521, "min": 0, "max": 2**32 - 1, "step": 1}),
                "resolution": ("INT", {"default": 1314, "min": 16, "max": 16384, "step": 2}),
                "max_resolution": ("INT", {"default": 4096, "min": 0, "max": 16384, "step": 2}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab"}),
                "latent_noise_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "offload_device": (get_device_list(include_none=True, include_cpu=True), {"default": "cpu"}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2/debug"

    def execute(
        self,
        image,
        native_latent,
        dit,
        vae,
        seed,
        resolution=1314,
        max_resolution=4096,
        batch_size=1,
        color_correction="lab",
        latent_noise_scale=0.0,
        offload_device="cpu",
        enable_debug=False,
    ):
        _ensure_numz_import_path()
        import torch

        from src.core.generation_phases import decode_all_batches, postprocess_all_batches, upscale_all_batches
        from src.core.generation_utils import compute_generation_info, prepare_runner, setup_generation_context
        from src.optimization.memory_manager import cleanup_text_embeddings, complete_cleanup, manage_model_device
        from src.utils.constants import get_base_cache_dir
        from src.utils.debug import Debug

        debug = Debug(enabled=enable_debug)
        runner = None
        ctx = None
        dit_model = dit["model"]
        vae_model = vae["model"]
        dit_device = torch.device(dit["device"])
        vae_device = torch.device(vae["device"])
        dit_offload_str = dit.get("offload_device", "none")
        vae_offload_str = vae.get("offload_device", "none")
        tensor_offload_device = torch.device(offload_device) if offload_device != "none" else None
        block_swap_config = None
        if dit.get("blocks_to_swap", 0) > 0 or dit.get("swap_io_components", False):
            block_swap_config = {
                "blocks_to_swap": dit.get("blocks_to_swap", 0),
                "swap_io_components": dit.get("swap_io_components", False),
            }
            if dit_offload_str != "none":
                block_swap_config["offload_device"] = torch.device(dit_offload_str)

        try:
            ctx = setup_generation_context(
                dit_device=dit_device,
                vae_device=vae_device,
                dit_offload_device=torch.device(dit_offload_str) if dit_offload_str != "none" else None,
                vae_offload_device=torch.device(vae_offload_str) if vae_offload_str != "none" else None,
                tensor_offload_device=tensor_offload_device,
                debug=debug,
            )
            runner, cache_context = prepare_runner(
                dit_model=dit_model,
                vae_model=vae_model,
                model_dir=get_base_cache_dir(),
                debug=debug,
                ctx=ctx,
                dit_cache=dit.get("cache_model", False),
                vae_cache=vae.get("cache_model", False),
                dit_id=dit.get("node_id"),
                vae_id=vae.get("node_id"),
                block_swap_config=block_swap_config,
                encode_tiled=vae.get("encode_tiled", False),
                encode_tile_size=(vae.get("encode_tile_size", 512), vae.get("encode_tile_size", 512)),
                encode_tile_overlap=(vae.get("encode_tile_overlap", 64), vae.get("encode_tile_overlap", 64)),
                decode_tiled=vae.get("decode_tiled", False),
                decode_tile_size=(vae.get("decode_tile_size", 512), vae.get("decode_tile_size", 512)),
                decode_tile_overlap=(vae.get("decode_tile_overlap", 64), vae.get("decode_tile_overlap", 64)),
                tile_debug=vae.get("tile_debug", "false"),
                attention_mode=dit.get("attention_mode", "sdpa"),
                torch_compile_args_dit=dit.get("torch_compile_args"),
                torch_compile_args_vae=vae.get("torch_compile_args"),
            )
            ctx["cache_context"] = cache_context
            rgb_image = image[..., :3].contiguous()
            rgb_image, _ = compute_generation_info(
                ctx=ctx,
                images=rgb_image,
                resolution=resolution,
                max_resolution=max_resolution,
                batch_size=batch_size,
                uniform_batch_size=False,
                seed=seed,
                prepend_frames=0,
                temporal_overlap=0,
                debug=debug,
            )
            ctx["input_images"] = rgb_image
            ctx["is_rgba"] = False
            ctx["actual_temporal_overlap"] = 0
            ctx["all_ori_lengths"] = [rgb_image.shape[0]]
            ctx["all_latents"] = [_native_latent_to_numz_latent(native_latent)]
            if color_correction != "none":
                ctx["batch_metadata"] = [(0, rgb_image.shape[0], 0)]

            ctx = upscale_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                seed=seed,
                latent_noise_scale=latent_noise_scale,
                cache_model=dit.get("cache_model", False),
            )
            ctx = decode_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                cache_model=vae.get("cache_model", False),
            )
            ctx = postprocess_all_batches(
                ctx=ctx,
                debug=debug,
                color_correction=color_correction,
                prepend_frames=0,
                temporal_overlap=0,
                batch_size=batch_size,
            )
            sample = ctx["final_video"]
            if sample.is_cuda or sample.is_mps:
                sample = sample.cpu()
            if sample.dtype != torch.float32:
                sample = sample.to(torch.float32)
            return (sample,)
        finally:
            if runner is not None:
                complete_cleanup(
                    runner=runner,
                    debug=debug,
                    dit_cache=dit.get("cache_model", False),
                    vae_cache=vae.get("cache_model", False),
                )
            if ctx is not None:
                cleanup_text_embeddings(ctx, debug)


class SeedVR2NumzRawDiTFromNativeProbe:
    @classmethod
    def INPUT_TYPES(cls):
        _ensure_numz_import_path()
        from src.optimization.memory_manager import get_device_list

        return {
            "required": {
                "dit": ("SEEDVR2_DIT",),
                "vae": ("SEEDVR2_VAE",),
                "native_probe_path": (
                    "STRING",
                    {
                        "default": "/home/johnj/dev_master/mydevelopment/github_issues/283/scratch/divergence_probe/native_raw_dit_probe.pt",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "/home/johnj/dev_master/mydevelopment/github_issues/283/scratch/divergence_probe/numz_raw_from_native_probe.pt",
                    },
                ),
                "offload_device": (get_device_list(include_none=True, include_cpu=True), {"default": "cpu"}),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2/debug"
    OUTPUT_NODE = True

    def execute(self, dit, vae, native_probe_path, output_path, offload_device="cpu", enable_debug=False, capture_blocks=True):
        _ensure_numz_import_path()
        import torch

        from src.core.generation_utils import ensure_precision_initialized, prepare_runner, setup_generation_context
        from src.core.model_loader import materialize_model
        from src.optimization.memory_manager import cleanup_text_embeddings, complete_cleanup, manage_model_device
        from src.utils.constants import get_base_cache_dir
        from src.utils.debug import Debug

        output = Path(output_path)
        if "/scratch/" not in str(output):
            raise ValueError(f"SeedVR2NumzRawDiTFromNativeProbe output_path must be under a scratch directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        probe_path = Path(native_probe_path)
        native = torch.load(probe_path, map_location="cpu", weights_only=False)

        debug = Debug(enabled=enable_debug)
        runner = None
        ctx = None
        dit_offload_str = dit.get("offload_device", "none")
        vae_offload_str = vae.get("offload_device", "none")
        tensor_offload_device = torch.device(offload_device) if offload_device != "none" else None
        block_swap_config = None
        if dit.get("blocks_to_swap", 0) > 0 or dit.get("swap_io_components", False):
            block_swap_config = {
                "blocks_to_swap": dit.get("blocks_to_swap", 0),
                "swap_io_components": dit.get("swap_io_components", False),
            }
            if dit_offload_str != "none":
                block_swap_config["offload_device"] = torch.device(dit_offload_str)

        try:
            ctx = setup_generation_context(
                dit_device=torch.device(dit["device"]),
                vae_device=torch.device(vae["device"]),
                dit_offload_device=torch.device(dit_offload_str) if dit_offload_str != "none" else None,
                vae_offload_device=torch.device(vae_offload_str) if vae_offload_str != "none" else None,
                tensor_offload_device=tensor_offload_device,
                debug=debug,
            )
            runner, cache_context = prepare_runner(
                dit_model=dit["model"],
                vae_model=vae["model"],
                model_dir=get_base_cache_dir(),
                debug=debug,
                ctx=ctx,
                dit_cache=dit.get("cache_model", False),
                vae_cache=vae.get("cache_model", False),
                dit_id=dit.get("node_id"),
                vae_id=vae.get("node_id"),
                block_swap_config=block_swap_config,
                encode_tiled=vae.get("encode_tiled", False),
                encode_tile_size=(vae.get("encode_tile_size", 512), vae.get("encode_tile_size", 512)),
                encode_tile_overlap=(vae.get("encode_tile_overlap", 64), vae.get("encode_tile_overlap", 64)),
                decode_tiled=vae.get("decode_tiled", False),
                decode_tile_size=(vae.get("decode_tile_size", 512), vae.get("decode_tile_size", 512)),
                decode_tile_overlap=(vae.get("decode_tile_overlap", 64), vae.get("decode_tile_overlap", 64)),
                tile_debug=vae.get("tile_debug", "false"),
                attention_mode=dit.get("attention_mode", "sdpa"),
                torch_compile_args_dit=dit.get("torch_compile_args"),
                torch_compile_args_vae=vae.get("torch_compile_args"),
            )
            ctx["cache_context"] = cache_context
            if runner.dit and next(runner.dit.parameters()).device.type == "meta":
                materialize_model(runner, "dit", ctx["dit_device"], runner.config, debug)
            ensure_precision_initialized(ctx, runner, debug)
            manage_model_device(
                model=runner.dit,
                target_device=ctx["dit_device"],
                model_name="DiT",
                debug=debug,
                runner=runner,
            )

            device = ctx["dit_device"]
            dtype = ctx["compute_dtype"]
            x_t = native["tensors"]["x_t"].to(device=device, dtype=dtype)
            condition = native["tensors"]["condition"].to(device=device, dtype=dtype)
            context = native["tensors"]["context"].to(device=device, dtype=dtype)

            b, ct, h, w = x_t.shape
            if b != 1 or ct % 16 != 0:
                raise ValueError(f"expected native x_t shape (1,16*T,H,W), got {tuple(x_t.shape)}")
            latent_t = ct // 16
            x_numz = x_t.reshape(b, 16, latent_t, h, w)[0].movedim(0, -1).contiguous()
            cond_numz = condition.reshape(b, 17, latent_t, h, w)[0].movedim(0, -1).contiguous()
            vid = torch.cat([x_numz.flatten(0, -2), cond_numz.flatten(0, -2)], dim=-1)
            vid_shape = torch.tensor([[latent_t, h, w]], device=device, dtype=torch.long)
            txt = context.squeeze(0).contiguous()
            txt_shape = torch.tensor([[txt.shape[0]]], device=device, dtype=torch.long)
            timestep = torch.tensor([1000.0], device=device, dtype=dtype)

            block_probes: list[dict[str, Any]] = []
            submodule_probes: list[dict[str, Any]] = []
            hook_handles = []
            if capture_blocks:
                block0 = runner.dit.blocks[0]

                def add_submodule_probe(name, module):
                    def hook(mod, inputs, kwargs, output):
                        stage = name
                        if name.endswith(".ada"):
                            stage = f"{name}.{kwargs.get('layer', 'unknown')}.{kwargs.get('mode', 'unknown')}"
                        vid_out = output[0] if isinstance(output, tuple) else output
                        submodule_probes.append({"stage": stage, "vid": _debug_tensor_probe(vid_out)})
                    hook_handles.append(module.register_forward_hook(hook, with_kwargs=True))

                add_submodule_probe("block_0.attn_norm", block0.attn_norm)
                add_submodule_probe("block_0.ada", block0.ada)
                add_submodule_probe("block_0.attn", block0.attn)
                add_submodule_probe("block_0.attn.proj_qkv", block0.attn.proj_qkv)
                add_submodule_probe("block_0.attn.norm_q", block0.attn.norm_q)
                add_submodule_probe("block_0.attn.norm_k", block0.attn.norm_k)
                if block0.attn.rope is not None:
                    add_submodule_probe("block_0.attn.rope", block0.attn.rope)
                add_submodule_probe("block_0.attn.proj_out", block0.attn.proj_out)
                add_submodule_probe("block_0.mlp_norm", block0.mlp_norm)
                add_submodule_probe("block_0.mlp", block0.mlp)

                def attention_call_probe(module, inputs, kwargs, output):
                    for key in ("q", "k", "v"):
                        value = kwargs.get(key)
                        if value is not None:
                            submodule_probes.append({"stage": f"block_0.attn.var.{key}", "vid": _debug_tensor_probe(value)})
                    submodule_probes.append({"stage": "block_0.attn.var.out", "vid": _debug_tensor_probe(output)})

                if hasattr(block0.attn, "attn"):
                    hook_handles.append(block0.attn.attn.register_forward_hook(attention_call_probe, with_kwargs=True))

                def pre_block_0(module, inputs, kwargs):
                    block_probes.append({"stage": "block_0_in", "vid": _debug_tensor_probe(kwargs["vid"])})

                hook_handles.append(runner.dit.blocks[0].register_forward_pre_hook(pre_block_0, with_kwargs=True))

                for block_idx, block in enumerate(runner.dit.blocks):
                    def make_hook(i):
                        def hook(module, inputs, kwargs, output):
                            block_probes.append({"stage": f"block_{i}_out", "vid": _debug_tensor_probe(output[0])})
                        return hook

                    hook_handles.append(block.register_forward_hook(make_hook(block_idx), with_kwargs=True))

            with torch.no_grad():
                try:
                    if device.type == "cuda":
                        with torch.autocast(device.type, dtype, enabled=True):
                            raw_flat = runner.dit(
                                vid=vid,
                                txt=txt,
                                vid_shape=vid_shape,
                                txt_shape=txt_shape,
                                timestep=timestep,
                            ).vid_sample
                    else:
                        raw_flat = runner.dit(
                            vid=vid,
                            txt=txt,
                            vid_shape=vid_shape,
                            txt_shape=txt_shape,
                            timestep=timestep,
                        ).vid_sample
                finally:
                    for handle in hook_handles:
                        handle.remove()

            raw_numz = raw_flat.reshape(latent_t, h, w, 16).movedim(-1, 0).reshape(1, 16 * latent_t, h, w)
            native_raw = native["tensors"]["raw_pred"].to(dtype=raw_numz.dtype)
            diff = raw_numz.detach().cpu().float() - native_raw.detach().cpu().float()

            artifact = {
                "schema": "seedvr2_numz_raw_from_native_probe.v1",
                "native_probe_path": str(probe_path),
                "stats": {
                    "vid": _debug_tensor_stats(vid),
                    "txt": _debug_tensor_stats(txt),
                    "raw_numz": _debug_tensor_stats(raw_numz),
                    "native_raw": _debug_tensor_stats(native_raw),
                    "diff": _debug_tensor_stats(diff),
                },
                "diff": {
                    "max_abs": float(diff.abs().max().item()),
                    "mean_abs": float(diff.abs().mean().item()),
                },
                "block_probes": block_probes,
                "submodule_probes": submodule_probes,
                "tensors": {
                    "raw_numz": raw_numz.detach().cpu(),
                    "native_raw": native_raw.detach().cpu(),
                    "diff": diff.detach().cpu(),
                },
            }
            torch.save(artifact, output)
            sidecar = output.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "path": str(output),
                        "sha256": _file_sha256(output),
                        "schema": artifact["schema"],
                        "stats": artifact["stats"],
                        "diff": artifact["diff"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return (str(sidecar),)
        finally:
            if runner is not None:
                complete_cleanup(
                    runner=runner,
                    debug=debug,
                    dit_cache=dit.get("cache_model", False),
                    vae_cache=vae.get("cache_model", False),
                )
            if ctx is not None:
                cleanup_text_embeddings(ctx, debug)


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
