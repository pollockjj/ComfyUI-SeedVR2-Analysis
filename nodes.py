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


def _frames_from_webp(path: Path) -> tuple[Any, Fraction]:
    import numpy as np
    import torch
    from PIL import Image, ImageSequence

    im = Image.open(path)
    frames_np = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(im)]
    if not frames_np:
        raise ValueError(f"WebP has zero frames: {path}")
    images_u8 = np.stack(frames_np, axis=0)
    images = torch.from_numpy(images_u8).to(dtype=torch.float32).div_(255.0).contiguous()
    return images, WEBP_DEFAULT_FRAME_RATE


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
                # FR metric toggles (require reference_video)
                "enable_psnr": ("BOOLEAN", {"default": True}),
                "enable_ssim": ("BOOLEAN", {"default": True}),
                "enable_lpips": ("BOOLEAN", {"default": True}),
                "enable_dists": ("BOOLEAN", {"default": True}),
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


EQUIVALENCE_SCHEMA_VERSION = "2.1"

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
    "niqe": False,
}


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


def _render_rev3_dover_text_table(metrics_doc: dict) -> str:
    block = metrics_doc.get("rev3_dover", {}) or {}
    videos = block.get("videos", {}) or {}
    scores = block.get("scores", {}) or {}
    raw = block.get("raw_percent_score", {}) or {}

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

    def _fmt_percent(value, width=14, precision=2):
        if value is None:
            return f"{'--':>{width}}"
        try:
            return f"{float(value):>{width - 1}.{precision}f}%"
        except (TypeError, ValueError):
            return f"{'--':>{width}}"

    lines = []
    lines.append("=" * 96)
    lines.append("SeedVR2 Rev3 DOVER Analysis  -  reference / numz / native / floor")
    lines.append("=" * 96)
    lines.append(f"reference (HQ):       {_video_label('reference')}")
    lines.append(f"numz:                 {_video_label('numz')}")
    lines.append(f"native:               {_video_label('native')}")
    lines.append(f"floor (ESRGAN):       {_video_label('floor')}")
    lines.append("-" * 96)
    lines.append(
        f"{'metric':<14} {'reference':>14} {'numz':>14} "
        f"{'native':>14} {'floor':>14} {'numz_pct':>14} {'native_pct':>14}"
    )
    lines.append("-" * 96)
    lines.append(
        f"{'dover_fused':<14} {_fmt(scores.get('reference'))} "
        f"{_fmt(scores.get('numz'))} {_fmt(scores.get('native'))} "
        f"{_fmt(scores.get('floor'))} {_fmt_percent(raw.get('numz'))} "
        f"{_fmt_percent(raw.get('native'))}"
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
                "enable_niqe": ("BOOLEAN", {"default": True}),
                "enable_musiq": ("BOOLEAN", {"default": True}),
                "enable_clip_iqa": ("BOOLEAN", {"default": True}),
                "enable_dover": ("BOOLEAN", {"default": True}),
                # ROPE half-widths (operator must justify these on prior grounds)
                "rope_psnr": ("FLOAT", {"default": 0.10, "min": 0.0, "step": 0.01}),
                "rope_ssim": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
                "rope_lpips": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
                "rope_dists": ("FLOAT", {"default": 0.005, "min": 0.0, "step": 0.001}),
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
    def _rev3_dover_block(
        ref_score: float,
        floor_score: float,
        video1_score: float,
        video2_score: float,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        reference_normal = abs(ref_score - floor_score)
        video1_normal = abs(video1_score - floor_score)
        video2_normal = abs(video2_score - floor_score)
        normalized = {
            "reference": reference_normal,
            "numz": video1_normal,
            "native": video2_normal,
            "floor": 0.0,
        }
        if reference_normal < 1e-9:
            return normalized, {"numz": None, "native": None}, True
        return (
            normalized,
            {
                "numz": 100.0 * video1_normal / reference_normal,
                "native": 100.0 * video2_normal / reference_normal,
            },
            False,
        )

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
        enable_niqe: bool = True,
        enable_musiq: bool = True,
        enable_clip_iqa: bool = True,
        enable_dover: bool = True,
        rope_psnr: float = 0.10,
        rope_ssim: float = 0.005,
        rope_lpips: float = 0.005,
        rope_dists: float = 0.005,
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
        enabled_nr: set[str] = set()
        if enable_niqe:     enabled_nr.add("niqe")
        if enable_musiq:    enabled_nr.add("musiq")
        if enable_clip_iqa: enabled_nr.add("clip_iqa")
        run_rev3_dover = floor_video is not None and reference is not None and enable_dover
        if floor_video is not None and reference is None:
            raise ValueError("floor_video requires reference for Rev3 DOVER analysis")

        ropes = {
            "psnr": rope_psnr, "ssim": rope_ssim, "lpips": rope_lpips, "dists": rope_dists,
            "niqe": rope_niqe, "musiq": rope_musiq, "clip_iqa": rope_clip_iqa, "dover_fused": rope_dover,
        }

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

        metrics_json = json.dumps(metrics_doc, indent=2)
        artifact_path.write_text(metrics_json, encoding="utf-8")
        try:
            print(_render_equivalence_text_table(metrics_doc), flush=True)
        except Exception:
            pass
        if run_rev3_dover:
            prev_cwd = os.getcwd()
            try:
                os.chdir(str(DOVER_ROOT))
                print("[SeedVR2EquivalenceAnalysis] Rev3 DOVER scoring reference", flush=True)
                ref_dover = self._score_dover_input(reference, ref_frames, "cuda")
                print("[SeedVR2EquivalenceAnalysis] Rev3 DOVER scoring floor", flush=True)
                floor_dover = self._score_dover_input(floor_video, floor_frames, "cuda")
                print("[SeedVR2EquivalenceAnalysis] Rev3 DOVER scoring numz", flush=True)
                video1_dover = self._score_dover_input(video1, v1_frames, "cuda")
                print("[SeedVR2EquivalenceAnalysis] Rev3 DOVER scoring native", flush=True)
                video2_dover = self._score_dover_input(video2, v2_frames, "cuda")
            finally:
                os.chdir(prev_cwd)
            normalized, raw_percent, denom_unstable = self._rev3_dover_block(
                ref_dover,
                floor_dover,
                video1_dover,
                video2_dover,
            )
            metrics_doc["rev3_dover"] = {
                "formula": "raw_percent_score = 100 * abs(candidate - floor) / abs(reference - floor)",
                "videos": {
                    "reference": ref_meta,
                    "numz": v1_meta,
                    "native": v2_meta,
                    "floor": floor_meta,
                },
                "scores": {
                    "reference": ref_dover,
                    "numz": video1_dover,
                    "native": video2_dover,
                    "floor": floor_dover,
                },
                "normalized_delta_from_floor": normalized,
                "raw_percent_score": raw_percent,
                "denominator_unstable": denom_unstable,
            }
            metrics_json = json.dumps(metrics_doc, indent=2)
            artifact_path.write_text(metrics_json, encoding="utf-8")
            print(_render_rev3_dover_text_table(metrics_doc), flush=True)
        # Combined verdict surfaced as the third return for downstream nodes:
        # "{equivalence_overall}|{aggregate_non_inferiority}"
        combined = f"{overall}|{joint_ni.get('aggregate_verdict', 'UNDECIDED')}"
        return (metrics_json, str(artifact_path), combined)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Analysis": SeedVR2Analysis,
    "SeedVR2EquivalenceAnalysis": SeedVR2EquivalenceAnalysis,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Analysis": "SeedVR2 Analysis",
    "SeedVR2EquivalenceAnalysis": "SeedVR2 Equivalence Analysis (BEST + ROPE)",
}
