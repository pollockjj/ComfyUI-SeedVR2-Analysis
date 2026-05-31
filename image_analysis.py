"""Image-mode comparative analysis (augments the SeedVR2-Analysis pack).

Takes a reference (HR), a floor (ESRGAN baseline), and two candidate inputs;
reuses the pack's pyiqa FR + color metrics (SeedVR2MetricBackend.compute_fr_metrics)
on a single still. Drops the video-only legs (DOVER) and per-frame variance
(a still has one value). For RGBA inputs the RGB FR is computed on the
alpha-composited (premultiplied) content so the invisible transparent regions
don't pollute the score, and the alpha is compared separately (MAE + PSNR).

Output: per-metric value for {input1, input2, floor} vs reference, the better
direction, the input1-vs-input2 winner, whether each beats the floor, and an
overall verdict. JSON + readable table.

CLI: image_analysis.py --reference R --floor F --input1 I1 --input2 I2
                        [--label1 native] [--label2 numz] [--out result.json]
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import cv2
import torch

_PACK = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("seedvr2_analysis_nodes", os.path.join(_PACK, "nodes.py"))
_nodes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nodes)

FR = list(_nodes.FR_METRIC_NAMES)
HIGHER_BETTER = {"psnr", "ssim", "alpha_psnr"}


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cannot read " + path)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB); alpha = None
    elif img.shape[-1] == 4:
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB); alpha = img[..., 3].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB); alpha = None
    return rgb.astype(np.float32) / 255.0, alpha


def _resize(arr, H, W):
    if arr.shape[0] == H and arr.shape[1] == W:
        return arr
    interp = cv2.INTER_AREA if arr.shape[0] > H else cv2.INTER_CUBIC
    return cv2.resize(arr, (W, H), interpolation=interp)


def _nhwc(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb))[None].float()


def score(cand_rgb, cand_alpha, ref_rgb, ref_alpha, backend):
    H, W = ref_rgb.shape[:2]
    c = np.clip(_resize(cand_rgb, H, W), 0.0, 1.0)
    if ref_alpha is not None:
        ca = np.clip(_resize(cand_alpha, H, W), 0.0, 1.0) if cand_alpha is not None else np.ones((H, W), np.float32)
        c_fr = c * ca[..., None]
        r_fr = ref_rgb * ref_alpha[..., None]
    else:
        c_fr, r_fr = c, ref_rgb
    c_fr = np.clip(c_fr, 0.0, 1.0)
    r_fr = np.clip(r_fr, 0.0, 1.0)
    res = backend.compute_fr_metrics(_nhwc(c_fr), _nhwc(r_fr))
    out = {m: float(res[m]["mean"]) for m in res}
    if ref_alpha is not None:
        ca = _resize(cand_alpha, H, W) if cand_alpha is not None else np.ones((H, W), np.float32)
        mse = float(np.mean((ca - ref_alpha) ** 2))
        out["alpha_mae"] = float(np.abs(ca - ref_alpha).mean())
        out["alpha_psnr"] = 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--input1", required=True)
    ap.add_argument("--input2", required=True)
    ap.add_argument("--label1", default="input1")
    ap.add_argument("--label2", default="input2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    backend = _nodes.SeedVR2MetricBackend(lpips_backbone="alex")
    ref_rgb, ref_alpha = _load(a.reference)
    legs = {}
    for name, path in [(a.label1, a.input1), (a.label2, a.input2), ("floor", a.floor)]:
        rgb, alpha = _load(path)
        legs[name] = score(rgb, alpha, ref_rgb, ref_alpha, backend)

    metrics = [m for m in (FR + (["alpha_mae", "alpha_psnr"] if ref_alpha is not None else [])) if m in legs[a.label1]]
    rows, wins = [], {a.label1: 0, a.label2: 0}
    for m in metrics:
        hb = m in HIGHER_BETTER
        v1, v2, vf = legs[a.label1][m], legs[a.label2][m], legs["floor"][m]
        winner = a.label1 if ((v1 > v2) == hb) else (a.label2 if v1 != v2 else "tie")
        if winner in wins:
            wins[winner] += 1
        beats = lambda v: ("yes" if ((v > vf) == hb) else "no")
        rows.append(dict(metric=m, dir=("higher" if hb else "lower"),
                         **{a.label1: v1, a.label2: v2, "floor": vf,
                            "winner": winner, a.label1 + "_beats_floor": beats(v1), a.label2 + "_beats_floor": beats(v2)}))
    verdict = a.label1 if wins[a.label1] > wins[a.label2] else (a.label2 if wins[a.label2] > wins[a.label1] else "tie")
    result = dict(reference=a.reference, floor=a.floor,
                  inputs={a.label1: a.input1, a.label2: a.input2},
                  rgba=ref_alpha is not None, metrics=rows,
                  metric_wins=wins, verdict=verdict)

    print("%-12s %10s %12s %12s %12s  %-8s" % ("metric", "dir", a.label1, a.label2, "floor", "winner"))
    for r in rows:
        print("%-12s %10s %12.4f %12.4f %12.4f  %-8s" % (r["metric"], r["dir"], r[a.label1], r[a.label2], r["floor"], r["winner"]))
    print("\nmetric wins: %s=%d  %s=%d  -> verdict: %s" % (a.label1, wins[a.label1], a.label2, wins[a.label2], verdict))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
