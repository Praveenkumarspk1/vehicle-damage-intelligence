from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def extract_classical_embedding(img_bgr: np.ndarray, bbox: List[float]) -> List[float]:
    if cv2 is None:
        return [0.0] * 128
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return [0.0] * 128
    crop = img_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (32, 32))
    hsv_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_crop], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv_crop], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_crop], [2], None, [16], [0, 256]).flatten()
    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    gray_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    cell_h, cell_w = 8, 8
    texture_feat = []
    for row in range(4):
        for col in range(4):
            cy1, cy2 = row * cell_h, (row + 1) * cell_h
            cx1, cx2 = col * cell_w, (col + 1) * cell_w
            cell_mag = mag[cy1:cy2, cx1:cx2].flatten()
            cell_ang = ang[cy1:cy2, cx1:cx2].flatten()
            hist, _ = np.histogram(cell_ang, bins=5, range=(0, 2 * np.pi), weights=cell_mag)
            texture_feat.append(hist)
    texture_feat_arr = np.concatenate(texture_feat)
    combined = np.concatenate([color_feat, texture_feat_arr])
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm
    return combined.tolist()


def extract_reid_embedding(model: Any, img_bgr: np.ndarray, bbox: List[float], embedding_mode: str = "classical") -> List[float]:
    if embedding_mode == "learned" and model is not None and hasattr(model, "embed"):
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return []
        embeddings = model.embed(img_bgr[y1:y2, x1:x2], verbose=False)
        if not embeddings:
            return []
        vector = embeddings[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm <= 1e-6:
            return []
        return (vector / norm).tolist()
    return extract_classical_embedding(img_bgr, bbox)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a, b = np.array(vec_a, dtype=np.float32), np.array(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def hybrid_reid_match(
    baseline_list: List[Dict[str, Any]],
    current_records: List[Any],
    tolerance: float = 0.12,
    alpha: float = 0.65,
) -> Tuple[List[Any], List[Dict[str, Any]], float]:
    matched_ids: set = set()
    new_liability_total = 0.0
    for cur in current_records:
        best_match = None
        best_score = -1.0
        for base in baseline_list:
            if base["damage_id"] in matched_ids:
                continue
            if base["slot"] != cur.slot or base["class_name"] != cur.class_name:
                continue
            spatial_dist = math.hypot(
                cur.rel_center[0] - base["rel_center"][0],
                cur.rel_center[1] - base["rel_center"][1],
            )
            if spatial_dist > tolerance * 2.0:
                continue
            spatial_score = max(0.0, 1.0 - spatial_dist / tolerance)
            base_emb = base.get("visual_embedding")
            cur_emb = cur.visual_embedding
            if base_emb and cur_emb and len(base_emb) == len(cur_emb):
                vis_score = cosine_similarity(base_emb, cur_emb)
            else:
                vis_score = 0.5
            hybrid_score = alpha * vis_score + (1.0 - alpha) * spatial_score
            if hybrid_score > best_score and hybrid_score > 0.35:
                best_score = hybrid_score
                best_match = base
        if best_match:
            matched_ids.add(best_match["damage_id"])
            cur.matched_id = best_match["damage_id"]
            cur.reid_score = round(best_score, 3)
            if cur.size_rank > best_match.get("size_rank", 0):
                cur.verdict = "WORSENED"
                base_cost = best_match.get("quote", {}).get("total_cost", 0.0)
                diff_cost = max(0.0, cur.quote.total_cost - base_cost)
                new_liability_total += diff_cost
            else:
                cur.verdict = "EXISTING"
        else:
            cur.verdict = "NEW"
            cur.reid_score = None
            new_liability_total += cur.quote.total_cost
    unmatched_baseline = [b for b in baseline_list if b["damage_id"] not in matched_ids]
    return current_records, unmatched_baseline, round(new_liability_total, 2)


def spatial_only_match(
    baseline_list: List[Dict[str, Any]],
    current_records: List[Any],
    tolerance: float = 0.08,
) -> Tuple[List[Any], List[Dict[str, Any]], float]:
    return hybrid_reid_match(baseline_list, current_records, tolerance=tolerance, alpha=0.0)


__all__ = [
    "cosine_similarity",
    "extract_classical_embedding",
    "extract_reid_embedding",
    "hybrid_reid_match",
    "spatial_only_match",
]
