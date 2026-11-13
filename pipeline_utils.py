"""ClearQuote Gap-Solving Vehicle Damage Intelligence & Automotive Costing Engine.

Canonical facade re-exporting from cost_engine.indian_repair_quote to eliminate duplicate code.
Maintains 100% backward compatibility with existing tests and scripts.
"""

from __future__ import annotations

from cost_engine.indian_repair_quote import (
    DamageRecord,
    QualityResult,
    RepairQuote,
    annotate_frame,
    build_vehicle_mesh_traces,
    calculate_indian_repair_quote,
    check_specular_glare,
    cosine_similarity,
    diff_damage_records,
    estimate_severity,
    estimate_turnaround_and_downtime,
    evaluate_insurance_claim,
    extract_damage_records,
    extract_learned_visual_embedding,
    extract_visual_embedding,
    fast_video_walkaround,
    get_panel_material,
    hybrid_reid_match,
    load_surveyor_audit_log,
    log_surveyor_feedback,
    project_to_3d_coordinates,
    validate_image_quality,
)

__all__ = [
    "DamageRecord",
    "QualityResult",
    "RepairQuote",
    "annotate_frame",
    "build_vehicle_mesh_traces",
    "calculate_indian_repair_quote",
    "check_specular_glare",
    "cosine_similarity",
    "diff_damage_records",
    "estimate_severity",
    "estimate_turnaround_and_downtime",
    "evaluate_insurance_claim",
    "extract_damage_records",
    "extract_learned_visual_embedding",
    "extract_visual_embedding",
    "fast_video_walkaround",
    "get_panel_material",
    "hybrid_reid_match",
    "load_surveyor_audit_log",
    "log_surveyor_feedback",
    "project_to_3d_coordinates",
    "validate_image_quality",
]
