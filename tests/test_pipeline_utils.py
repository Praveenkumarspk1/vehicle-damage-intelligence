import unittest

import cv2
import numpy as np

from pipeline_utils import (
    DamageRecord,
    RepairQuote,
    calculate_indian_repair_quote,
    cosine_similarity,
    estimate_severity,
    extract_visual_embedding,
    hybrid_reid_match,
    project_to_3d_coordinates,
    validate_image_quality,
)


CONFIG = {
    "severity_thresholds": {"minor_max_ratio": 0.02, "moderate_max_ratio": 0.07},
    "location_tiers": {"Tier 2": {"denting_labor_hr": 360.0, "paint_booth_panel": 2200.0, "multiplier": 1.0}},
    "workshop_types": {"OEM": {"labor_markup": 1.0, "parts_markup": 1.0, "gst_rate": 0.18}},
    "vehicle_models": {"Test": {"parts": {"bonnet-dent": 4200.0, "default_part": 3000.0}}},
    "quality_checks": {"min_blur_score": 0.0, "min_brightness": 0.0, "max_brightness": 255.0, "min_contrast": 0.0},
}


def make_record(damage_id="front_1", size_rank=1):
    quote = RepairQuote("repair", "Bolt-On Sheet Metal Panel", 1.0, 100.0, 200.0, 0.0, 54.0, 354.0, "test")
    return DamageRecord(
        damage_id=damage_id,
        class_name="bonnet-dent",
        slot="front",
        rel_center=[0.5, 0.5],
        rel_area=0.01,
        size_bucket="Minor",
        size_rank=size_rank,
        confidence=0.9,
        bbox=[10.0, 10.0, 50.0, 50.0],
        quote=quote,
        visual_embedding=[1.0] + [0.0] * 127,
    )


class PipelineUtilsTests(unittest.TestCase):
    def test_severity_thresholds_and_critical_damage(self):
        self.assertEqual(estimate_severity(0.01, "bonnet-dent", CONFIG)[:2], ("Minor", 1))
        self.assertEqual(estimate_severity(0.03, "bonnet-dent", CONFIG)[:2], ("Moderate", 2))
        self.assertEqual(estimate_severity(0.001, "Headlight-damage", CONFIG)[:2], ("Severe", 3))

    def test_embedding_is_normalized_and_deterministic(self):
        image = np.full((80, 80, 3), 120, dtype=np.uint8)
        first = extract_visual_embedding(image, [10, 10, 60, 60])
        second = extract_visual_embedding(image, [10, 10, 60, 60])
        self.assertEqual(len(first), 128)
        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine_similarity(first, first), 1.0, places=5)

    def test_matching_marks_existing_and_new_damage(self):
        baseline = [make_record().__dict__]
        baseline[0]["quote"] = {"total_cost": 354.0}
        existing = make_record("current_existing")
        new = make_record("current_new")
        new.rel_center = [0.9, 0.9]
        evaluated, unmatched, liability = hybrid_reid_match(baseline, [existing, new], tolerance=0.08)
        self.assertEqual(evaluated[0].verdict, "EXISTING")
        self.assertEqual(evaluated[1].verdict, "NEW")
        self.assertEqual(len(unmatched), 0)
        self.assertGreater(liability, 0.0)

    def test_quote_includes_gst(self):
        quote = calculate_indian_repair_quote("bonnet-dent", "Moderate", 0.03, "Test", "Tier 2", "OEM", CONFIG)
        self.assertGreater(quote.total_cost, quote.gst_amount)
        self.assertAlmostEqual(quote.total_cost, quote.denting_labor_cost + quote.paint_booth_cost + quote.part_cost + quote.gst_amount)

    def test_quality_and_3d_projection_return_valid_values(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        quality = validate_image_quality(image, CONFIG)
        self.assertFalse(quality.glare_detected)
        point = project_to_3d_coordinates("front", "bonnet-dent", [0.5, 0.5])
        self.assertEqual(len(point), 3)
        self.assertTrue(all(np.isfinite(value) for value in point))


if __name__ == "__main__":
    unittest.main()
