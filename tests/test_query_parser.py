"""
Unit Tests for Query Parser and Change Router.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import unittest
from src.part1_retrieval.query_parser import QueryParser
from src.part1_retrieval.query_schema import ParsedQuery, ChangeType
from src.part1_retrieval.change_router import ChangeRouter
from src.part1_retrieval.gazetteer import Gazetteer


class TestQueryParser(unittest.TestCase):

    def setUp(self):
        self.parser = QueryParser()

    # --- Test Required Benchmark Queries ---

    def test_q1_agricultural_areas_near_chennai(self):
        q = "Find agricultural areas near Chennai."
        res = self.parser.parse(q)
        self.assertIn("agricultural areas", res.semantic_targets)
        self.assertIn("Chennai", res.location_names)
        self.assertFalse(res.change_intent)
        self.assertEqual(res.requested_change_type, ChangeType.STATIC_SEMANTIC_SEARCH.value)
        self.assertIsNotNone(res.geometry_constraints)
        self.assertIsNone(res.start_date)
        self.assertIsNone(res.end_date)
        # Should have spatial relation for "near Chennai"
        self.assertTrue(any(r["relation"] == "near" and r["reference"] == "Chennai" for r in res.spatial_relations))

    def test_q2_industrial_buildings_within_2km_of_highways(self):
        q = "Find industrial buildings within 2 km of highways."
        res = self.parser.parse(q)
        self.assertIn("industrial buildings", res.semantic_targets)
        self.assertFalse(res.change_intent)
        self.assertEqual(res.requested_change_type, ChangeType.STATIC_SEMANTIC_SEARCH.value)
        self.assertEqual(len(res.location_names), 0)
        self.assertIsNone(res.geometry_constraints)
        
        # Spatial distance assertion
        matched_rel = next((r for r in res.spatial_relations if r["relation"] == "within"), None)
        self.assertIsNotNone(matched_rel)
        self.assertEqual(matched_rel["distance_km"], 2.0)
        self.assertEqual(matched_rel["reference"], "highways")

    def test_q3_water_bodies_in_2024(self):
        q = "Show water bodies in 2024."
        res = self.parser.parse(q)
        self.assertIn("water bodies", res.semantic_targets)
        self.assertEqual(res.start_date, "2024-01-01")
        self.assertEqual(res.end_date, "2024-12-31")
        self.assertFalse(res.change_intent)
        self.assertEqual(res.requested_change_type, ChangeType.STATIC_SEMANTIC_SEARCH.value)

    def test_q4_areas_changed_between_2020_and_2025(self):
        q = "Find areas that changed between 2020 and 2025."
        res = self.parser.parse(q)
        self.assertTrue(res.change_intent)
        self.assertEqual(res.requested_change_type, ChangeType.GENERIC_CHANGE.value)
        self.assertEqual(res.start_date, "2020-01-01")
        self.assertEqual(res.end_date, "2025-12-31")

    def test_q5_road_development(self):
        q = "Find road development."
        res = self.parser.parse(q)
        self.assertTrue(res.change_intent)
        self.assertEqual(res.requested_change_type, ChangeType.ROAD_DEVELOPMENT.value)
        self.assertIn("roads", res.semantic_targets)
        # Check rule 7: missing constraints must be None, not invented
        self.assertIsNone(res.start_date)
        self.assertIsNone(res.end_date)
        self.assertEqual(len(res.location_names), 0)
        self.assertTrue(any("requires_confirmation" in w for w in res.warnings))

    # --- Test Rule 7: Never Invent Constraints ---

    def test_no_invented_constraints(self):
        q = "Find warehouses."
        res = self.parser.parse(q)
        self.assertIsNone(res.start_date)
        self.assertIsNone(res.end_date)
        self.assertEqual(res.location_names, [])
        self.assertIsNone(res.geometry_constraints)
        self.assertEqual(res.sensor_constraints, [])
        self.assertIsNone(res.resolution_constraints)
        self.assertIsNone(res.initial_state)
        self.assertIsNone(res.final_state)

    # --- Test Change Router Classifications ---

    def test_change_types_routing(self):
        router = ChangeRouter()
        
        cases = [
            ("Demolition and land clearance near border", ChangeType.CLEARANCE),
            ("New industrial construction in Mumbai", ChangeType.CONSTRUCTION_CHANGE),
            ("Highway expansion and road construction", ChangeType.ROAD_DEVELOPMENT),
            ("Reservoir expansion and flooding in 2023", ChangeType.WATER_EXPANSION),
            ("Lake drying up and severe drought", ChangeType.WATER_CONTRACTION),
            ("Deforestation and vegetation loss", ChangeType.VEGETATION_LOSS),
            ("Urban expansion and city growth in Bengaluru", ChangeType.URBAN_EXPANSION),
            ("Unidentified object appeared on airfield", ChangeType.GENERIC_APPEARANCE),
            ("Storage tanks vanished from port", ChangeType.GENERIC_DISAPPEARANCE),
            ("Multi-temporal change between 2021 and 2024", ChangeType.GENERIC_CHANGE),
            ("High resolution optical satellite view of airport", ChangeType.STATIC_SEMANTIC_SEARCH),
        ]

        for text, expected_type in cases:
            intent, routed_type, _, _ = router.route(text)
            if expected_type == ChangeType.STATIC_SEMANTIC_SEARCH:
                self.assertFalse(intent, f"Query '{text}' should not have change intent")
            else:
                self.assertTrue(intent, f"Query '{text}' should have change intent")
            self.assertEqual(routed_type, expected_type, f"Failed for query: {text}")

    # --- Test Distance Conversion ---

    def test_distance_units_conversion(self):
        # Meters
        res_m = self.parser.parse("Find buildings within 500 m of highways.")
        self.assertEqual(res_m.spatial_relations[0]["distance_km"], 0.5)

        # Kilometers
        res_km = self.parser.parse("Find buildings within 15 km of highways.")
        self.assertEqual(res_km.spatial_relations[0]["distance_km"], 15.0)

    # --- Test Sensors & Resolution ---

    def test_sensors_and_resolution(self):
        q = "Show Sentinel-2 and Cartosat-3 imagery with sub-meter resolution."
        res = self.parser.parse(q)
        self.assertIn("Sentinel-2", res.sensor_constraints)
        self.assertIn("Cartosat-3", res.sensor_constraints)
        self.assertIsNotNone(res.resolution_constraints)
        self.assertEqual(res.resolution_constraints["max_resolution_m"], 1.0)

    # --- Test Serialization & Deserialization ---

    def test_parsed_query_serialization(self):
        q = "Find agricultural areas near Chennai in 2024."
        parsed = self.parser.parse(q)
        data_dict = parsed.to_dict()
        reconstructed = ParsedQuery.from_dict(data_dict)

        self.assertEqual(parsed.raw_query, reconstructed.raw_query)
        self.assertEqual(parsed.semantic_targets, reconstructed.semantic_targets)
        self.assertEqual(parsed.location_names, reconstructed.location_names)
        self.assertEqual(parsed.requested_change_type, reconstructed.requested_change_type)


if __name__ == "__main__":
    unittest.main()
