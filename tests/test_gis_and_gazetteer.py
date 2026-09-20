"""
Unit Tests for Geographic Understanding, Gazetteer, Metadata Filtering, and GIS Verifier.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology

Mandated Scenarios:
1. Chennai Resolution (ResolvedPlace, coordinates, admin hierarchy, offline)
2. Ambiguous Locations (Handling ambiguity safely, never silently choosing incorrect place, disambiguating with context)
3. Date, Sensor, Resolution, Quality, and AOI Filtering (Pre-vector search catalog reduction)
4. 500m / 1km Distance GIS Verification (Deterministic PostGIS-style dwithin, buffer, contains, intersects)
"""

import unittest
from typing import List

from src.part1_retrieval.gazetteer import (
    Gazetteer,
    ResolvedPlace,
    AmbiguousLocationError,
)
from src.part1_retrieval.gis_geometry import (
    Point,
    LineString,
    Polygon,
    BoundingBox,
    haversine_distance,
    to_projected_metric,
    from_projected_metric,
)
from src.part1_retrieval.metadata_filter import (
    MetadataFilter,
    IndexedTile,
    generate_mock_archive,
)
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.query_parser import QueryParser
from src.contracts.candidate_aoi import CandidateAOI


class TestGISAndGazetteer(unittest.TestCase):

    def setUp(self):
        self.gazetteer = Gazetteer()
        self.metadata_filter = MetadataFilter()
        self.gis_verifier = GISVerifier()
        self.query_parser = QueryParser(gazetteer=self.gazetteer)
        self.archive = generate_mock_archive()

    # =========================================================================
    # SCENARIO 1: CHENNAI RESOLUTION
    # =========================================================================

    def test_chennai_resolution(self):
        """Verify offline resolution of Chennai to a typed ResolvedPlace structure."""
        place = self.gazetteer.resolve("Chennai")
        self.assertIsInstance(place, ResolvedPlace)
        self.assertEqual(place.name, "Chennai")
        self.assertEqual(place.canonical_name, "Chennai, Tamil Nadu")
        self.assertEqual(place.place_type, "city")
        self.assertEqual(place.admin_level, 4)
        self.assertEqual(place.country, "India")
        self.assertEqual(place.parent_admin, "Tamil Nadu")
        self.assertEqual(place.source, "defence_offline_gazetteer_v2")
        self.assertGreaterEqual(place.confidence, 0.95)

        # Centroid check: Chennai is approx (80.27, 13.08)
        lon, lat = place.centroid
        self.assertAlmostEqual(lon, 80.2707, places=2)
        self.assertAlmostEqual(lat, 13.0827, places=2)

        # Geometry bounds check
        bbox = place.get_bounding_box()
        self.assertTrue(bbox[0] < lon < bbox[2])
        self.assertTrue(bbox[1] < lat < bbox[3])

    def test_chennai_alias_madras(self):
        """Test historical alias 'Madras' resolves accurately to Chennai."""
        place = self.gazetteer.resolve("Madras")
        self.assertIsInstance(place, ResolvedPlace)
        self.assertEqual(place.canonical_name, "Chennai, Tamil Nadu")

    # =========================================================================
    # SCENARIO 2: AMBIGUOUS LOCATIONS HANDLING
    # =========================================================================

    def test_ambiguous_location_raises_error_without_context(self):
        """
        DEFENCE REQUIREMENT: Never silently select an incorrect place.
        'Aurangabad' exists in both Maharashtra and Bihar.
        Without context, resolve() must raise AmbiguousLocationError.
        """
        with self.assertRaises(AmbiguousLocationError) as ctx:
            self.gazetteer.resolve("Aurangabad")

        err = ctx.exception
        self.assertEqual(err.query_name.lower(), "aurangabad")
        self.assertGreaterEqual(len(err.candidates), 2)
        candidate_states = [c.parent_admin for c in err.candidates]
        self.assertIn("Maharashtra", candidate_states)
        self.assertIn("Bihar", candidate_states)

    def test_ambiguous_location_disambiguated_with_context(self):
        """Context hints (e.g., 'Maharashtra' or 'Bihar') must cleanly disambiguate."""
        # 1. Disambiguate to Maharashtra
        place_mh = self.gazetteer.resolve("Aurangabad", context_hint="Maharashtra")
        self.assertIsInstance(place_mh, ResolvedPlace)
        self.assertEqual(place_mh.canonical_name, "Aurangabad, Maharashtra")
        self.assertEqual(place_mh.parent_admin, "Maharashtra")

        # 2. Disambiguate to Bihar
        place_br = self.gazetteer.resolve("Aurangabad", context_hint="Bihar")
        self.assertIsInstance(place_br, ResolvedPlace)
        self.assertEqual(place_br.canonical_name, "Aurangabad, Bihar")
        self.assertEqual(place_br.parent_admin, "Bihar")

    def test_ambiguous_location_in_query_parser(self):
        """Query parsing text with ambiguous location records a warning for confirmation."""
        parsed = self.query_parser.parse("Find airfields near Aurangabad")
        self.assertTrue(any("requires_confirmation" in w and "aurangabad" in w.lower() for w in parsed.warnings))

    # =========================================================================
    # SCENARIO 3: DATE / SENSOR / QUALITY / AOI METADATA FILTERING
    # =========================================================================

    def test_aoi_filtering(self):
        """Filter archive to Chennai bounding box (80.15, 13.00, 80.35, 13.20)."""
        chennai_bbox = (80.15, 13.00, 80.35, 13.20)
        filtered = self.metadata_filter.filter_by_aoi(self.archive, target_bbox=chennai_bbox)

        # Should only retain tiles intersecting Chennai, excluding Ladakh, Pokhran, Mumbai
        self.assertTrue(len(filtered) > 0)
        for tile in filtered:
            self.assertTrue("CHE" in tile.tile_id)

    def test_date_filtering(self):
        """Filter archive to scenes acquired in 2024."""
        filtered = self.metadata_filter.filter_by_date(
            self.archive,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        self.assertTrue(len(filtered) > 0)
        for tile in filtered:
            self.assertTrue(tile.datetime.startswith("2024"))

    def test_sensor_filtering(self):
        """Filter archive strictly to Sentinel-2 sensor tiles."""
        filtered = self.metadata_filter.filter_by_sensor(self.archive, allowed_sensors=["Sentinel-2"])
        self.assertTrue(len(filtered) > 0)
        for tile in filtered:
            self.assertEqual(tile.sensor, "Sentinel-2")

    def test_resolution_filtering(self):
        """Filter archive for sub-meter resolution (GSD <= 1.0 m) e.g., Cartosat-3."""
        filtered = self.metadata_filter.filter_by_resolution(self.archive, max_resolution_m=1.0)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].sensor, "Cartosat-3")
        self.assertAlmostEqual(filtered[0].resolution, 0.28)

    def test_quality_filtering_rejects_clouds(self):
        """Reject scenes with > 20% cloud cover or unusable data flag."""
        filtered = self.metadata_filter.filter_by_quality(self.archive, max_cloud_percent=20.0)
        tile_ids = [t.tile_id for t in filtered]
        # The cloud-contaminated tile must be rejected
        self.assertNotIn("TILE-CHE-S2-20240722-CLOUD", tile_ids)

    def test_pipeline_filter_archive(self):
        """End-to-end 6-stage metadata filtering audit trail."""
        parsed = self.query_parser.parse("Find agricultural areas near Chennai in 2024 using Sentinel-2")
        surviving, audit = self.metadata_filter.filter_archive(self.archive, parsed, max_cloud_cover=15.0)

        self.assertGreaterEqual(audit["initial_tile_count"], len(self.archive))
        self.assertGreater(audit["dropped_geographic"], 0)
        self.assertGreaterEqual(len(surviving), 1)
        for tile in surviving:
            self.assertIn("CHE", tile.tile_id)
            self.assertEqual(tile.sensor, "Sentinel-2")
            self.assertTrue(tile.datetime.startswith("2024"))

    # =========================================================================
    # SCENARIO 4: 500M / 1KM DETERMINISTIC GIS DISTANCE VERIFICATION
    # =========================================================================

    def test_projected_crs_distance_calculation(self):
        """Verify metric distance between two points in Chennai (80.20, 13.06) and (80.20, 13.07)."""
        p1 = Point(80.20, 13.06)
        p2 = Point(80.20, 13.07)  # 0.01 deg latitude difference ~ 1113.2 meters

        dist_m = GISVerifier.distance(p1, p2, metric=True)
        # 1 arcminute is ~1.11 km
        self.assertAlmostEqual(dist_m, 1113.0, delta=20.0)

    def test_dwithin_500m_and_1km(self):
        """
        Deterministic verification of PostGIS ST_DWithin:
        - A point 300m away should satisfy dwithin(p, line, 500m) and dwithin(p, line, 1000m)
        - A point 800m away should fail dwithin(p, line, 500m) and satisfy dwithin(p, line, 1000m)
        - A point 1500m away should fail both
        """
        highway = LineString([(80.0, 13.0), (80.1, 13.0)])  # Horizontal segment along lat 13.0

        # Point A: approx 300m North (delta lat = 300 / 110900 ~ 0.0027 deg)
        pt_300m = Point(80.05, 13.0 + (300.0 / 110900.0))
        # Point B: approx 800m North (delta lat = 800 / 110900 ~ 0.0072 deg)
        pt_800m = Point(80.05, 13.0 + (800.0 / 110900.0))
        # Point C: approx 1500m North (delta lat = 1500 / 110900 ~ 0.0135 deg)
        pt_1500m = Point(80.05, 13.0 + (1500.0 / 110900.0))

        # Check 500m constraint
        self.assertTrue(GISVerifier.dwithin(pt_300m, highway, distance_m=500.0))
        self.assertFalse(GISVerifier.dwithin(pt_800m, highway, distance_m=500.0))
        self.assertFalse(GISVerifier.dwithin(pt_1500m, highway, distance_m=500.0))

        # Check 1000m (1 km) constraint
        self.assertTrue(GISVerifier.dwithin(pt_300m, highway, distance_m=1000.0))
        self.assertTrue(GISVerifier.dwithin(pt_800m, highway, distance_m=1000.0))
        self.assertFalse(GISVerifier.dwithin(pt_1500m, highway, distance_m=1000.0))

    def test_gis_buffer_operation(self):
        """Test ST_Buffer generates valid metric buffer polygon."""
        pt = Point(80.20, 13.05)
        buf_poly = GISVerifier.buffer(pt, distance_m=1000.0)  # 1 km radius buffer
        self.assertIsInstance(buf_poly, Polygon)
        self.assertTrue(len(buf_poly.shell) >= 16)
        # Point must be contained inside its own buffer
        self.assertTrue(GISVerifier.contains(buf_poly, pt))
        # Area of 1 km radius circle is approx pi * r^2 = 3.14159e6 m^2
        area_m2 = GISVerifier.area(buf_poly, metric=True)
        self.assertAlmostEqual(area_m2, 3.14e6, delta=0.2e6)

    def test_highway_corridor_candidate_verification(self):
        """
        Verify 'industrial buildings within 1 km of highway' using CandidateAOI and GISVerifier.
        """
        # Candidate 1: 450m from NH48 corridor -> Should pass
        cand_compliant = CandidateAOI(
            aoi_id="AOI-CHE-IND-01",
            name="Industrial Cluster Chennai",
            bbox=(80.19, 13.055, 80.21, 13.073),
            centroid=(80.20, 13.064),  # ~440m from highway segment
            area_sqkm=2.4,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2023-11-01", "2023-11-10"),
            primary_semantic_class="industrial buildings",
            semantic_score=0.92,
            spatial_score=0.85,
        )

        # Candidate 2: 2500m from corridor -> Should fail
        cand_non_compliant = CandidateAOI(
            aoi_id="AOI-CHE-IND-02",
            name="Distant Settlement",
            bbox=(80.19, 13.08, 80.21, 13.09),
            centroid=(80.20, 13.085),  # ~2700m from highway segment
            area_sqkm=1.8,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2023-11-01", "2023-11-10"),
            primary_semantic_class="industrial buildings",
            semantic_score=0.90,
            spatial_score=0.85,
        )

        query = self.query_parser.parse("Find industrial buildings within 1 km of highway")
        
        verified_1, score_1, log_1 = self.gis_verifier.verify(cand_compliant, query)
        self.assertTrue(verified_1)
        self.assertTrue(cand_compliant.verification_flags["gis_verified"])

        verified_2, score_2, log_2 = self.gis_verifier.verify(cand_non_compliant, query)
        self.assertFalse(verified_2)
        self.assertFalse(cand_non_compliant.verification_flags["gis_verified"])
        self.assertLess(score_2, 0.85)  # Penalized


if __name__ == "__main__":
    unittest.main()
