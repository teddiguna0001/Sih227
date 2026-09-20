"""
Unit and Hard-Negative Test Suite for Candidate Verification & Multi-Factor Reranking.

Validates:
1. ContextVerifier: Multi-scale spatial context consistency (small -> medium -> large).
2. SegmentationVerifier: Structural evidence (building fraction, roof size, NDWI, NDVI).
3. GISVerifier: Deterministic metric spatial predicates (ST_DWithin, ST_Intersects).
4. CandidateReranker: Explainable multi-factor scoring (relevance_score).
5. Hard-Negative Discrimination:
   - Industrial building vs. Agricultural greenhouse
   - Highway vs. Airport runway
   - Water body vs. Topographic/Cloud shadow
   - Agriculture vs. Bare land / dry construction plot
   - Residential roofs vs. Industrial roofs

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import unittest
from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.context_verifier import ContextVerifier
from src.part1_retrieval.segmentation_verifier import SegmentationVerifier
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.reranker import CandidateReranker
from src.part1_retrieval.confidence import ConfidenceEngine
from src.part1_retrieval.retrieval_pipeline import RetrievalPipeline


class TestVerificationAndReranking(unittest.TestCase):

    def setUp(self):
        self.context_verifier = ContextVerifier()
        self.segmentation_verifier = SegmentationVerifier()
        self.gis_verifier = GISVerifier()
        self.reranker = CandidateReranker()
        self.confidence_engine = ConfidenceEngine()

    # =========================================================================
    # HARD NEGATIVE TEST 1: Industrial Building vs. Greenhouse
    # =========================================================================
    def test_hard_negative_industrial_vs_greenhouse(self):
        """
        Query: 'large industrial buildings near highway'
        True Positive: Industrial warehouse with large roofs in logistics district along highway corridor.
        Hard Negative: Large glass/poly roofs with high surrounding vegetation in agricultural farmland.
        """
        query = ParsedQuery(
            raw_query="Find large industrial buildings near highway",
            normalized_semantic_query="industrial warehouse logistics park near highway",
            semantic_targets=["large industrial buildings", "warehouses"],
            spatial_relations=[{"relation": "near", "reference": "highway", "distance_km": 2.0}],
            confidence=0.90,
        )

        # 1. True Industrial Candidate (Near NH48 Highway corridor)
        true_industrial = CandidateAOI(
            aoi_id="AOI-IND-01",
            name="Sriperumbudur Industrial Hub",
            bbox=(80.12, 13.00, 80.14, 13.02),
            centroid=(80.13, 13.01),
            area_sqkm=2.4,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial_facility",
            semantic_score=0.84,
            spatial_score=0.85,
            metadata={
                "multiscale_context": {
                    "small": ["large_roofs", "paved_aprons", "loading_docks"],
                    "medium": ["warehouses", "logistics_yards", "industrial_estate"],
                    "large": ["transport_corridor", "highway_network", "chennai_nh48_industrial_corridor"],
                },
                "structural_evidence": {
                    "building_fraction": 0.35,
                    "road_fraction": 0.18,
                    "vegetation_fraction": 0.08,
                    "mean_roof_area_sqm": 4200.0,
                    "max_roof_area_sqm": 9500.0,
                    "building_density_per_sqkm": 14.0,
                    "semantic_class_probabilities": {"industrial": 0.88, "commercial": 0.10},
                },
            },
        )

        # 2. Hard Negative Greenhouse Candidate (Rural farmland, far from highway)
        greenhouse = CandidateAOI(
            aoi_id="AOI-GRN-01",
            name="Rural Polyhouse Farm",
            bbox=(79.80, 12.80, 79.82, 12.82),
            centroid=(79.81, 12.81),
            area_sqkm=1.8,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial_facility",  # Misclassified initially by raw text/RGB
            semantic_score=0.81,  # Similar high raw visual embedding score due to big reflective roofs
            spatial_score=0.40,
            metadata={
                "multiscale_context": {
                    "small": ["large_roofs", "translucent_panels"],
                    "medium": ["farmland", "cropland", "greenhouse", "rural_hamlet"],
                    "large": ["rural_basin", "agricultural_belt"],
                },
                "structural_evidence": {
                    "building_fraction": 0.22,
                    "road_fraction": 0.02,
                    "vegetation_fraction": 0.65,
                    "mean_roof_area_sqm": 2100.0,
                    "max_roof_area_sqm": 3500.0,
                    "building_density_per_sqkm": 8.0,
                    "semantic_class_probabilities": {"greenhouse": 0.75, "industrial": 0.15},
                },
            },
        )

        candidates = [greenhouse, true_industrial]

        # Execute Pipeline Steps
        self.context_verifier.verify_candidates(candidates, query)
        self.segmentation_verifier.verify_candidates(candidates, query)
        self.gis_verifier.verify_candidates(candidates, query)
        reranked = self.reranker.rerank(candidates, query)

        # Verifications
        # 1. True industrial passes all verifications
        self.assertTrue(true_industrial.verification_flags["context_verified"])
        self.assertTrue(true_industrial.verification_flags["segmentation_verified"])
        self.assertTrue(true_industrial.verification_flags["gis_verified"])

        # 2. Greenhouse triggered context conflict and structural mismatch
        self.assertFalse(greenhouse.verification_flags["context_verified"])
        self.assertTrue(greenhouse.metadata["context_verification"]["conflict_detected"])
        self.assertIn("greenhouse", greenhouse.metadata["context_verification"]["conflict_reason"].lower())

        self.assertFalse(greenhouse.verification_flags["segmentation_verified"])
        self.assertTrue(greenhouse.metadata["structural_verification"]["mismatch_detected"])

        # 3. Reranking orders True Industrial strictly above Greenhouse with substantial margin
        self.assertEqual(reranked[0].aoi_id, "AOI-IND-01")
        self.assertGreater(true_industrial.overall_score, greenhouse.overall_score + 0.30)
        self.assertGreaterEqual(true_industrial.overall_score, 0.70)
        self.assertLessEqual(greenhouse.overall_score, 0.40)

    # =========================================================================
    # HARD NEGATIVE TEST 2: Highway vs. Runway
    # =========================================================================
    def test_hard_negative_highway_vs_runway(self):
        """
        Query: 'highways near Chennai'
        True Positive: Linear civil roadway connecting urban centers with road interchanges.
        Hard Negative: Airport runway (long paved straight strip, similar visual line).
        """
        query = ParsedQuery(
            raw_query="Find highways near Chennai",
            normalized_semantic_query="highways expressway transit corridor Chennai",
            semantic_targets=["highway", "road network"],
            location_names=["Chennai"],
            confidence=0.92,
        )

        # 1. True Highway Candidate
        highway = CandidateAOI(
            aoi_id="AOI-HWY-01",
            name="GST Road Arterial Highway",
            bbox=(80.11, 13.00, 80.13, 13.02),
            centroid=(80.12, 13.01),
            area_sqkm=1.5,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="highway",
            semantic_score=0.86,
            spatial_score=0.88,
            metadata={
                "multiscale_context": {
                    "small": ["paved_road_strip", "road_markings", "median_strip"],
                    "medium": ["highway_corridor", "interchanges", "access_ramps"],
                    "large": ["interstate_artery", "regional_connectivity_grid"],
                },
                "structural_evidence": {
                    "road_fraction": 0.42,
                    "building_fraction": 0.05,
                    "semantic_class_probabilities": {"road_network": 0.90, "urban": 0.10},
                },
            },
        )

        # 2. Hard Negative Runway Candidate
        runway = CandidateAOI(
            aoi_id="AOI-RNW-01",
            name="Meenambakkam Airport Runway Strip",
            bbox=(80.16, 12.98, 80.18, 13.00),
            centroid=(80.17, 12.99),
            area_sqkm=1.8,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="highway",  # Misclassified initially as paved highway
            semantic_score=0.83,
            spatial_score=0.80,
            metadata={
                "multiscale_context": {
                    "small": ["paved_road_strip", "white_threshold_markings"],
                    "medium": ["runway", "taxiway", "airport", "airfield", "hangar"],
                    "large": ["aviation_complex", "urban_perimeter"],
                },
                "structural_evidence": {
                    "road_fraction": 0.38,
                    "building_fraction": 0.08,
                    "semantic_class_probabilities": {"runway": 0.85, "airport": 0.92},
                },
            },
        )

        candidates = [runway, highway]
        self.context_verifier.verify_candidates(candidates, query)
        self.segmentation_verifier.verify_candidates(candidates, query)
        self.gis_verifier.verify_candidates(candidates, query)
        reranked = self.reranker.rerank(candidates, query)

        self.assertTrue(highway.verification_flags["context_verified"])
        self.assertFalse(runway.verification_flags["context_verified"])
        self.assertTrue(runway.metadata["context_verification"]["conflict_detected"])
        self.assertIn("airport", runway.metadata["context_verification"]["conflict_reason"].lower())

        self.assertEqual(reranked[0].aoi_id, "AOI-HWY-01")
        self.assertGreater(highway.overall_score, runway.overall_score + 0.25)

    # =========================================================================
    # HARD NEGATIVE TEST 3: Water Body vs. Shadow
    # =========================================================================
    def test_hard_negative_water_vs_shadow(self):
        """
        Query: 'Find water bodies and reservoirs'
        True Positive: Chembarambakkam Lake with high NDWI (> 0.5) and water fraction.
        Hard Negative: Deep mountain/cloud shadow with low visual brightness, but negative NDWI (< -0.2).
        """
        query = ParsedQuery(
            raw_query="Find water bodies and reservoirs",
            normalized_semantic_query="water body reservoir lake freshwater surface",
            semantic_targets=["water bodies", "reservoirs"],
            confidence=0.95,
        )

        # 1. True Water Body
        water = CandidateAOI(
            aoi_id="AOI-WTR-01",
            name="Chembarambakkam Reservoir",
            bbox=(80.00, 13.00, 80.04, 13.04),
            centroid=(80.02, 13.02),
            area_sqkm=15.2,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="water_body",
            semantic_score=0.88,
            spatial_score=0.80,
            metadata={
                "multiscale_context": {
                    "small": ["specular_water_surface", "shoreline_boundary"],
                    "medium": ["water_body", "wetland_buffer", "drainage_catchment"],
                    "large": ["river_basin", "coastal_drainage"],
                },
                "structural_evidence": {
                    "water_fraction": 0.85,
                    "vegetation_fraction": 0.08,
                    "spectral_indices": {"ndwi_mean": 0.58, "ndvi_mean": -0.18},
                    "semantic_class_probabilities": {"water_body": 0.94},
                },
            },
        )

        # 2. Hard Negative Shadow
        shadow = CandidateAOI(
            aoi_id="AOI-SHD-01",
            name="Nagari Hills Escarpment Shadow",
            bbox=(79.55, 13.30, 79.57, 13.32),
            centroid=(79.56, 13.31),
            area_sqkm=2.1,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="water_body",  # False positive from dark reflectance
            semantic_score=0.82,
            spatial_score=0.60,
            metadata={
                "multiscale_context": {
                    "small": ["dark_low_reflectance_patch"],
                    "medium": ["cloud_shadow", "steep_cliff_shadow", "mountain_ridge"],
                    "large": ["rocky_highlands"],
                },
                "structural_evidence": {
                    "water_fraction": 0.02,
                    "vegetation_fraction": 0.15,
                    "bare_soil_fraction": 0.80,
                    "spectral_indices": {"ndwi_mean": -0.32, "ndvi_mean": 0.05},
                    "semantic_class_probabilities": {"shadow": 0.88, "water_body": 0.05},
                },
            },
        )

        candidates = [shadow, water]
        self.context_verifier.verify_candidates(candidates, query)
        self.segmentation_verifier.verify_candidates(candidates, query)
        self.gis_verifier.verify_candidates(candidates, query)
        reranked = self.reranker.rerank(candidates, query)

        self.assertTrue(water.verification_flags["segmentation_verified"])
        self.assertFalse(shadow.verification_flags["segmentation_verified"])
        self.assertTrue(shadow.metadata["structural_verification"]["mismatch_detected"])
        self.assertIn("shadow", shadow.metadata["structural_verification"]["mismatch_reason"].lower())

        self.assertEqual(reranked[0].aoi_id, "AOI-WTR-01")
        self.assertGreater(water.overall_score, shadow.overall_score + 0.35)

    # =========================================================================
    # HARD NEGATIVE TEST 4: Agriculture vs. Bare Land
    # =========================================================================
    def test_hard_negative_agriculture_vs_bare_land(self):
        """
        Query: 'Find agricultural cropland'
        True Positive: Vigorous green crop fields with high NDVI (> 0.6) and parcel boundaries.
        Hard Negative: Dry barren excavation/saline flat with low NDVI (< 0.15) and high bare soil.
        """
        query = ParsedQuery(
            raw_query="Find agricultural cropland",
            normalized_semantic_query="agricultural cropland farming cultivation fertile fields",
            semantic_targets=["agricultural cropland", "farming"],
            confidence=0.91,
        )

        # 1. True Agriculture
        agriculture = CandidateAOI(
            aoi_id="AOI-AGR-01",
            name="Kanchipuram Paddy Fields",
            bbox=(79.68, 12.82, 79.72, 12.86),
            centroid=(79.70, 12.84),
            area_sqkm=8.4,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="cropland",
            semantic_score=0.87,
            spatial_score=0.80,
            metadata={
                "multiscale_context": {
                    "small": ["crop_field_plots", "vegetation_canopy"],
                    "medium": ["agricultural_farmland", "irrigation_canals"],
                    "large": ["fertile_plains", "river_valley"],
                },
                "structural_evidence": {
                    "vegetation_fraction": 0.72,
                    "bare_soil_fraction": 0.18,
                    "spectral_indices": {"ndvi_mean": 0.64, "ndwi_mean": -0.02},
                    "semantic_class_probabilities": {"cropland": 0.89},
                },
            },
        )

        # 2. Hard Negative Bare Land
        bare_land = CandidateAOI(
            aoi_id="AOI-BAR-01",
            name="Dry Construction Clearing",
            bbox=(79.92, 13.10, 79.94, 13.12),
            centroid=(79.93, 13.11),
            area_sqkm=4.2,
            sensor="Sentinel-2",
            resolution_m=10.0,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="cropland",  # Misclassified flat open parcel
            semantic_score=0.80,
            spatial_score=0.75,
            metadata={
                "multiscale_context": {
                    "small": ["cleared_earth", "open_parcel"],
                    "medium": ["barren_rock", "construction_excavation", "saline_flat"],
                    "large": ["arid_plateau"],
                },
                "structural_evidence": {
                    "vegetation_fraction": 0.05,
                    "bare_soil_fraction": 0.86,
                    "spectral_indices": {"ndvi_mean": 0.11, "ndwi_mean": -0.22},
                    "semantic_class_probabilities": {"barren": 0.82, "cropland": 0.08},
                },
            },
        )

        candidates = [bare_land, agriculture]
        self.context_verifier.verify_candidates(candidates, query)
        self.segmentation_verifier.verify_candidates(candidates, query)
        self.gis_verifier.verify_candidates(candidates, query)
        reranked = self.reranker.rerank(candidates, query)

        self.assertTrue(agriculture.verification_flags["segmentation_verified"])
        self.assertFalse(bare_land.verification_flags["segmentation_verified"])
        self.assertTrue(bare_land.metadata["structural_verification"]["mismatch_detected"])
        self.assertIn("bare land", bare_land.metadata["structural_verification"]["mismatch_reason"].lower())

        self.assertEqual(reranked[0].aoi_id, "AOI-AGR-01")
        self.assertGreater(agriculture.overall_score, bare_land.overall_score + 0.30)

    # =========================================================================
    # HARD NEGATIVE TEST 5: Residential Roofs vs. Industrial Roofs
    # =========================================================================
    def test_hard_negative_residential_vs_industrial_roofs(self):
        """
        Query: 'Find large industrial factories and warehouses'
        True Positive: Vast contiguous rooftops (mean > 3000 sqm) with low building count density.
        Hard Negative: Dense urban residential colony with high roof count but small roof size (< 200 sqm).
        """
        query = ParsedQuery(
            raw_query="Find large industrial factories and warehouses",
            normalized_semantic_query="large industrial factory manufacturing plant warehouse facilities",
            semantic_targets=["large industrial factories", "warehouses"],
            confidence=0.92,
        )

        # 1. True Industrial Factory
        industrial = CandidateAOI(
            aoi_id="AOI-FAC-01",
            name="Oragadam Automotive Manufacturing Zone",
            bbox=(79.95, 12.83, 79.98, 12.86),
            centroid=(79.965, 12.845),
            area_sqkm=4.8,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial_facility",
            semantic_score=0.89,
            spatial_score=0.82,
            metadata={
                "multiscale_context": {
                    "small": ["large_roofs", "assembly_plants", "chimneys"],
                    "medium": ["warehouses", "industrial_estate", "logistics_yards"],
                    "large": ["transport_corridor", "freight_artery"],
                },
                "structural_evidence": {
                    "building_fraction": 0.32,
                    "road_fraction": 0.16,
                    "mean_roof_area_sqm": 5400.0,
                    "max_roof_area_sqm": 16000.0,
                    "building_density_per_sqkm": 12.0,
                    "semantic_class_probabilities": {"industrial": 0.92},
                },
            },
        )

        # 2. Hard Negative Residential Suburb
        residential = CandidateAOI(
            aoi_id="AOI-RES-01",
            name="Tambaram Dense Residential Colony",
            bbox=(80.11, 12.92, 80.13, 12.94),
            centroid=(80.12, 12.93),
            area_sqkm=2.2,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial_facility",  # Confused by high rooftop density
            semantic_score=0.83,
            spatial_score=0.80,
            metadata={
                "multiscale_context": {
                    "small": ["small_pitched_roofs", "water_tanks", "rooftop_terraces"],
                    "medium": ["dense_residential_colony", "residential_suburb", "apartments"],
                    "large": ["metropolitan_urban_zone"],
                },
                "structural_evidence": {
                    "building_fraction": 0.48,  # High building presence, but small individual homes
                    "road_fraction": 0.22,
                    "mean_roof_area_sqm": 135.0,  # Far below industrial (> 800 sqm)
                    "max_roof_area_sqm": 240.0,
                    "building_density_per_sqkm": 210.0,
                    "semantic_class_probabilities": {"residential": 0.86, "industrial": 0.08},
                },
            },
        )

        candidates = [residential, industrial]
        self.context_verifier.verify_candidates(candidates, query)
        self.segmentation_verifier.verify_candidates(candidates, query)
        self.gis_verifier.verify_candidates(candidates, query)
        reranked = self.reranker.rerank(candidates, query)

        self.assertTrue(industrial.verification_flags["segmentation_verified"])
        self.assertFalse(residential.verification_flags["segmentation_verified"])
        self.assertTrue(residential.metadata["structural_verification"]["mismatch_detected"])
        self.assertIn("residential", residential.metadata["structural_verification"]["mismatch_reason"].lower())

        self.assertEqual(reranked[0].aoi_id, "AOI-FAC-01")
        self.assertGreater(industrial.overall_score, residential.overall_score + 0.30)

    # =========================================================================
    # DETERMINISTIC GIS VERIFICATION TESTS
    # =========================================================================
    def test_deterministic_gis_corridor_enforcement(self):
        """
        Query: 'industrial buildings within 1 km of highway'
        Verifies that candidate distance to highway geometry is evaluated
        deterministically using projected metric Cartesian calculations,
        not estimated heuristically by a vision model.
        """
        query = ParsedQuery(
            raw_query="Find industrial buildings within 1 km of highway",
            normalized_semantic_query="industrial buildings near highway",
            semantic_targets=["industrial buildings"],
            spatial_relations=[{"relation": "within", "reference": "highway", "distance_km": 1.0}],
            confidence=0.90,
        )

        # Candidate A: ~450 meters from NH48 corridor (Passes ST_DWithin 1000m)
        cand_a = CandidateAOI(
            aoi_id="AOI-GIS-PASS",
            name="Near Corridor AOI",
            bbox=(80.12, 13.01, 80.13, 13.02),
            centroid=(80.125, 13.015),
            area_sqkm=1.0,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial",
            semantic_score=0.85,
            spatial_score=0.80,
        )

        # Candidate B: ~6.2 km from NH48 corridor (Fails ST_DWithin 1000m)
        cand_b = CandidateAOI(
            aoi_id="AOI-GIS-FAIL",
            name="Far Corridor AOI",
            bbox=(80.00, 12.85, 80.02, 12.87),
            centroid=(80.01, 12.86),
            area_sqkm=1.0,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial",
            semantic_score=0.85,
            spatial_score=0.80,
        )

        self.gis_verifier.verify(cand_a, query)
        self.gis_verifier.verify(cand_b, query)

        self.assertTrue(cand_a.verification_flags["gis_verified"])
        self.assertFalse(cand_b.verification_flags["gis_verified"])

        pred_a = cand_a.metadata["gis_verification"]["predicates"][0]
        self.assertTrue(pred_a["is_compliant"])
        self.assertLessEqual(pred_a["measured_dist_m"], 1000.0)

        pred_b = cand_b.metadata["gis_verification"]["predicates"][0]
        self.assertFalse(pred_b["is_compliant"])
        self.assertGreater(pred_b["measured_dist_m"], 1000.0)

    # =========================================================================
    # EXPLAINABLE RERANKING & CONFIDENCE
    # =========================================================================
    def test_explainable_reranker_and_confidence(self):
        """
        Validates that reranker outputs explainable factors and strictly names
        the score 'relevance_score' / 'ranking_score' / 'retrieval_score' (NOT probability).
        """
        query = ParsedQuery(
            raw_query="Find industrial buildings near Chennai",
            normalized_semantic_query="industrial buildings Chennai",
            semantic_targets=["industrial buildings"],
            location_names=["Chennai"],
            confidence=0.88,
        )

        cand = CandidateAOI(
            aoi_id="AOI-EXP-01",
            name="Test AOI",
            bbox=(80.12, 13.00, 80.14, 13.02),
            centroid=(80.13, 13.01),
            area_sqkm=2.0,
            sensor="Cartosat-3",
            resolution_m=0.28,
            temporal_range=("2024-01-01", "2024-03-30"),
            primary_semantic_class="industrial",
            semantic_score=0.85,
            spatial_score=0.80,
            verification_flags={
                "gis_verified": True,
                "context_verified": True,
                "segmentation_verified": True,
                "resolution_compliant": True,
                "cloud_cover_compliant": True,
            },
        )

        reranked = self.reranker.rerank([cand], query)
        c = reranked[0]

        # Verify naming and bounds
        self.assertIn("relevance_score", c.metadata)
        self.assertIn("ranking_score", c.metadata)
        self.assertIn("retrieval_score", c.metadata)
        self.assertGreaterEqual(c.overall_score, 0.0)
        self.assertLessEqual(c.overall_score, 1.0)

        # Verify explanation structure
        exp = c.metadata["ranking_explanation"]
        self.assertIn("relevance_score", exp)
        self.assertIn("factor_breakdown", exp)
        self.assertIn("summary", exp)
        self.assertIn("NOT a calibrated probability", exp["notice"])

        # Confidence engine evaluation
        conf_report = self.confidence_engine.evaluate(query, reranked)
        self.assertIn("overall_confidence", conf_report)
        self.assertIn("verification_pass_rate", conf_report)
        self.assertFalse(conf_report["requires_user_confirmation"])


if __name__ == "__main__":
    unittest.main()
