"""
Unit Tests for Semantic Retrieval Layer (Vector Search & Candidate Generator).

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology

Tests:
  - Vision-Language compatible embedding space
  - Nearest-neighbour retrieval (text & image search)
  - Top-K limits enforcement (initial_top_k, verification_top_k, final_top_k)
  - Spatial duplicate tiles grouping (IoU / scene revisit)
  - Overlapping candidates spatial clustering
  - No results edge case (empty / non-matching)
  - Too many results edge case (pruning & clamping)
  - Pluggable ANN backend abstraction (pgvector, FAISS, Qdrant, in-memory)
"""

import unittest
import math
from typing import List

from src.part1_retrieval.vector_search import (
    VisionLanguageEmbeddingProvider,
    MockRemoteCLIPEmbeddingProvider,
    SemanticVectorSearch,
    FoundationVectorSearch,
    TileSearchResult,
    InMemoryCosineANNBackend,
    PgVectorANNBackend,
    FaissANNBackend,
    QdrantANNBackend,
    create_ann_backend,
    generate_mock_archive,
)
from src.part1_retrieval.candidate_generator import (
    CandidateGenerator,
    RetrievalConfig,
    SpatialClusterer,
    compute_bbox_iou,
    calculate_bbox_area_sqkm,
)
from src.part1_retrieval.query_parser import QueryParser
from src.part1_retrieval.metadata_filter import IndexedTile, MetadataFilter
from src.part1_retrieval.gis_geometry import BoundingBox


class TestSemanticRetrievalLayer(unittest.TestCase):

    def setUp(self):
        self.provider = MockRemoteCLIPEmbeddingProvider(embedding_dim=512)
        self.parser = QueryParser()
        self.archive = generate_mock_archive(count=80)
        self.config = RetrievalConfig(
            initial_top_k=100,
            verification_top_k=50,
            final_top_k=10,
            iou_duplicate_threshold=0.85,
            iou_overlap_threshold=0.25,
            centroid_distance_km_threshold=5.0,
            min_similarity_threshold=0.15,
        )
        self.generator = CandidateGenerator(
            archive_tiles=self.archive,
            config=self.config,
        )

    # --------------------------------------------------------------------------
    # 1. Vision-Language Embedding Space Compatibility
    # --------------------------------------------------------------------------
    def test_vision_language_embedding_space_compatibility(self):
        """
        Text and image embeddings MUST belong to the same compatible
        vision-language embedding space with unit L2 normalization.
        """
        # Text embedding
        txt_vec = self.provider.embed_text("agricultural areas and cropland")
        self.assertEqual(len(txt_vec), 512)

        # Norm should be 1.0 (unit vector)
        norm_txt = math.sqrt(sum(x * x for x in txt_vec))
        self.assertAlmostEqual(norm_txt, 1.0, places=4)

        # Compatible Image embedding (spectral + features)
        agri_img = {
            "semantic_features": ["agricultural areas", "cropland", "canals"],
            "spectral_features": {"ndvi_mean": 0.65, "ndwi_mean": -0.05},
            "sensor": "Sentinel-2",
        }
        img_vec = self.provider.embed_image(agri_img)
        self.assertEqual(len(img_vec), 512)
        norm_img = math.sqrt(sum(x * x for x in img_vec))
        self.assertAlmostEqual(norm_img, 1.0, places=4)

        # Cross-modal cosine similarity between matching text and image
        cos_sim_match = sum(t * i for t, i in zip(txt_vec, img_vec))
        self.assertGreater(cos_sim_match, 0.60, "Matching text-image pairs must exhibit high cosine similarity")

        # Cross-modal cosine similarity against completely mismatched concept (e.g. desert/arid)
        arid_img = {
            "semantic_features": ["arid terrain", "sand dunes"],
            "spectral_features": {"ndvi_mean": 0.05, "ndwi_mean": -0.45},
            "sensor": "Sentinel-2",
        }
        arid_vec = self.provider.embed_image(arid_img)
        cos_sim_mismatch = sum(t * a for t, a in zip(txt_vec, arid_vec))
        self.assertLess(cos_sim_mismatch, cos_sim_match, "Mismatched pairs must score significantly lower")

    # --------------------------------------------------------------------------
    # 2. Nearest-Neighbour Retrieval (Text & Image)
    # --------------------------------------------------------------------------
    def test_nearest_neighbour_retrieval(self):
        """
        Verifies ANN search finds semantic nearest neighbors for queries.
        """
        engine = SemanticVectorSearch(embedding_provider=self.provider)
        engine.index_tiles(self.archive)

        # Search for industrial buildings
        results = engine.search_text("Find industrial buildings and factories", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)

        # Top hit should have high similarity and contain industrial semantics
        top_hit = results[0]
        self.assertIsInstance(top_hit, TileSearchResult)
        self.assertGreater(top_hit.similarity_score, 0.40)
        self.assertTrue(
            any("industrial" in f or "factory" in f or "warehouse" in f for f in top_hit.semantic_features),
            f"Top hit features {top_hit.semantic_features} must contain industrial semantics",
        )

        # Search using image feature descriptor (future search_image pipeline)
        img_query = {
            "semantic_features": ["water bodies", "reservoir", "lake"],
            "spectral_features": {"ndvi_mean": -0.10, "ndwi_mean": 0.75},
        }
        img_results = engine.search_image(img_query, top_k=5)
        self.assertGreater(len(img_results), 0)
        top_water_hit = img_results[0]
        self.assertTrue(
            any("water" in f or "lake" in f or "reservoir" in f or "coastal" in f for f in top_water_hit.semantic_features),
            f"Top water hit features {top_water_hit.semantic_features} must contain water semantics",
        )

    # --------------------------------------------------------------------------
    # 3. Top-K Limits Configuration Enforcement
    # --------------------------------------------------------------------------
    def test_top_k_limits(self):
        """
        Verifies that initial_top_k, verification_top_k, and final_top_k are strictly
        enforced and not hardcoded.
        """
        custom_config = RetrievalConfig(
            initial_top_k=25,
            verification_top_k=15,
            final_top_k=3,
        )
        custom_gen = CandidateGenerator(archive_tiles=self.archive, config=custom_config)

        parsed = self.parser.parse("Find agricultural areas near Chennai.")
        candidates = custom_gen.generate_candidates(parsed)

        # Must respect final_top_k = 3
        self.assertLessEqual(len(candidates), 3)

        # Test overriding max_candidates dynamically to 1
        single_cand = custom_gen.generate_candidates(parsed, max_candidates=1)
        self.assertEqual(len(single_cand), 1)

    # --------------------------------------------------------------------------
    # 4. Duplicate Tiles Spatial Handling
    # --------------------------------------------------------------------------
    def test_duplicate_tiles_handling(self):
        """
        Multiple overlapping tiles representing the same physical location
        or revisit scenes must be grouped together rather than returned as duplicate AOIs.
        """
        # Create identical footprint duplicate tiles
        b = (80.15, 13.00, 80.25, 13.10)
        geom = BoundingBox(*b).to_geojson()

        t1 = IndexedTile(
            tile_id="TILE-DUP-TEST-001",
            scene_id="SCENE-BASE-001",
            geometry=geom,
            datetime="2024-01-01T05:00:00Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 1.0, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="ref1",
            semantic_features=["industrial buildings", "factories"],
            spectral_features={"ndvi_mean": 0.15, "ndwi_mean": -0.30},
        )
        t2 = IndexedTile(
            tile_id="TILE-DUP-TEST-002",
            scene_id="SCENE-REVISIT-002",
            geometry=geom,  # Identical geometry (IoU = 1.0)
            datetime="2024-02-01T05:00:00Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 2.0, "valid_pixel_pct": 99.0, "usable_data_flag": True},
            embedding_reference="ref2",
            semantic_features=["industrial buildings", "factories"],
            spectral_features={"ndvi_mean": 0.15, "ndwi_mean": -0.30},
        )

        gen = CandidateGenerator(archive_tiles=[t1, t2], config=self.config)
        parsed = self.parser.parse("Find industrial buildings")
        candidates = gen.generate_candidates(parsed)

        # Both tiles should be merged into 1 Candidate Geographic Region
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertIn("constituent_tiles", cand.metadata)
        self.assertEqual(len(cand.metadata["constituent_tiles"]), 2)
        self.assertGreaterEqual(cand.metadata["duplicate_count"], 1)

    # --------------------------------------------------------------------------
    # 5. Overlapping Candidates Grouping
    # --------------------------------------------------------------------------
    def test_overlapping_candidates_grouping(self):
        """
        Tiles with significant spatial overlap (IoU >= 0.25) must be spatially clustered
        into a unified candidate geographic region with combined bounding box.
        """
        # Box 1: (80.10, 13.00, 80.20, 13.10)
        # Box 2: (80.15, 13.00, 80.25, 13.10) -> 50% overlap along longitude
        b1 = (80.10, 13.00, 80.20, 13.10)
        b2 = (80.15, 13.00, 80.25, 13.10)

        iou = compute_bbox_iou(b1, b2)
        self.assertGreater(iou, 0.30)

        t1 = IndexedTile(
            tile_id="TILE-OVERLAP-A",
            scene_id="SCENE-A",
            geometry=BoundingBox(*b1).to_geojson(),
            datetime="2024-03-01T05:00:00Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 2.0, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="refA",
            semantic_features=["agricultural areas", "farmland"],
            spectral_features={"ndvi_mean": 0.60, "ndwi_mean": -0.10},
        )
        t2 = IndexedTile(
            tile_id="TILE-OVERLAP-B",
            scene_id="SCENE-B",
            geometry=BoundingBox(*b2).to_geojson(),
            datetime="2024-03-01T05:00:00Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 2.0, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="refB",
            semantic_features=["agricultural areas", "farmland"],
            spectral_features={"ndvi_mean": 0.60, "ndwi_mean": -0.10},
        )

        gen = CandidateGenerator(archive_tiles=[t1, t2], config=self.config)
        parsed = self.parser.parse("Find agricultural areas")
        candidates = gen.generate_candidates(parsed)

        self.assertEqual(len(candidates), 1)
        clustered_aoi = candidates[0]

        # Union bbox must span from min_lon of b1 (80.10) to max_lon of b2 (80.25)
        self.assertAlmostEqual(clustered_aoi.bbox[0], 80.10, places=2)
        self.assertAlmostEqual(clustered_aoi.bbox[2], 80.25, places=2)

        # Notice that similarity is NOT truth: it is a semantic retrieval signal
        self.assertTrue(clustered_aoi.metadata.get("similarity_is_hypothesis", False))
        self.assertIn("constituent_tiles", clustered_aoi.metadata)

    # --------------------------------------------------------------------------
    # 6. No Results Edge Case
    # --------------------------------------------------------------------------
    def test_no_results_edge_case(self):
        """
        Empty archives or queries that match zero tiles must return an empty list
        cleanly without raising unhandled exceptions.
        """
        empty_gen = CandidateGenerator(archive_tiles=[], config=self.config)
        parsed = self.parser.parse("Find agricultural areas near Chennai.")
        candidates = empty_gen.generate_candidates(parsed)
        self.assertEqual(candidates, [])

        # High similarity threshold that filters all hits
        strict_config = RetrievalConfig(min_similarity_threshold=0.999)
        strict_gen = CandidateGenerator(archive_tiles=self.archive, config=strict_config)
        strict_candidates = strict_gen.generate_candidates(parsed)
        self.assertEqual(strict_candidates, [])

    # --------------------------------------------------------------------------
    # 7. Too Many Results Edge Case
    # --------------------------------------------------------------------------
    def test_too_many_results_edge_case(self):
        """
        When hundreds of tiles match a query, the candidate generator must strictly
        prune and clamp results at initial_top_k, verification_top_k, and final_top_k.
        """
        large_archive = generate_mock_archive(count=100)
        self.assertEqual(len(large_archive), 100)

        config = RetrievalConfig(
            initial_top_k=50,
            verification_top_k=20,
            final_top_k=5,
        )
        gen = CandidateGenerator(archive_tiles=large_archive, config=config)
        parsed = self.parser.parse("Show satellite imagery")
        candidates = gen.generate_candidates(parsed)

        self.assertLessEqual(len(candidates), 5)
        self.assertGreater(len(candidates), 0)

    # --------------------------------------------------------------------------
    # 8. ANN Backend Abstraction (pgvector, FAISS, Qdrant, in-memory)
    # --------------------------------------------------------------------------
    def test_ann_backends_abstraction(self):
        """
        Verifies that ANN backend abstractions (pgvector, FAISS, Qdrant, in-memory)
        adhere to the ANNBackend contract.
        """
        backends = [
            create_ann_backend("in_memory"),
            create_ann_backend("pgvector"),
            create_ann_backend("faiss", dim=512),
            create_ann_backend("qdrant"),
        ]

        dummy_vec = [0.1] * 512
        items = [
            ("tile_1", dummy_vec, {"sensor": "Sentinel-2", "quality": {"cloud_cover_percent": 5.0}}),
            ("tile_2", [-x for x in dummy_vec], {"sensor": "Cartosat-3", "quality": {"cloud_cover_percent": 1.0}}),
        ]

        for backend in backends:
            backend.clear()
            backend.index(items)
            results = backend.search(dummy_vec, top_k=1)
            self.assertEqual(len(results), 1)
            top_id, score, meta = results[0]
            self.assertEqual(top_id, "tile_1")
            self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
