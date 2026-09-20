#!/usr/bin/env python3
"""
CLI & API Helper for Semantic Vector Search, Vision-Language Embeddings, and Spatial Clustering.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part1_retrieval.vector_search import (
    SemanticVectorSearch,
    MockRemoteCLIPEmbeddingProvider,
    generate_mock_archive,
    TileSearchResult,
)
from src.part1_retrieval.candidate_generator import (
    CandidateGenerator,
    RetrievalConfig,
    SpatialClusterer,
    compute_bbox_iou,
)
from src.part1_retrieval.query_parser import QueryParser
from src.part1_retrieval.gazetteer import Gazetteer


def main():
    parser = argparse.ArgumentParser(description="Semantic Vector Search and Spatial Clustering CLI")
    parser.add_argument(
        "--action",
        choices=["search", "candidates", "archive_stats", "embedding_compare"],
        default="candidates",
    )
    parser.add_argument("--query", type=str, default="Find agricultural areas near Chennai.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--config_json", type=str, default="")

    args = parser.parse_args()

    # Load configuration
    config = RetrievalConfig.load_from_yaml()
    if args.config_json:
        try:
            cfg_dict = json.loads(args.config_json)
            for k, v in cfg_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        except Exception:
            pass

    archive = generate_mock_archive(count=80)
    provider = MockRemoteCLIPEmbeddingProvider(embedding_dim=512)
    search_engine = SemanticVectorSearch(embedding_provider=provider)
    search_engine.index_tiles(archive)

    if args.action == "search":
        hits = search_engine.search_text(args.query, top_k=args.top_k)
        results = [h.to_dict() for h in hits]
        print(json.dumps({
            "query": args.query,
            "top_k": args.top_k,
            "total_hits": len(results),
            "hits": results,
            "embedding_model": provider.model_name,
            "embedding_dim": provider.embedding_dim,
        }, indent=2))

    elif args.action == "candidates":
        parser_obj = QueryParser()
        parsed = parser_obj.parse(args.query)
        generator = CandidateGenerator(
            archive_tiles=archive,
            config=config,
        )
        candidates = generator.generate_candidates(parsed, max_candidates=args.top_k)

        out = {
            "query": args.query,
            "normalized_semantic_query": parsed.normalized_semantic_query,
            "semantic_targets": parsed.semantic_targets,
            "location_names": parsed.location_names,
            "config": {
                "initial_top_k": config.initial_top_k,
                "verification_top_k": config.verification_top_k,
                "final_top_k": config.final_top_k,
                "iou_duplicate_threshold": config.iou_duplicate_threshold,
                "iou_overlap_threshold": config.iou_overlap_threshold,
                "centroid_distance_km_threshold": config.centroid_distance_km_threshold,
            },
            "candidate_count": len(candidates),
            "candidates": [c.to_dict() for c in candidates],
            "archive_tile_count": len(archive),
        }
        print(json.dumps(out, indent=2))

    elif args.action == "archive_stats":
        # Group tiles by region / sensor
        by_sensor = {}
        by_theme = {}
        for t in archive:
            by_sensor[t.sensor] = by_sensor.get(t.sensor, 0) + 1
            for f in t.semantic_features:
                by_theme[f] = by_theme.get(f, 0) + 1

        print(json.dumps({
            "total_tiles": len(archive),
            "sensor_distribution": by_sensor,
            "top_semantic_themes": sorted(by_theme.items(), key=lambda x: x[1], reverse=True)[:10],
            "embedding_provider": provider.model_name,
            "embedding_dim": provider.embedding_dim,
        }, indent=2))

    elif args.action == "embedding_compare":
        # Compare text query with various concepts
        text_vec = provider.embed_text(args.query)
        concepts = [
            ("agricultural areas", ["cropland", "farmland"], {"ndvi_mean": 0.65, "ndwi_mean": -0.05}),
            ("industrial buildings", ["factories", "warehouses"], {"ndvi_mean": 0.15, "ndwi_mean": -0.30}),
            ("highways", ["expressways", "bridges"], {"ndvi_mean": 0.20, "ndwi_mean": -0.25}),
            ("water bodies", ["lakes", "reservoirs"], {"ndvi_mean": -0.10, "ndwi_mean": 0.75}),
            ("harbour port", ["docks", "shipping"], {"ndvi_mean": -0.12, "ndwi_mean": 0.85}),
            ("arid desert", ["sand dunes", "earthworks"], {"ndvi_mean": 0.05, "ndwi_mean": -0.45}),
        ]
        scores = []
        for name, tags, spec in concepts:
            img_vec = provider.embed_image({
                "semantic_features": [name] + tags,
                "spectral_features": spec,
            })
            sim = sum(t * i for t, i in zip(text_vec, img_vec))
            scores.append({
                "concept": name,
                "tags": tags,
                "cosine_similarity": round(sim, 4),
            })
        scores.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        print(json.dumps({
            "query": args.query,
            "embedding_dim": provider.embedding_dim,
            "similarities": scores,
        }, indent=2))


if __name__ == "__main__":
    main()
