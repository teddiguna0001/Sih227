#!/usr/bin/env python3
"""
CLI Execution Script for Part 1: Semantic Retrieval Pipeline.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part1_retrieval.retrieval_pipeline import RetrievalPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run SIH26227 Part 1 Semantic Retrieval & Change Analysis Pipeline"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Find agricultural areas near Chennai.",
        help="Natural language satellite imagery retrieval query",
    )
    parser.add_argument(
        "--candidates",
        "-c",
        type=int,
        default=5,
        help="Maximum candidate AOIs to retrieve and score",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )

    args = parser.parse_args()

    pipeline = RetrievalPipeline()
    result = pipeline.run(args.query, max_candidates=args.candidates)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("SIH26227 / SH227 - PART 1 SEMANTIC RETRIEVAL PIPELINE EXECUTION")
    print("Organization: Ministry of Defence | Theme: Space Technology")
    print("=" * 80)
    print(f"QUERY: \"{result.parsed_query.raw_query}\"")
    print("-" * 80)

    # 1. Parsed Query Summary
    pq = result.parsed_query
    print("1. PARSED QUERY SCHEMA:")
    print(f"   Normalized Semantic Query : {pq.normalized_semantic_query}")
    print(f"   Semantic Targets          : {pq.semantic_targets}")
    print(f"   Location Names            : {pq.location_names}")
    print(f"   Spatial Relations         : {pq.spatial_relations}")
    print(f"   Temporal Span             : {pq.start_date} to {pq.end_date}")
    print(f"   Sensors                   : {pq.sensor_constraints or 'None (Default Optical)'}")
    print(f"   Resolution Constraints    : {pq.resolution_constraints or 'Standard GSD (10m)'}")
    print(f"   Change Intent             : {pq.change_intent}")
    print(f"   Requested Change Type     : {pq.requested_change_type}")
    print(f"   Query Confidence          : {pq.confidence}")
    if pq.warnings:
        print("   Warnings / Confirmations  :")
        for w in pq.warnings:
            print(f"     * {w}")

    # 2. Candidate AOIs
    print("\n2. RETRIEVED & RERANKED CANDIDATE AOIs:")
    for idx, aoi in enumerate(result.candidate_aois, 1):
        v = aoi.verification_flags
        v_summary = f"GIS:{'Y' if v['gis_verified'] else 'N'} | Ctx:{'Y' if v['context_verified'] else 'N'} | Mask:{'Y' if v['segmentation_verified'] else 'N'}"
        print(f"   [{idx}] ID: {aoi.aoi_id} | Name: {aoi.name}")
        print(f"       Centroid: {aoi.centroid} | BBox: {aoi.bbox} | Area: {aoi.area_sqkm} sq km")
        print(f"       Sensor: {aoi.sensor} ({aoi.resolution_m}m) | Time: {aoi.temporal_range[0]} - {aoi.temporal_range[1]}")
        print(f"       Scores: Overall={aoi.overall_score:.3f} (Semantic={aoi.semantic_score:.2f}, Spatial={aoi.spatial_score:.2f}, Temporal={aoi.temporal_score:.2f})")
        print(f"       Verifications: {v_summary}")

    # 3. Confidence & Quality Assurance
    ca = result.confidence_assessment
    print("\n3. PIPELINE CONFIDENCE & QUALITY ASSURANCE:")
    print(f"   Overall Retrieval Confidence : {ca['overall_confidence']}")
    print(f"   Query Parsing Confidence     : {ca['query_parsing_confidence']}")
    print(f"   Candidate Quality Confidence : {ca['candidate_quality_confidence']}")
    print(f"   Verification Pass Rate       : {ca['verification_pass_rate'] * 100:.1f}%")
    print(f"   Requires User Confirmation   : {ca['requires_user_confirmation']}")
    if ca.get("confirmation_reasons"):
        for r in ca["confirmation_reasons"]:
            print(f"     * {r}")

    # 4. Pipeline Execution Trace
    print("\n4. EXECUTION TRACE:")
    for t in result.execution_trace:
        print(f"   -> {t}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
