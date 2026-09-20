#!/usr/bin/env python3
"""
Test Query Script for Part 1: Semantic Retrieval.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import sys
import os
import json
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part1_retrieval.query_parser import QueryParser
from src.part1_retrieval.query_schema import ChangeType


REQUIRED_BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "Find agricultural areas near Chennai.",
        "expected_target": "agricultural areas",
        "expected_location": "Chennai",
        "expected_change_intent": False,
        "expected_change_type": ChangeType.STATIC_SEMANTIC_SEARCH.value,
    },
    {
        "id": "Q2",
        "query": "Find industrial buildings within 2 km of highways.",
        "expected_target": "industrial buildings",
        "expected_relation": "within",
        "expected_distance_km": 2.0,
        "expected_reference": "highways",
        "expected_change_intent": False,
        "expected_change_type": ChangeType.STATIC_SEMANTIC_SEARCH.value,
    },
    {
        "id": "Q3",
        "query": "Show water bodies in 2024.",
        "expected_target": "water bodies",
        "expected_start_date": "2024-01-01",
        "expected_end_date": "2024-12-31",
        "expected_change_intent": False,
        "expected_change_type": ChangeType.STATIC_SEMANTIC_SEARCH.value,
    },
    {
        "id": "Q4",
        "query": "Find areas that changed between 2020 and 2025.",
        "expected_start_date": "2020-01-01",
        "expected_end_date": "2025-12-31",
        "expected_change_intent": True,
        "expected_change_type": ChangeType.GENERIC_CHANGE.value,
    },
    {
        "id": "Q5",
        "query": "Find road development.",
        "expected_change_intent": True,
        "expected_change_type": ChangeType.ROAD_DEVELOPMENT.value,
    },
]


def run_benchmark(verbose: bool = True) -> bool:
    parser = QueryParser()
    all_passed = True

    print("=" * 80)
    print("SIH26227 / SH227 - PART 1 SEMANTIC RETRIEVAL: QUERY PARSER BENCHMARK")
    print("Organization: Ministry of Defence | Theme: Space Technology")
    print("=" * 80)

    for item in REQUIRED_BENCHMARK_QUERIES:
        q_id = item["id"]
        raw = item["query"]
        res = parser.parse(raw)
        passed = True
        failures = []

        # Target assertion
        if "expected_target" in item:
            if item["expected_target"] not in res.semantic_targets:
                passed = False
                failures.append(f"Expected target '{item['expected_target']}', got {res.semantic_targets}")

        # Location assertion
        if "expected_location" in item:
            if item["expected_location"] not in res.location_names:
                passed = False
                failures.append(f"Expected location '{item['expected_location']}', got {res.location_names}")

        # Spatial relation assertion
        if "expected_relation" in item:
            matched_rel = any(
                r.get("relation") == item["expected_relation"]
                and r.get("distance_km") == item.get("expected_distance_km")
                for r in res.spatial_relations
            )
            if not matched_rel:
                passed = False
                failures.append(
                    f"Spatial relation mismatch. Expected {item['expected_relation']} {item.get('expected_distance_km')}km, got {res.spatial_relations}"
                )

        # Dates assertion
        if "expected_start_date" in item:
            if res.start_date != item["expected_start_date"] or res.end_date != item["expected_end_date"]:
                passed = False
                failures.append(
                    f"Date mismatch: Expected ({item['expected_start_date']}, {item['expected_end_date']}), got ({res.start_date}, {res.end_date})"
                )

        # Change intent & type assertion
        if "expected_change_intent" in item:
            if res.change_intent != item["expected_change_intent"]:
                passed = False
                failures.append(f"Expected change_intent={item['expected_change_intent']}, got {res.change_intent}")

        if "expected_change_type" in item:
            if res.requested_change_type != item["expected_change_type"]:
                passed = False
                failures.append(
                    f"Expected change_type='{item['expected_change_type']}', got '{res.requested_change_type}'"
                )

        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False

        print(f"\n{status} {q_id}: \"{raw}\"")
        if verbose or not passed:
            print(f"       Normalized: \"{res.normalized_semantic_query}\"")
            print(f"       Targets   : {res.semantic_targets}")
            print(f"       Locations : {res.location_names}")
            print(f"       Spatial   : {res.spatial_relations}")
            print(f"       Temporal  : {res.start_date} -> {res.end_date}")
            print(f"       Routing   : change_intent={res.change_intent} | {res.requested_change_type}")
            print(f"       Confidence: {res.confidence}")
            print(f"       Warnings  : {res.warnings}")
            if failures:
                for f in failures:
                    print(f"       >> ERROR: {f}")

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULT: {'ALL 5 TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)
    return all_passed


def main():
    parser_cli = argparse.ArgumentParser(description="Test query parser for SIH26227 Space Tech")
    parser_cli.add_argument("--query", "-q", type=str, help="Custom query to test")
    parser_cli.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser_cli.parse_args()

    if args.query:
        qp = QueryParser()
        res = qp.parse(args.query)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"Query     : {res.raw_query}")
            print(f"Normalized: {res.normalized_semantic_query}")
            print(f"Targets   : {res.semantic_targets}")
            print(f"Locations : {res.location_names}")
            print(f"Spatial   : {res.spatial_relations}")
            print(f"Temporal  : {res.start_date} to {res.end_date}")
            print(f"Sensors   : {res.sensor_constraints}")
            print(f"Routing   : {res.requested_change_type} (intent={res.change_intent})")
            print(f"Confidence: {res.confidence}")
            print(f"Warnings  : {res.warnings}")
    else:
        success = run_benchmark(verbose=True)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
