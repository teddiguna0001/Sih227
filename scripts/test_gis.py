#!/usr/bin/env python3
"""
CLI & API Helper for Geographic Understanding, Gazetteer, Metadata Filtering, and GIS Operations.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part1_retrieval.gazetteer import Gazetteer, AmbiguousLocationError
from src.part1_retrieval.metadata_filter import MetadataFilter, generate_mock_archive
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.gis_geometry import Point, LineString, Polygon, BoundingBox
from src.part1_retrieval.query_parser import QueryParser


def main():
    parser = argparse.ArgumentParser(description="GIS, Gazetteer & Metadata Filter Helper")
    parser.add_argument("--action", choices=["gazetteer_list", "gazetteer_resolve", "metadata_filter", "gis_eval"], default="gazetteer_list")
    parser.add_argument("--query", type=str, default="Chennai")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--filter_json", type=str, default="")
    parser.add_argument("--gis_json", type=str, default="")

    args = parser.parse_args()
    gazetteer = Gazetteer()

    if args.action == "gazetteer_list":
        places = [p.to_dict() for p in gazetteer.all_places_list]
        print(json.dumps({"total": len(places), "places": places}))

    elif args.action == "gazetteer_resolve":
        ctx = args.context.strip() if args.context else None
        try:
            resolved = gazetteer.resolve(args.query, context_hint=ctx, allow_ambiguous=False)
            print(json.dumps({"status": "resolved", "place": resolved.to_dict()}))
        except AmbiguousLocationError as e:
            print(json.dumps({
                "status": "ambiguous",
                "error": str(e),
                "query": e.query_name,
                "candidates": [c.to_dict() for c in e.candidates]
            }))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}))

    elif args.action == "metadata_filter":
        mf = MetadataFilter()
        archive = generate_mock_archive()
        qp = QueryParser(gazetteer=gazetteer)
        parsed = qp.parse(args.query)

        # Allow overrides from filter_json
        max_cloud = 20.0
        if args.filter_json:
            try:
                f_data = json.loads(args.filter_json)
                max_cloud = float(f_data.get("max_cloud_percent", 20.0))
            except Exception:
                pass

        surviving, audit = mf.filter_archive(archive, parsed, max_cloud_cover=max_cloud)
        print(json.dumps({
            "archive_total": len(archive),
            "surviving_count": len(surviving),
            "audit_trail": audit,
            "surviving_tiles": [t.to_dict() for t in surviving],
            "all_tiles": [t.to_dict() for t in archive],
        }))

    elif args.action == "gis_eval":
        # Deterministic GIS operations
        gv = GISVerifier()
        # Default test: Chennai NH48 corridor distance checks
        res = {
            "predicates": {
                "dwithin_500m_compliant": GISVerifier.dwithin(Point(80.20, 13.063), gv.highways["highways"], 500.0),
                "dwithin_500m_non_compliant": GISVerifier.dwithin(Point(80.20, 13.085), gv.highways["highways"], 500.0),
                "dwithin_1000m_compliant": GISVerifier.dwithin(Point(80.20, 13.067), gv.highways["highways"], 1000.0),
                "dwithin_1000m_non_compliant": GISVerifier.dwithin(Point(80.20, 13.085), gv.highways["highways"], 1000.0),
            },
            "corridors": {
                name: line.to_geojson() for name, line in gv.highways.items()
            },
            "coastlines": {
                name: line.to_geojson() for name, line in gv.coastlines.items()
            },
        }
        print(json.dumps(res))


if __name__ == "__main__":
    main()
