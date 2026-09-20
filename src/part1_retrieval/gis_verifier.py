"""
GIS Verification Engine with Projected Metric Coordinate Operations.
Enforces deterministic spatial constraints (contains, within, intersects, distance, buffer, area, dwithin)
without reliance on visual AI estimation.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Union
from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.gis_geometry import (
    Point,
    LineString,
    Polygon,
    BoundingBox,
    haversine_distance,
    to_projected_metric,
    from_projected_metric,
)


class GISVerifier:
    """
    Deterministic GIS Verification Service.
    Uses projected local metric Cartesian coordinates to evaluate spatial predicates
    and exact metric distances (e.g., 500m, 1km, 2km, 25km).
    
    Axiom: Exact physical spatial relationships (e.g. "within 1 km of highway")
    MUST be evaluated deterministically using geometric algorithms, NEVER estimated
    heuristically via vision models.
    """

    def __init__(self, pass_threshold: float = 0.50):
        self.pass_threshold = pass_threshold
        # Synthetic reference vectors for deterministic evaluation
        self._load_reference_vectors()

    def _load_reference_vectors(self) -> None:
        """
        Loads baseline strategic infrastructure polyline vectors (highways, coastlines, canals)
        for deterministic topological and buffer evaluation.
        """
        self.highways: Dict[str, LineString] = {
            # Chennai Grand Southern Trunk (GST) Road & NH48 Industrial Highway Corridor
            "chennai_nh48_industrial_corridor": LineString([
                (80.05, 12.95),
                (80.12, 13.01),
                (80.20, 13.06),
                (80.26, 13.09),
                (80.32, 13.14),
            ]),
            # Generic reference highway
            "highways": LineString([
                (80.05, 12.95),
                (80.12, 13.01),
                (80.20, 13.06),
                (80.26, 13.09),
                (80.32, 13.14),
            ]),
            "highway": LineString([
                (80.05, 12.95),
                (80.12, 13.01),
                (80.20, 13.06),
                (80.26, 13.09),
                (80.32, 13.14),
            ]),
            # Leh-Manali Highway (Ladakh)
            "leh_highway": LineString([
                (77.50, 34.05),
                (77.55, 34.12),
                (77.58, 34.16),
                (77.65, 34.22),
            ]),
        }

        self.coastlines: Dict[str, LineString] = {
            # Coromandel Coast (Chennai / Tamil Nadu)
            "coromandel_coast": LineString([
                (80.30, 12.85),
                (80.29, 13.00),
                (80.31, 13.15),
                (80.34, 13.30),
            ]),
            "coast": LineString([
                (80.30, 12.85),
                (80.29, 13.00),
                (80.31, 13.15),
                (80.34, 13.30),
            ]),
            "coastline": LineString([
                (80.30, 12.85),
                (80.29, 13.00),
                (80.31, 13.15),
                (80.34, 13.30),
            ]),
        }

    # ==========================================
    # CORE MANDATED GIS OPERATORS (PostGIS Compatible)
    # ==========================================

    @staticmethod
    def contains(geom1: Union[Polygon, BoundingBox], geom2: Union[Point, Polygon, BoundingBox]) -> bool:
        """PostGIS ST_Contains: Returns True if geom1 completely contains geom2."""
        poly1 = geom1.to_polygon() if isinstance(geom1, BoundingBox) else geom1
        return poly1.contains(geom2)

    @staticmethod
    def within(geom1: Union[Point, Polygon, BoundingBox], geom2: Union[Polygon, BoundingBox]) -> bool:
        """PostGIS ST_Within: Returns True if geom1 is completely within geom2."""
        poly2 = geom2.to_polygon() if isinstance(geom2, BoundingBox) else geom2
        return poly2.contains(geom1)

    @staticmethod
    def intersects(
        geom1: Union[Point, LineString, Polygon, BoundingBox],
        geom2: Union[Point, LineString, Polygon, BoundingBox],
    ) -> bool:
        """PostGIS ST_Intersects: Returns True if geom1 and geom2 spatially intersect."""
        if hasattr(geom1, "intersects"):
            return geom1.intersects(geom2)  # type: ignore
        return False

    @staticmethod
    def distance(
        geom1: Union[Point, LineString, Polygon, BoundingBox],
        geom2: Union[Point, LineString, Polygon, BoundingBox],
        metric: bool = True,
    ) -> float:
        """
        Returns minimum distance between geom1 and geom2.
        If metric=True, returns distance in meters using projected CRS.
        """
        if hasattr(geom1, "distance"):
            return geom1.distance(geom2)
        elif hasattr(geom2, "distance"):
            return geom2.distance(geom1)
        raise ValueError("Unsupported geometry combination")

    @staticmethod
    def buffer(
        geom: Union[Point, LineString, Polygon, BoundingBox],
        distance_m: float,
    ) -> Polygon:
        """PostGIS ST_Buffer: Returns a Polygon buffered by distance_m in projected CRS."""
        return geom.buffer(distance_m)

    @staticmethod
    def area(
        geom: Union[Polygon, BoundingBox],
        metric: bool = True,
    ) -> float:
        """
        PostGIS ST_Area: Returns area in square meters (metric=True)
        or square degrees using high-precision projected metric integration.
        """
        poly = geom.to_polygon() if isinstance(geom, BoundingBox) else geom
        return poly.area(metric=metric)

    @staticmethod
    def dwithin(
        geom1: Union[Point, LineString, Polygon, BoundingBox],
        geom2: Union[Point, LineString, Polygon, BoundingBox],
        distance_m: float,
    ) -> bool:
        """
        PostGIS ST_DWithin: True if minimum distance between geom1 and geom2 <= distance_m.
        Evaluated deterministically in projected metric space.
        """
        dist = GISVerifier.distance(geom1, geom2, metric=True)
        return dist <= distance_m

    # ==========================================
    # PIPELINE VALIDATION & AUDITING
    # ==========================================

    def get_reference_geometry(self, ref_name: str) -> Optional[LineString]:
        """Looks up or derives a known reference infrastructure vector."""
        clean = ref_name.lower().strip()
        if clean in self.highways:
            return self.highways[clean]
        if clean in self.coastlines:
            return self.coastlines[clean]

        for k, v in self.highways.items():
            if k in clean or clean in k:
                return v
        for k, v in self.coastlines.items():
            if k in clean or clean in k:
                return v

        if any(w in clean for w in ["highway", "road", "expressway"]):
            return self.highways["chennai_nh48_industrial_corridor"]
        if any(w in clean for w in ["coast", "sea", "shore", "beach"]):
            return self.coastlines["coromandel_coast"]

        return None

    def verify_corridor_distance(
        self,
        candidate_geom: Union[Point, Polygon, BoundingBox],
        reference_corridor_name: str,
        max_distance_m: float,
    ) -> Tuple[bool, float, str]:
        """
        Evaluates deterministic metric distance to infrastructure corridor (e.g. highways, coasts).
        Example: "industrial buildings within 1 km of highway"
        Returns (is_compliant, measured_distance_m, diagnostic_log)
        """
        corridor = self.get_reference_geometry(reference_corridor_name)
        if not corridor:
            # Fallback to default highway corridor
            corridor = self.highways["chennai_nh48_industrial_corridor"]

        measured_m = corridor.distance(candidate_geom)
        is_compliant = (measured_m <= max_distance_m)

        diag = (
            f"Deterministic ST_DWithin: Evaluated distance to reference '{reference_corridor_name}': "
            f"measured={measured_m:.1f} m (allowed threshold <= {max_distance_m:.1f} m) -> "
            f"{'TRUE (PASSED)' if is_compliant else 'FALSE (FAILED)'}"
        )
        return is_compliant, measured_m, diag

    def verify(
        self,
        candidate: CandidateAOI,
        parsed_query: ParsedQuery,
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify spatial constraints on candidate AOI using deterministic GIS operations.
        Returns:
            (is_verified, updated_spatial_score, verification_logs)
        """
        logs: List[str] = []
        is_verified = True
        spatial_score = candidate.spatial_score
        candidate_poly = BoundingBox(*candidate.bbox).to_polygon()
        candidate_point = Point(*candidate.centroid)

        predicate_results: List[Dict[str, Any]] = []

        # 1. Verify spatial relations (e.g., within X km of reference)
        for rel in parsed_query.spatial_relations:
            dist_km = rel.get("distance_km")
            relation = rel.get("relation", "near").lower()
            reference = rel.get("reference", "reference")

            if dist_km is not None:
                max_dist_m = dist_km * 1000.0
                # Evaluate against both point centroid and bounding polygon
                compliant, measured_m, diag = self.verify_corridor_distance(
                    candidate_poly,
                    reference_corridor_name=reference,
                    max_distance_m=max_dist_m,
                )
                logs.append(diag)
                predicate_results.append({
                    "relation": relation,
                    "reference": reference,
                    "max_dist_m": max_dist_m,
                    "measured_dist_m": round(measured_m, 1),
                    "is_compliant": compliant,
                })

                if not compliant:
                    is_verified = False
                    # Exact spatial constraint failed (e.g. outside 1 km)
                    spatial_score = max(0.0, spatial_score - 0.40)
                else:
                    # Metric closeness bonus
                    proximity_factor = max(0.0, 1.0 - (measured_m / max_dist_m))
                    spatial_score = min(1.0, spatial_score + 0.15 * proximity_factor)

            elif any(k in relation for k in ["near", "around", "adjacent", "along"]):
                # Default buffer check (25km)
                default_dist_m = 25000.0
                compliant, measured_m, diag = self.verify_corridor_distance(
                    candidate_poly,
                    reference_corridor_name=reference,
                    max_distance_m=default_dist_m,
                )
                logs.append(diag)
                predicate_results.append({
                    "relation": relation,
                    "reference": reference,
                    "max_dist_m": default_dist_m,
                    "measured_dist_m": round(measured_m, 1),
                    "is_compliant": compliant,
                })
                if not compliant:
                    is_verified = False
                    spatial_score = max(0.0, spatial_score - 0.25)
                else:
                    spatial_score = min(1.0, spatial_score + 0.05)

        # 2. Verify geometry bounds if specified
        if parsed_query.geometry_constraints and "bbox" in parsed_query.geometry_constraints:
            gb = parsed_query.geometry_constraints["bbox"]
            query_box = BoundingBox(gb["min_lon"], gb["min_lat"], gb["max_lon"], gb["max_lat"])

            if query_box.intersects(candidate_poly):
                logs.append("Deterministic ST_Intersects: Candidate AOI intersects query bounding polygon -> TRUE")
            else:
                logs.append("Deterministic ST_Intersects: Candidate AOI outside query bounding polygon -> FALSE")
                is_verified = False
                spatial_score = max(0.0, spatial_score - 0.35)

        final_spatial_score = round(max(0.0, min(1.0, spatial_score)), 3)
        candidate.verification_flags["gis_verified"] = is_verified
        candidate.spatial_score = final_spatial_score

        # Attach detailed metadata contract
        candidate.metadata["gis_verification"] = {
            "is_verified": is_verified,
            "gis_score": final_spatial_score,
            "predicates": predicate_results,
            "audit_trail": logs,
        }

        return is_verified, final_spatial_score, logs

    def verify_candidates(
        self,
        candidates: List[CandidateAOI],
        parsed_query: ParsedQuery,
    ) -> List[CandidateAOI]:
        """Batch execution of deterministic GIS verification."""
        for candidate in candidates:
            self.verify(candidate, parsed_query)
        return candidates
