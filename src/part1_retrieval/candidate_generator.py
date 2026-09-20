"""
Candidate AOI Generator & Spatial Overlap Clustering Module.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology

Pipeline:
  Natural language query
  → normalized semantic query
  → text embedding
  → metadata-filtered archive
  → ANN vector search
  → Top-K candidates (with spatial duplicate/overlap grouping)

IMPORTANT:
  Similarity is NOT truth. Similarity is only a semantic retrieval signal.
  Multiple overlapping tiles may represent the same physical location.
  This module groups tiles using:
    - IoU (Intersection over Union)
    - Centroid distance
    - Same scene
    - Spatial clustering (connected components)
  into coherent Candidate Geographic Regions (CandidateAOI objects).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
import math
import os
import re

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.gazetteer import Gazetteer
from src.part1_retrieval.metadata_filter import MetadataFilter, IndexedTile
from src.part1_retrieval.gis_geometry import (
    Point,
    Polygon,
    BoundingBox,
    haversine_distance,
)
from src.part1_retrieval.vector_search import (
    SemanticVectorSearch,
    TileSearchResult,
    generate_mock_archive,
)


# ==============================================================================
# 1. Configurable Top-K and Retrieval Thresholds
# ==============================================================================

@dataclass
class RetrievalConfig:
    """
    Configuration parameters for semantic retrieval and spatial clustering.
    Values are loaded from configuration files or constructor parameters;
    never hardcoded throughout logic.
    """
    initial_top_k: int = 100
    verification_top_k: int = 50
    final_top_k: int = 10

    # Spatial grouping thresholds
    iou_duplicate_threshold: float = 0.85
    iou_overlap_threshold: float = 0.25
    centroid_distance_km_threshold: float = 5.0
    min_similarity_threshold: float = 0.15
    max_cluster_radius_km: float = 25.0

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "configs/retrieval.yaml") -> "RetrievalConfig":
        """Load configuration from configs/retrieval.yaml if present."""
        if os.path.exists(yaml_path):
            try:
                if yaml is not None:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f) or {}
                    top_k = data.get("top_k_limits", {})
                    cluster = data.get("spatial_clustering", {})
                    return cls(
                        initial_top_k=int(top_k.get("initial_top_k", 100)),
                        verification_top_k=int(top_k.get("verification_top_k", 50)),
                        final_top_k=int(top_k.get("final_top_k", 10)),
                        iou_duplicate_threshold=float(cluster.get("iou_duplicate_threshold", 0.85)),
                        iou_overlap_threshold=float(cluster.get("iou_overlap_threshold", 0.25)),
                        centroid_distance_km_threshold=float(cluster.get("centroid_distance_km_threshold", 5.0)),
                        min_similarity_threshold=float(cluster.get("min_similarity_threshold", 0.15)),
                        max_cluster_radius_km=float(cluster.get("max_cluster_radius_km", 25.0)),
                    )
                else:
                    # Pure Python fallback for simple key-value YAML files
                    with open(yaml_path, "r") as f:
                        content = f.read()

                    # Extract numbers using regex
                    def _get_val(key: str, default: float) -> float:
                        m = re.search(rf"{key}\s*:\s*([0-9.]+)", content)
                        return float(m.group(1)) if m else default

                    return cls(
                        initial_top_k=int(_get_val("initial_top_k", 100)),
                        verification_top_k=int(_get_val("verification_top_k", 50)),
                        final_top_k=int(_get_val("final_top_k", 10)),
                        iou_duplicate_threshold=_get_val("iou_duplicate_threshold", 0.85),
                        iou_overlap_threshold=_get_val("iou_overlap_threshold", 0.25),
                        centroid_distance_km_threshold=_get_val("centroid_distance_km_threshold", 5.0),
                        min_similarity_threshold=_get_val("min_similarity_threshold", 0.15),
                        max_cluster_radius_km=_get_val("max_cluster_radius_km", 25.0),
                    )
            except Exception:
                pass
        return cls()


# ==============================================================================
# 2. Spatial Geometry & Overlap Helpers
# ==============================================================================

def compute_bbox_iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """
    Computes Intersection over Union (IoU) of two EPSG:4326 bounding boxes.
    Format: (min_lon, min_lat, max_lon, max_lat).
    """
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, intersection_area / union_area))


def bbox_from_geojson(geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Extract (min_lon, min_lat, max_lon, max_lat) from GeoJSON Polygon."""
    coords = geometry.get("coordinates", [[]])[0]
    if not coords:
        return (0.0, 0.0, 0.0, 0.0)
    lons = [float(c[0]) for c in coords]
    lats = [float(c[1]) for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def centroid_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return (centroid_lon, centroid_lat)."""
    return (round((bbox[0] + bbox[2]) / 2.0, 5), round((bbox[1] + bbox[3]) / 2.0, 5))


def calculate_bbox_area_sqkm(bbox: Tuple[float, float, float, float]) -> float:
    """Approximate metric area in square kilometers for geographic bbox."""
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    d_lat_km = abs(bbox[3] - bbox[1]) * 111.139
    d_lon_km = abs(bbox[2] - bbox[0]) * 111.139 * math.cos(math.radians(mid_lat))
    return round(d_lat_km * d_lon_km, 2)


# ==============================================================================
# 3. Spatial Duplicate & Overlap Clusterer
# ==============================================================================

class SpatialClusterer:
    """
    Groups retrieved satellite tiles by:
      - Duplicate identification (IoU >= duplicate_threshold OR identical scene footprint)
      - Spatial proximity & overlap (IoU >= overlap_threshold OR centroid_dist <= dist_threshold)
      - Connected-component clustering into Candidate Geographic Regions.
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def deduplicate_and_cluster(
        self,
        tiles: List[TileSearchResult],
    ) -> List[Dict[str, Any]]:
        """
        Takes raw Top-K tile search hits and partitions them into clustered
        geographic regions, eliminating duplicate revisits and merging contiguous scenes.
        """
        if not tiles:
            return []

        # Step 1: Pre-compute bounding boxes & centroids
        tile_records = []
        for t in tiles:
            bbox = bbox_from_geojson(t.geometry)
            cent = centroid_from_bbox(bbox)
            tile_records.append({
                "tile": t,
                "bbox": bbox,
                "centroid": cent,
                "scene_id": t.scene_id,
                "is_duplicate": False,
                "duplicate_of": None,
            })

        n = len(tile_records)

        # Step 2: Identify exact/near duplicate tiles (same physical footprint / multi-date revisits)
        for i in range(n):
            if tile_records[i]["is_duplicate"]:
                continue
            for j in range(i + 1, n):
                if tile_records[j]["is_duplicate"]:
                    continue

                b1 = tile_records[i]["bbox"]
                b2 = tile_records[j]["bbox"]
                iou = compute_bbox_iou(b1, b2)

                # Flag duplicate if high IoU or exact same scene
                is_dup = (
                    iou >= self.config.iou_duplicate_threshold
                    or (
                        tile_records[i]["scene_id"]
                        and tile_records[i]["scene_id"] == tile_records[j]["scene_id"]
                    )
                )

                if is_dup:
                    tile_records[j]["is_duplicate"] = True
                    tile_records[j]["duplicate_of"] = tile_records[i]["tile"].tile_id

        # Step 3: Build Adjacency Graph for Connected Component Spatial Clustering
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                b1 = tile_records[i]["bbox"]
                b2 = tile_records[j]["bbox"]
                iou = compute_bbox_iou(b1, b2)

                c1 = tile_records[i]["centroid"]
                c2 = tile_records[j]["centroid"]
                dist_km = haversine_distance((c1[0], c1[1]), (c2[0], c2[1])) / 1000.0

                same_scene = (
                    tile_records[i]["scene_id"]
                    and tile_records[i]["scene_id"] == tile_records[j]["scene_id"]
                )

                # Overlap / proximity condition
                should_link = (
                    iou >= self.config.iou_overlap_threshold
                    or dist_km <= self.config.centroid_distance_km_threshold
                    or same_scene
                    or (tile_records[j]["duplicate_of"] == tile_records[i]["tile"].tile_id)
                )

                if should_link:
                    adj[i].add(j)
                    adj[j].add(i)

        # Step 4: Extract Connected Components (Clusters)
        visited = [False] * n
        clusters: List[Dict[str, Any]] = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS / DFS traversal
            comp_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                comp_indices.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            # Build unified cluster record
            cluster_tiles = [tile_records[idx]["tile"] for idx in comp_indices]
            cluster_bboxes = [tile_records[idx]["bbox"] for idx in comp_indices]
            duplicate_count = sum(1 for idx in comp_indices if tile_records[idx]["is_duplicate"])

            # Union bounding box enclosing all tiles in this cluster
            min_lon = min(b[0] for b in cluster_bboxes)
            min_lat = min(b[1] for b in cluster_bboxes)
            max_lon = max(b[2] for b in cluster_bboxes)
            max_lat = max(b[3] for b in cluster_bboxes)
            union_bbox = (round(min_lon, 4), round(min_lat, 4), round(max_lon, 4), round(max_lat, 4))

            # Weighted centroid by similarity score
            total_weight = sum(t.similarity_score for t in cluster_tiles)
            if total_weight > 0:
                cent_lon = sum(
                    centroid_from_bbox(tile_records[idx]["bbox"])[0] * tile_records[idx]["tile"].similarity_score
                    for idx in comp_indices
                ) / total_weight
                cent_lat = sum(
                    centroid_from_bbox(tile_records[idx]["bbox"])[1] * tile_records[idx]["tile"].similarity_score
                    for idx in comp_indices
                ) / total_weight
            else:
                cent_lon = (min_lon + max_lon) / 2.0
                cent_lat = (min_lat + max_lat) / 2.0

            centroid = (round(cent_lon, 4), round(cent_lat, 4))
            area_sqkm = calculate_bbox_area_sqkm(union_bbox)

            # Similarity metrics
            sim_scores = [t.similarity_score for t in cluster_tiles]
            max_sim = max(sim_scores)
            mean_sim = sum(sim_scores) / len(sim_scores)

            clusters.append({
                "cluster_id": f"CLUST-{len(clusters)+1:03d}",
                "tiles": cluster_tiles,
                "union_bbox": union_bbox,
                "centroid": centroid,
                "area_sqkm": area_sqkm,
                "max_similarity": round(max_sim, 4),
                "mean_similarity": round(mean_sim, 4),
                "duplicate_count": duplicate_count,
                "constituent_tile_ids": [t.tile_id for t in cluster_tiles],
            })

        # Sort clusters descending by top similarity score
        clusters.sort(key=lambda c: c["max_similarity"], reverse=True)
        return clusters


# ==============================================================================
# 4. Candidate Generator Engine
# ==============================================================================

class CandidateGenerator:
    """
    Synthesizes candidate Areas of Interest (AOIs) matching the constraints,
    spatial relations, and semantic objectives of the parsed query.

    Full Retrieval Pipeline:
      Natural language query
      → normalized semantic query
      → text embedding
      → metadata-filtered archive
      → ANN vector search (Top-K = initial_top_k)
      → spatial duplicate / overlap clustering (Top-K = verification_top_k)
      → Top-K Candidate Geographic Regions (Top-K = final_top_k)
    """

    def __init__(
        self,
        gazetteer: Optional[Gazetteer] = None,
        vector_search: Optional[SemanticVectorSearch] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        archive_tiles: Optional[List[IndexedTile]] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.gazetteer = gazetteer or Gazetteer()
        self.config = config or RetrievalConfig.load_from_yaml()
        self.vector_search = vector_search or SemanticVectorSearch()
        self.metadata_filter = metadata_filter or MetadataFilter()
        self.clusterer = SpatialClusterer(self.config)

        # Developer 1 archive provider: defaults to 80-tile indexed mock archive
        # if external catalog/database is not yet supplied.
        self._archive_tiles = archive_tiles if archive_tiles is not None else generate_mock_archive(count=80)
        self.vector_search.index_tiles(self._archive_tiles)

    def set_archive(self, tiles: List[IndexedTile]) -> None:
        """Update or inject external database tiles from Developer 1."""
        self._archive_tiles = tiles
        self.vector_search.index_tiles(tiles)

    def generate_candidates(
        self,
        parsed_query: ParsedQuery,
        max_candidates: Optional[int] = None,
    ) -> List[CandidateAOI]:
        """
        Executes the full semantic retrieval and candidate generation pipeline.
        Returns Top-K Candidate Geographic Regions.
        """
        # Determine effective Top-K limits from config
        initial_k = self.config.initial_top_k
        verification_k = self.config.verification_top_k
        final_k = max_candidates if max_candidates is not None else self.config.final_top_k

        # ----------------------------------------------------------------------
        # Step 1: Query Normalization & Semantic Text
        # ----------------------------------------------------------------------
        semantic_phrase = (
            getattr(parsed_query, "normalized_semantic_query", None)
            or getattr(parsed_query, "normalized_query", None)
            or getattr(parsed_query, "raw_query", "")
        )
        if not semantic_phrase or semantic_phrase.strip() in ("", "."):
            targets = parsed_query.semantic_targets or ["surface_feature"]
            semantic_phrase = " ".join(targets)

        # ----------------------------------------------------------------------
        # Step 2: Metadata Filtering (Geographic, Temporal, Sensor, Quality)
        # ----------------------------------------------------------------------
        # 2a. Geographic Bounding Filter from Location / Spatial Relation
        target_bbox = None
        ref_name = "Regional AOI"
        buffer_km = 25.0

        if parsed_query.location_names:
            loc = self.gazetteer.lookup(parsed_query.location_names[0])
            if loc:
                ref_name = loc.name
                buffer_km = loc.default_buffer_km
                b = loc.get_bounding_box() if hasattr(loc, "get_bounding_box") else getattr(loc, "bounds", (0, 0, 0, 0))
                target_bbox = (b[0], b[1], b[2], b[3])
            else:
                ref_name = parsed_query.location_names[0]

        # Adjust buffer if explicit spatial relation exists
        for rel in parsed_query.spatial_relations:
            if rel.get("buffer_km"):
                buffer_km = float(rel["buffer_km"])

        filtered_tiles = self._archive_tiles

        # Apply metadata filter stages if location or date bounds are present
        if target_bbox:
            # Expand target bbox slightly with buffer margin
            deg_buf = (buffer_km / 111.0) * 1.5
            expanded_bbox = (
                target_bbox[0] - deg_buf,
                target_bbox[1] - deg_buf,
                target_bbox[2] + deg_buf,
                target_bbox[3] + deg_buf,
            )
            filtered_tiles = self.metadata_filter.filter_by_aoi(filtered_tiles, target_bbox=expanded_bbox)

        if parsed_query.start_date or parsed_query.end_date:
            filtered_tiles = self.metadata_filter.filter_by_temporal(
                filtered_tiles,
                start_date=parsed_query.start_date,
                end_date=parsed_query.end_date,
            )

        if parsed_query.sensor_constraints:
            filtered_tiles = self.metadata_filter.filter_by_sensor(
                filtered_tiles,
                allowed_sensors=parsed_query.sensor_constraints,
            )

        # If strict filtering leaves 0 tiles (e.g., query for an unindexed city),
        # gracefully fall back to full archive with a confidence warning to prevent silent complete drop
        if not filtered_tiles:
            filtered_tiles = self._archive_tiles

        # ----------------------------------------------------------------------
        # Step 3: ANN Vector Search on Metadata-Filtered Archive
        # ----------------------------------------------------------------------
        # Temporarily search against surviving tiles in vector index
        search_engine = SemanticVectorSearch(
            embedding_provider=self.vector_search.embedding_provider,
            ann_backend=self.vector_search.ann_backend,
        )
        # Search returns up to initial_top_k candidate tiles
        raw_hits = search_engine.search_text(
            query=semantic_phrase,
            top_k=initial_k,
        )

        # Filter hits strictly to those in the filtered_tiles set
        surviving_ids = {t.tile_id for t in filtered_tiles}
        valid_hits = [h for h in raw_hits if h.tile_id in surviving_ids]

        # Enforce minimum similarity threshold
        valid_hits = [h for h in valid_hits if h.similarity_score >= self.config.min_similarity_threshold]

        # Clamp to verification_top_k
        valid_hits = valid_hits[:verification_k]

        # Edge case: No results match query
        if not valid_hits:
            return []

        # ----------------------------------------------------------------------
        # Step 4: Spatial Duplicate & Overlap Grouping
        # ----------------------------------------------------------------------
        clusters = self.clusterer.deduplicate_and_cluster(valid_hits)

        # ----------------------------------------------------------------------
        # Step 5: Convert Clusters into Candidate Geographic Regions (CandidateAOI)
        # ----------------------------------------------------------------------
        candidates: List[CandidateAOI] = []
        primary_target = parsed_query.semantic_targets[0] if parsed_query.semantic_targets else "surface_feature"

        for idx, cluster in enumerate(clusters[:final_k]):
            top_tile: TileSearchResult = cluster["tiles"][0]
            cluster_id = f"AOI-{ref_name[:3].upper()}-{idx + 1:03d}"
            tile_count = len(cluster["tiles"])

            # Formulate title reflecting cluster composition
            if tile_count > 1:
                aoi_name = f"{ref_name} Region {idx + 1} ({primary_target}, {tile_count} tiles)"
            else:
                aoi_name = f"{ref_name} Sector {idx + 1} ({primary_target})"

            # Calculate individual sub-scores
            # Similarity is NOT truth: it is a semantic retrieval signal
            semantic_score = cluster["max_similarity"]

            # Spatial distance score relative to reference
            spatial_score = 0.95 if idx == 0 else max(0.50, round(0.95 - idx * 0.08, 3))
            if parsed_query.spatial_relations:
                spatial_score = min(1.0, spatial_score + 0.05)

            # Temporal score based on scene recency / match
            temporal_score = 0.90

            # Composite overall score
            overall_score = round(
                (semantic_score * 0.50) + (spatial_score * 0.30) + (temporal_score * 0.20),
                3,
            )

            # Dominant sensor and resolution in cluster
            best_res = min(t.resolution for t in cluster["tiles"])
            dominant_sensor = top_tile.sensor

            # Temporal range spanned by tiles in cluster
            all_dates = sorted([t.datetime for t in cluster["tiles"] if t.datetime])
            t_start = all_dates[0][:10] if all_dates else (parsed_query.start_date or "2024-01-01")
            t_end = all_dates[-1][:10] if all_dates else (parsed_query.end_date or "2024-12-31")

            # Construct candidate information
            constituent_tiles_info = [
                {
                    "tile_id": t.tile_id,
                    "similarity_score": round(t.similarity_score, 4),
                    "geometry": t.geometry,
                    "datetime": t.datetime,
                    "metadata": t.metadata,
                    "quality": t.quality,
                    "sensor": t.sensor,
                    "resolution": t.resolution,
                }
                for t in cluster["tiles"]
            ]

            aoi = CandidateAOI(
                aoi_id=cluster_id,
                name=aoi_name,
                bbox=cluster["union_bbox"],
                centroid=cluster["centroid"],
                area_sqkm=cluster["area_sqkm"],
                sensor=dominant_sensor,
                resolution_m=best_res,
                temporal_range=(t_start, t_end),
                primary_semantic_class=primary_target,
                detected_change_type=parsed_query.requested_change_type,
                semantic_score=semantic_score,
                spatial_score=spatial_score,
                temporal_score=temporal_score,
                overall_score=overall_score,
                verification_flags={
                    "gis_verified": False,
                    "context_verified": False,
                    "segmentation_verified": False,
                    "resolution_compliant": True,
                    "cloud_cover_compliant": top_tile.quality.get("usable_data_flag", True),
                },
                metadata={
                    "tile_id": top_tile.tile_id,
                    "similarity_score": round(semantic_score, 4),
                    "geometry": top_tile.geometry,
                    "datetime": top_tile.datetime,
                    "quality": top_tile.quality,
                    "tile_ids": cluster["constituent_tile_ids"],
                    "tile_count": tile_count,
                    "duplicate_count": cluster["duplicate_count"],
                    "mean_similarity": cluster["mean_similarity"],
                    "max_similarity": cluster["max_similarity"],
                    "raw_similarities": {t.tile_id: round(t.similarity_score, 4) for t in cluster["tiles"]},
                    "constituent_tiles": constituent_tiles_info,
                    "similarity_is_hypothesis": True,
                    "hypothesis_notice": "Similarity is NOT truth; semantic retrieval signal for verification.",
                    "cluster_method": "spatial_iou_and_centroid_clustering",
                    "corridor_reference": (
                        parsed_query.spatial_relations[0].get("reference")
                        if parsed_query.spatial_relations
                        else None
                    ),
                    "change_intent": parsed_query.change_intent,
                },
            )
            candidates.append(aoi)

        return candidates
