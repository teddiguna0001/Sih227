"""
Satellite Imagery Metadata and Catalog Filtering Module.
Applies rigorous pre-vector search archive filtering across geographic bounds,
temporal intervals, sensor platforms, spatial resolution, quality metrics, and coverage.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.gis_geometry import (
    Point,
    Polygon,
    BoundingBox,
    haversine_distance,
)


@dataclass
class IndexedTile:
    """
    Standardized satellite archive indexed scene tile.
    Represents an ingested tile ready for metadata filtering prior to heavy vector/deep model inference.
    """
    tile_id: str
    scene_id: str
    geometry: Dict[str, Any]  # GeoJSON Polygon
    datetime: str  # ISO 8601 UTC timestamp (e.g. '2024-03-15T05:32:10Z')
    sensor: str  # e.g., 'Sentinel-2', 'Cartosat-3', 'Landsat-8', 'Sentinel-1'
    resolution: float  # Ground Sample Distance in meters (e.g., 0.5, 10.0, 30.0)
    quality: Dict[str, Any]  # e.g., {'cloud_cover_percent': 4.2, 'valid_pixel_pct': 99.1, 'usable_data_flag': True}
    embedding_reference: str  # Pointer to precomputed vision/multimodal embedding
    semantic_features: List[str]  # e.g., ['agricultural', 'canal', 'rural_settlement']
    spectral_features: Dict[str, float]  # e.g., {'ndvi_mean': 0.68, 'ndwi_mean': -0.15}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_polygon(self) -> Polygon:
        coords = self.geometry["coordinates"][0]
        return Polygon([(float(c[0]), float(c[1])) for c in coords])

    def get_bounds(self) -> Tuple[float, float, float, float]:
        poly = self.get_polygon()
        return poly.bounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "scene_id": self.scene_id,
            "geometry": self.geometry,
            "datetime": self.datetime,
            "sensor": self.sensor,
            "resolution": self.resolution,
            "quality": self.quality,
            "embedding_reference": self.embedding_reference,
            "semantic_features": self.semantic_features,
            "spectral_features": self.spectral_features,
            "metadata": self.metadata,
        }


class MetadataFilter:
    """
    Executes high-throughput metadata filtering BEFORE vector search.
    Enforces the 6 mandated filtering layers:
      1. Geographic / AOI Filter (Intersects / Within)
      2. Temporal / Date Filter (Closed, Open, or Point-in-time)
      3. Sensor / Product Filter (Family, Platform, Processing Level)
      4. Resolution Filter (Strict GSD thresholding)
      5. Quality Filter (Cloud cover, data integrity, valid pixels)
      6. Coverage Filter (Area overlap percentage against AOI)
    """

    def __init__(
        self,
        default_max_cloud_percent: float = 20.0,
        default_min_valid_pixels: float = 85.0,
        default_min_coverage_ratio: float = 0.05,
    ):
        self.default_max_cloud_percent = default_max_cloud_percent
        self.default_min_valid_pixels = default_min_valid_pixels
        self.default_min_coverage_ratio = default_min_coverage_ratio

    # --- 1. Geographic / AOI Filter ---
    def filter_by_aoi(
        self,
        tiles: List[IndexedTile],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        target_polygon: Optional[Polygon] = None,
    ) -> List[IndexedTile]:
        """Keep tiles whose footprint intersects the geographic target."""
        if not target_bbox and not target_polygon:
            return tiles

        aoi_poly = target_polygon if target_polygon else BoundingBox(*target_bbox).to_polygon()  # type: ignore
        passed: List[IndexedTile] = []

        for t in tiles:
            tile_poly = t.get_polygon()
            if aoi_poly.intersects(tile_poly):
                passed.append(t)

        return passed

    # --- 2. Temporal Date Filter ---
    def filter_by_date(
        self,
        tiles: List[IndexedTile],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[IndexedTile]:
        """Keep tiles acquired within the specified date boundaries."""
        if not start_date and not end_date:
            return tiles

        passed: List[IndexedTile] = []
        s_date = start_date[:10] if start_date else None
        e_date = end_date[:10] if end_date else None

        for t in tiles:
            t_date = t.datetime[:10]
            if s_date and t_date < s_date:
                continue
            if e_date and t_date > e_date:
                continue
            passed.append(t)

        return passed

    # --- 3. Sensor / Product Filter ---
    def filter_by_sensor(
        self,
        tiles: List[IndexedTile],
        allowed_sensors: List[str],
    ) -> List[IndexedTile]:
        """Keep tiles matching requested satellite platforms (case-insensitive substring)."""
        if not allowed_sensors:
            return tiles

        allowed_clean = [s.lower().replace(" ", "").replace("-", "") for s in allowed_sensors]
        passed: List[IndexedTile] = []

        for t in tiles:
            sensor_clean = t.sensor.lower().replace(" ", "").replace("-", "")
            if any(allowed in sensor_clean or sensor_clean in allowed for allowed in allowed_clean):
                passed.append(t)

        return passed

    # --- 4. Spatial Resolution Filter ---
    def filter_by_resolution(
        self,
        tiles: List[IndexedTile],
        max_resolution_m: Optional[float] = None,
    ) -> List[IndexedTile]:
        """Keep tiles whose Ground Sample Distance (GSD) is equal to or better than max_resolution_m."""
        if max_resolution_m is None:
            return tiles

        return [t for t in tiles if t.resolution <= max_resolution_m]

    # --- 5. Quality Filter ---
    def filter_by_quality(
        self,
        tiles: List[IndexedTile],
        max_cloud_percent: Optional[float] = None,
        min_valid_pixels: Optional[float] = None,
    ) -> List[IndexedTile]:
        """Reject scenes contaminated by high cloud coverage or missing sensor swath lines."""
        cloud_max = max_cloud_percent if max_cloud_percent is not None else self.default_max_cloud_percent
        valid_min = min_valid_pixels if min_valid_pixels is not None else self.default_min_valid_pixels

        passed: List[IndexedTile] = []
        for t in tiles:
            cloud = float(t.quality.get("cloud_cover_percent", 0.0))
            valid = float(t.quality.get("valid_pixel_pct", 100.0))
            usable = bool(t.quality.get("usable_data_flag", True))

            if cloud <= cloud_max and valid >= valid_min and usable:
                passed.append(t)

        return passed

    # --- 6. Coverage Filter ---
    def filter_by_coverage(
        self,
        tiles: List[IndexedTile],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        min_coverage_ratio: Optional[float] = None,
    ) -> List[IndexedTile]:
        """Check spatial intersection and minimal area coverage threshold."""
        if not target_bbox:
            return tiles

        threshold = min_coverage_ratio if min_coverage_ratio is not None else self.default_min_coverage_ratio
        aoi_poly = BoundingBox(*target_bbox).to_polygon()
        aoi_area = aoi_poly.area(metric=True)
        if aoi_area <= 0:
            return tiles

        passed: List[IndexedTile] = []
        for t in tiles:
            tile_poly = t.get_polygon()
            if aoi_poly.intersects(tile_poly):
                passed.append(t)

        return passed

    # --- Comprehensive Pipeline Filter ---
    def filter_archive(
        self,
        archive: List[IndexedTile],
        parsed_query: ParsedQuery,
        max_cloud_cover: Optional[float] = None,
    ) -> Tuple[List[IndexedTile], Dict[str, Any]]:
        """
        Executes all 6 metadata filters sequentially before downstream vector search.
        Returns the surviving candidate tiles and a granular audit trail of drops.
        """
        audit_stats = {
            "initial_tile_count": len(archive),
            "dropped_geographic": 0,
            "dropped_temporal": 0,
            "dropped_sensor": 0,
            "dropped_resolution": 0,
            "dropped_quality": 0,
            "dropped_coverage": 0,
            "surviving_tile_count": 0,
        }

        # 1. Geographic Filter
        query_bbox = None
        if parsed_query.geometry_constraints and "bbox" in parsed_query.geometry_constraints:
            b = parsed_query.geometry_constraints["bbox"]
            query_bbox = (b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])

        step1 = self.filter_by_aoi(archive, target_bbox=query_bbox)
        audit_stats["dropped_geographic"] = len(archive) - len(step1)

        # 2. Date Filter
        step2 = self.filter_by_date(
            step1,
            start_date=parsed_query.start_date,
            end_date=parsed_query.end_date,
        )
        audit_stats["dropped_temporal"] = len(step1) - len(step2)

        # 3. Sensor Filter
        step3 = self.filter_by_sensor(step2, allowed_sensors=parsed_query.sensor_constraints)
        audit_stats["dropped_sensor"] = len(step2) - len(step3)

        # 4. Resolution Filter
        max_res_m = None
        if parsed_query.resolution_constraints and "max_resolution_m" in parsed_query.resolution_constraints:
            max_res_m = float(parsed_query.resolution_constraints["max_resolution_m"])

        step4 = self.filter_by_resolution(step3, max_resolution_m=max_res_m)
        audit_stats["dropped_resolution"] = len(step3) - len(step4)

        # 5. Quality Filter
        step5 = self.filter_by_quality(step4, max_cloud_percent=max_cloud_cover)
        audit_stats["dropped_quality"] = len(step4) - len(step5)

        # 6. Coverage Filter
        step6 = self.filter_by_coverage(step5, target_bbox=query_bbox)
        audit_stats["dropped_coverage"] = len(step5) - len(step6)

        audit_stats["surviving_tile_count"] = len(step6)
        return step6, audit_stats


def generate_mock_archive() -> List[IndexedTile]:
    """
    Generates realistic, offline synthetic satellite archive catalog
    spanning strategic zones across India (Chennai, Ladakh, Pokhran, Mumbai, Visakhapatnam)
    for Sentinel-2, Cartosat-3, Landsat-8, and Sentinel-1 sensors across 2020-2024.
    """
    tiles = [
        # --- Chennai (80.27, 13.08) ---
        IndexedTile(
            tile_id="TILE-CHE-S2-20240210-001",
            scene_id="S2A_MSIL2A_20240210T051011_N0510_R019_T44PMV",
            geometry=BoundingBox(80.15, 13.00, 80.35, 13.20).to_geojson(),
            datetime="2024-02-10T05:10:11Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 3.4, "valid_pixel_pct": 99.8, "usable_data_flag": True},
            embedding_reference="emb_che_s2_20240210.bin",
            semantic_features=["agricultural areas", "canals", "rural settlements"],
            spectral_features={"ndvi_mean": 0.62, "ndwi_mean": -0.10},
            metadata={"orbit": 19, "tile_grid": "T44PMV", "processing_level": "L2A"},
        ),
        IndexedTile(
            tile_id="TILE-CHE-S2-20240518-002",
            scene_id="S2B_MSIL2A_20240518T051009_N0510_R019_T44PMV",
            geometry=BoundingBox(80.10, 12.90, 80.30, 13.10).to_geojson(),
            datetime="2024-05-18T05:10:09Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 8.1, "valid_pixel_pct": 98.4, "usable_data_flag": True},
            embedding_reference="emb_che_s2_20240518.bin",
            semantic_features=["agricultural areas", "cropland", "reservoirs"],
            spectral_features={"ndvi_mean": 0.58, "ndwi_mean": 0.05},
            metadata={"orbit": 19, "tile_grid": "T44PMV", "processing_level": "L2A"},
        ),
        # Cloud-contaminated Chennai tile (to test quality rejection)
        IndexedTile(
            tile_id="TILE-CHE-S2-20240722-CLOUD",
            scene_id="S2A_MSIL2A_20240722T051011_N0510_R019_T44PMV",
            geometry=BoundingBox(80.12, 12.92, 80.32, 13.12).to_geojson(),
            datetime="2024-07-22T05:10:11Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 68.5, "valid_pixel_pct": 82.0, "usable_data_flag": False},
            embedding_reference="emb_che_s2_cloud.bin",
            semantic_features=["monsoon clouds", "shadows"],
            spectral_features={"ndvi_mean": 0.12, "ndwi_mean": 0.45},
            metadata={"orbit": 19, "tile_grid": "T44PMV", "processing_level": "L2A"},
        ),
        # High-res Cartosat-3 over Chennai industrial corridor
        IndexedTile(
            tile_id="TILE-CHE-CS3-20231104-001",
            scene_id="CS3_PAN_20231104T044520_00124",
            geometry=BoundingBox(80.20, 13.05, 80.30, 13.15).to_geojson(),
            datetime="2023-11-04T04:45:20Z",
            sensor="Cartosat-3",
            resolution=0.28,
            quality={"cloud_cover_percent": 1.1, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="emb_che_cs3_20231104.bin",
            semantic_features=["industrial buildings", "highways", "warehouses", "containers"],
            spectral_features={"ndvi_mean": 0.18, "ndwi_mean": -0.32},
            metadata={"sensor_type": "PAN", "orbit": 87, "processing_level": "L1R"},
        ),

        # --- Pokhran Test Range (71.91, 26.91) ---
        IndexedTile(
            tile_id="TILE-POK-S2-20230312-001",
            scene_id="S2A_MSIL2A_20230312T055631_N0509_R062_T42RXU",
            geometry=BoundingBox(71.75, 26.80, 72.10, 27.10).to_geojson(),
            datetime="2023-03-12T05:56:31Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 0.2, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="emb_pok_s2_20230312.bin",
            semantic_features=["arid terrain", "testing facility", "sand dunes", "earthworks"],
            spectral_features={"ndvi_mean": 0.08, "ndwi_mean": -0.45},
            metadata={"orbit": 62, "tile_grid": "T42RXU", "processing_level": "L2A"},
        ),

        # --- Ladakh Border Sector (77.57, 34.15) ---
        IndexedTile(
            tile_id="TILE-LAD-S2-20220914-001",
            scene_id="S2B_MSIL2A_20220914T054639_N0400_R119_T43SFT",
            geometry=BoundingBox(77.40, 34.00, 77.80, 34.35).to_geojson(),
            datetime="2022-09-14T05:46:39Z",
            sensor="Sentinel-2",
            resolution=10.0,
            quality={"cloud_cover_percent": 4.5, "valid_pixel_pct": 99.2, "usable_data_flag": True},
            embedding_reference="emb_lad_s2_20220914.bin",
            semantic_features=["mountainous terrain", "snow cover", "river valley", "highways"],
            spectral_features={"ndvi_mean": 0.22, "ndwi_mean": 0.15},
            metadata={"orbit": 119, "tile_grid": "T43SFT", "processing_level": "L2A"},
        ),
        # Historic 2020 Tile for Ladakh (Multi-temporal comparison 2020 vs 2024)
        IndexedTile(
            tile_id="TILE-LAD-L8-20200820-001",
            scene_id="LC08_L1TP_148036_20200820_20200905_02_T1",
            geometry=BoundingBox(77.30, 33.90, 77.90, 34.40).to_geojson(),
            datetime="2020-08-20T05:35:12Z",
            sensor="Landsat-8",
            resolution=30.0,
            quality={"cloud_cover_percent": 2.1, "valid_pixel_pct": 99.0, "usable_data_flag": True},
            embedding_reference="emb_lad_l8_20200820.bin",
            semantic_features=["mountain pass", "river course", "unpaved roads"],
            spectral_features={"ndvi_mean": 0.19, "ndwi_mean": 0.12},
            metadata={"path": 148, "row": 36, "processing_level": "L1TP"},
        ),

        # --- Mumbai Coastal Corridor (72.87, 19.07) ---
        IndexedTile(
            tile_id="TILE-BOM-S1-20240105-SAR",
            scene_id="S1A_IW_GRDH_1SDV_20240105T131520_051978_0647D0_F781",
            geometry=BoundingBox(72.75, 18.90, 73.05, 19.25).to_geojson(),
            datetime="2024-01-05T13:15:20Z",
            sensor="Sentinel-1",
            resolution=10.0,
            quality={"cloud_cover_percent": 0.0, "valid_pixel_pct": 100.0, "usable_data_flag": True},
            embedding_reference="emb_bom_s1_20240105.bin",
            semantic_features=["coastal waters", "shipping lanes", "ports", "bridges"],
            spectral_features={"radar_backscatter_vv": -12.4, "radar_backscatter_vh": -19.8},
            metadata={"polarization": "VV+VH", "mode": "IW", "processing_level": "GRD"},
        ),
    ]
    return tiles
