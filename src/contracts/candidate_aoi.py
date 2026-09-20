"""
Candidate AOI (Area of Interest) Data Contract.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Optional, Any


@dataclass
class CandidateAOI:
    """
    Contract representing a candidate Area of Interest (AOI)
    identified during satellite imagery semantic retrieval.
    """
    aoi_id: str
    name: str
    # Bounding box in geographic coordinates: (min_lon, min_lat, max_lon, max_lat) - EPSG:4326
    bbox: Tuple[float, float, float, float]
    # Centroid coordinate: (lon, lat)
    centroid: Tuple[float, float]
    area_sqkm: float
    sensor: str
    resolution_m: float
    # Temporal range of satellite acquisition: (start_date, end_date) in ISO format
    temporal_range: Tuple[str, str]
    primary_semantic_class: str
    detected_change_type: Optional[str] = None
    
    # Retrieval & Ranking Scores (0.0 to 1.0)
    semantic_score: float = 0.0
    spatial_score: float = 0.0
    temporal_score: float = 0.0
    overall_score: float = 0.0
    
    # Verification & Quality Assurance Flags
    verification_flags: Dict[str, bool] = field(default_factory=lambda: {
        "gis_verified": False,
        "context_verified": False,
        "segmentation_verified": False,
        "resolution_compliant": True,
        "cloud_cover_compliant": True,
    })
    
    # Arbitrary satellite metadata (tile_id, cloud_percentage, sun_elevation, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateAOI dataclass to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAOI":
        """Instantiate CandidateAOI from dictionary."""
        return cls(
            aoi_id=data["aoi_id"],
            name=data["name"],
            bbox=tuple(data["bbox"]),  # type: ignore
            centroid=tuple(data["centroid"]),  # type: ignore
            area_sqkm=float(data.get("area_sqkm", 0.0)),
            sensor=data.get("sensor", "Unknown"),
            resolution_m=float(data.get("resolution_m", 10.0)),
            temporal_range=tuple(data.get("temporal_range", ("", ""))),  # type: ignore
            primary_semantic_class=data.get("primary_semantic_class", "unknown"),
            detected_change_type=data.get("detected_change_type"),
            semantic_score=float(data.get("semantic_score", 0.0)),
            spatial_score=float(data.get("spatial_score", 0.0)),
            temporal_score=float(data.get("temporal_score", 0.0)),
            overall_score=float(data.get("overall_score", 0.0)),
            verification_flags=data.get("verification_flags", {}),
            metadata=data.get("metadata", {}),
        )
