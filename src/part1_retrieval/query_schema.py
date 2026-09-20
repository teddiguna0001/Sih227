"""
Query Schema Definition for Satellite Imagery Semantic Retrieval.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any


class ChangeType(str, Enum):
    """
    Standardized change classification types for satellite multi-temporal analysis.
    """
    CONSTRUCTION_CHANGE = "CONSTRUCTION_CHANGE"
    CLEARANCE = "CLEARANCE"
    ROAD_DEVELOPMENT = "ROAD_DEVELOPMENT"
    WATER_EXPANSION = "WATER_EXPANSION"
    WATER_CONTRACTION = "WATER_CONTRACTION"
    VEGETATION_LOSS = "VEGETATION_LOSS"
    URBAN_EXPANSION = "URBAN_EXPANSION"
    GENERIC_APPEARANCE = "GENERIC_APPEARANCE"
    GENERIC_DISAPPEARANCE = "GENERIC_DISAPPEARANCE"
    GENERIC_CHANGE = "GENERIC_CHANGE"
    STATIC_SEMANTIC_SEARCH = "STATIC_SEMANTIC_SEARCH"


@dataclass
class SpatialRelation:
    """
    Structured representation of spatial relationship and proximity constraints.
    """
    relation: str  # e.g., 'near', 'within', 'adjacent_to', 'along', 'inside'
    target: str    # e.g., 'industrial buildings', 'agricultural areas'
    reference: str # e.g., 'highways', 'Chennai', 'water bodies'
    distance_km: Optional[float] = None
    buffer_km: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedQuery:
    """
    Structured parsed query contract adhering strictly to the SIH26227 specification.
    
    Rule: Missing fields MUST be null/unknown/requires_confirmation.
    Never invent constraints.
    """
    # 1. Raw user input query
    raw_query: str

    # 2. Normalized query string without conversational stop phrases
    normalized_semantic_query: str

    # 3. Extracted semantic entity classes (e.g., ['agricultural areas', 'industrial buildings'])
    semantic_targets: List[str] = field(default_factory=list)

    # 4. Multi-temporal state transitions (null if not specified)
    initial_state: Optional[str] = None
    final_state: Optional[str] = None

    # 5. Named geographical entities extracted from the query
    location_names: List[str] = field(default_factory=list)

    # 6. Explicit geometry or bounding box / buffer constraints
    geometry_constraints: Optional[Dict[str, Any]] = None

    # 7. Spatial proximity & topological relation constraints
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)

    # 8. Temporal bounds (ISO date strings YYYY-MM-DD or YYYY)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # 9. Satellite sensor constraints (Sentinel-2, Landsat-8, Cartosat, etc.)
    sensor_constraints: List[str] = field(default_factory=list)

    # 10. Spatial resolution constraints (e.g., max_resolution_m, min_resolution_m)
    resolution_constraints: Optional[Dict[str, Any]] = None

    # 11. Flag indicating if temporal change analysis is requested
    change_intent: bool = False

    # 12. Routed change classification type (from ChangeType enum)
    requested_change_type: Optional[str] = None

    # 13. Parsing confidence score (0.0 to 1.0)
    confidence: float = 1.0

    # 14. Actionable warnings or requires_confirmation flags
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """
        Validate semantic and logical consistency of the parsed query.
        Returns a list of validation error/warning strings.
        """
        validation_issues: List[str] = []

        # Temporal bounds check
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                validation_issues.append(
                    f"Temporal inconsistency: start_date ({self.start_date}) is after end_date ({self.end_date})"
                )

        # Spatial relations distance check
        for rel in self.spatial_relations:
            dist = rel.get("distance_km")
            if dist is not None and dist <= 0:
                validation_issues.append(f"Invalid spatial distance constraint: {dist} km <= 0")

        # Confidence bounds check
        if not (0.0 <= self.confidence <= 1.0):
            validation_issues.append(f"Confidence score {self.confidence} out of range [0.0, 1.0]")

        # Change intent vs change type alignment
        if self.change_intent and self.requested_change_type == ChangeType.STATIC_SEMANTIC_SEARCH.value:
            validation_issues.append("Change intent is True but change type routed as STATIC_SEMANTIC_SEARCH")

        return validation_issues

    def to_dict(self) -> Dict[str, Any]:
        """Export parsed query to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedQuery":
        """Reconstruct ParsedQuery instance from dictionary."""
        return cls(
            raw_query=data["raw_query"],
            normalized_semantic_query=data.get("normalized_semantic_query", ""),
            semantic_targets=data.get("semantic_targets", []),
            initial_state=data.get("initial_state"),
            final_state=data.get("final_state"),
            location_names=data.get("location_names", []),
            geometry_constraints=data.get("geometry_constraints"),
            spatial_relations=data.get("spatial_relations", []),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            sensor_constraints=data.get("sensor_constraints", []),
            resolution_constraints=data.get("resolution_constraints"),
            change_intent=bool(data.get("change_intent", False)),
            requested_change_type=data.get("requested_change_type"),
            confidence=float(data.get("confidence", 1.0)),
            warnings=data.get("warnings", []),
        )
