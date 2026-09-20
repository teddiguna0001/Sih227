"""
Multi-Scale Geospatial Context Verifier Module.

Evaluates candidates hierarchically across small-scale (local candidate footprint),
medium-scale (neighborhood land-use), and large-scale (regional corridors and terrain)
to calculate an explainable Context Consistency Score.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery


@dataclass
class MultiScaleContext:
    """Represents contextual observations across three spatial scales."""
    small_scale: List[str]   # Local scale: ~500m - 1km (rooftop geometry, immediate footprint)
    medium_scale: List[str]  # Neighborhood scale: ~2km - 5km (district, surrounding land use)
    large_scale: List[str]   # Regional scale: ~10km - 25km (highways, terrain, water networks)


@dataclass
class ContextVerificationResult:
    """Detailed audit result of context verification."""
    is_verified: bool
    context_score: float  # [0.0, 1.0]
    conflict_detected: bool
    conflict_reason: Optional[str]
    scale_observations: Dict[str, List[str]]
    audit_trail: List[str]


class ContextVerifier:
    """
    Hierarchical Multi-Scale Context Verifier.
    
    Verifies candidate detections against spatial hierarchy:
    Small View -> Medium View -> Large View.
    
    Example:
      Query: "industrial area near highway"
      Small view: large roofs
      Medium view: warehouses
      Large view: warehouses + highway
      -> Strengthens candidate.
      
      Small view: large roofs
      Medium view: farmland / greenhouses
      Large view: rural basin (no highway)
      -> Reduces confidence (flags greenhouse anomaly).
    """

    # Incompatible pairings that trigger context conflicts
    CONTEXT_CONFLICT_RULES = [
        {
            "target": "industrial",
            "medium_conflict_tokens": ["farmland", "cropland", "greenhouse", "orchard", "rural_hamlet"],
            "conflict_message": "Context conflict: Large roof structures situated inside rural farmland/greenhouses indicate agricultural greenhouses or sheds rather than industrial warehouses.",
            "penalty": 0.55,
        },
        {
            "target": "highway",
            "medium_conflict_tokens": ["runway", "taxiway", "airport", "airfield", "hangar"],
            "conflict_message": "Context conflict: Long linear paved strip is surrounded by airport taxiways and runways, not a public transportation highway.",
            "penalty": 0.60,
        },
        {
            "target": "water",
            "medium_conflict_tokens": ["cloud_shadow", "mountain_shadow", "steep_cliff_shadow", "shadow"],
            "conflict_message": "Context conflict: Low-reflectance dark signature is caused by topographic or cloud shadow, not an open water body.",
            "penalty": 0.65,
        },
        {
            "target": "agriculture",
            "medium_conflict_tokens": ["arid_sand", "barren_rock", "construction_excavation", "saline_flat"],
            "conflict_message": "Context conflict: Open parcels lack irrigation networks and exhibit barren/construction soil characteristics.",
            "penalty": 0.50,
        },
        {
            "target": "industrial",
            "medium_conflict_tokens": ["dense_residential_colony", "residential_suburb", "slum_cluster", "apartments"],
            "conflict_message": "Context conflict: High rooftop density composed of small residential parcels rather than expansive industrial facilities.",
            "penalty": 0.40,
        },
    ]

    def __init__(self, pass_threshold: float = 0.50):
        self.pass_threshold = pass_threshold

    def extract_context_scales(self, candidate: CandidateAOI) -> MultiScaleContext:
        """
        Extracts multi-scale observations from candidate metadata or constituent tiles,
        or infers them from semantic properties.
        """
        meta = candidate.metadata or {}

        # 1. Direct explicit multi-scale context in metadata
        if "multiscale_context" in meta:
            msc = meta["multiscale_context"]
            return MultiScaleContext(
                small_scale=[s.lower() for s in msc.get("small", [])],
                medium_scale=[s.lower() for s in msc.get("medium", [])],
                large_scale=[s.lower() for s in msc.get("large", [])],
            )
        
        if "context_scales" in meta:
            msc = meta["context_scales"]
            return MultiScaleContext(
                small_scale=[s.lower() for s in msc.get("small", [])],
                medium_scale=[s.lower() for s in msc.get("medium", [])],
                large_scale=[s.lower() for s in msc.get("large", [])],
            )

        # 2. Derive from constituent tiles and semantic features
        features = set()
        for f in meta.get("semantic_features", []):
            features.add(f.lower())

        for tile in meta.get("constituent_tiles", []):
            for f in tile.get("metadata", {}).get("semantic_features", []):
                features.add(f.lower())

        cls_lower = candidate.primary_semantic_class.lower()

        # Build scale-appropriate tags based on class and features
        small: List[str] = []
        medium: List[str] = []
        large: List[str] = []

        if "industrial" in cls_lower or "building" in cls_lower:
            small = ["large_roofs", "paved_aprons", "high_reflectance_geometry"]
            medium = ["warehouses", "logistics_yards", "industrial_estate"]
            large = ["transport_corridor", "highway_network", "freight_artery"]
        elif "water" in cls_lower or "reservoir" in cls_lower or "lake" in cls_lower:
            small = ["specular_water_surface", "shoreline_boundary"]
            medium = ["water_body", "wetland_buffer", "drainage_catchment"]
            large = ["river_basin", "coastal_drainage", "regional_hydrology"]
        elif "agriculture" in cls_lower or "crop" in cls_lower or "farm" in cls_lower:
            small = ["crop_field_plots", "vegetation_canopy", "tilled_soil"]
            medium = ["agricultural_farmland", "irrigation_canals", "rural_pockets"]
            large = ["fertile_plains", "river_valley", "peri_urban_greenbelt"]
        elif "highway" in cls_lower or "road" in cls_lower:
            small = ["paved_road_strip", "road_markings", "median_strip"]
            medium = ["highway_corridor", "interchanges", "access_ramps"]
            large = ["interstate_artery", "regional_connectivity_grid"]
        else:
            small = [f"{cls_lower}_footprint"]
            medium = [f"{cls_lower}_surroundings"]
            large = ["regional_terrain"]

        # Augment with any explicit tile semantic features
        for f in features:
            if any(k in f for k in ["roof", "building", "patch", "strip"]):
                small.append(f)
            elif any(k in f for k in ["park", "estate", "farmland", "airport", "residential", "shadow"]):
                medium.append(f)
            else:
                large.append(f)

        return MultiScaleContext(
            small_scale=small,
            medium_scale=medium,
            large_scale=large,
        )

    def verify(
        self,
        candidate: CandidateAOI,
        parsed_query: ParsedQuery,
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify candidate against hierarchical spatial context.
        Returns:
            (is_verified, context_score, audit_logs)
        """
        msc = self.extract_context_scales(candidate)
        audit_logs: List[str] = []
        raw_query_lower = (parsed_query.raw_query or "").lower()
        semantic_targets = [t.lower() for t in (parsed_query.semantic_targets or [])]
        spatial_relations = parsed_query.spatial_relations or []

        audit_logs.append(
            f"Multi-Scale Context Inspection for AOI '{candidate.aoi_id}': "
            f"Small={msc.small_scale}, Medium={msc.medium_scale}, Large={msc.large_scale}"
        )

        base_score = 0.50
        conflict_detected = False
        conflict_reason: Optional[str] = None

        # ----------------------------------------------------------------------
        # 1. Small-scale check: Does local feature match target concepts?
        # ----------------------------------------------------------------------
        small_matches = 0
        small_str = " ".join(msc.small_scale)
        for t in semantic_targets:
            # Token match
            t_tokens = t.split()
            if any(token in small_str for token in t_tokens):
                small_matches += 1
            elif "industrial" in t and any(k in small_str for k in ["roof", "building", "paved"]):
                small_matches += 1
            elif "water" in t and any(k in small_str for k in ["water", "specular", "shoreline"]):
                small_matches += 1
            elif "agriculture" in t and any(k in small_str for k in ["crop", "canopy", "field", "farm"]):
                small_matches += 1
            elif "highway" in t and any(k in small_str for k in ["road", "strip", "paved", "median"]):
                small_matches += 1

        if small_matches > 0:
            base_score += 0.15
            audit_logs.append("Small-scale view (+0.15): Local structures align with semantic target signature.")
        else:
            base_score -= 0.10
            audit_logs.append("Small-scale view (-0.10): Local structures exhibit weak target alignment.")

        # ----------------------------------------------------------------------
        # 2. Medium-scale check: Does neighborhood context support or refute?
        # ----------------------------------------------------------------------
        med_str = " ".join(msc.medium_scale)
        med_support = 0

        # Positive reinforcements
        if any("industrial" in t for t in semantic_targets):
            if any(k in med_str for k in ["warehouse", "logistics", "industrial", "factory", "commercial"]):
                med_support += 1
        if any("water" in t for t in semantic_targets):
            if any(k in med_str for k in ["water", "wetland", "drainage", "river", "lake", "catchment"]):
                med_support += 1
        if any("agriculture" in t for t in semantic_targets):
            if any(k in med_str for k in ["farm", "cropland", "rural", "irrigation", "orchard"]):
                med_support += 1
        if any("highway" in t for t in semantic_targets):
            if any(k in med_str for k in ["corridor", "interchange", "road", "transit"]):
                med_support += 1

        if med_support > 0:
            base_score += 0.20
            audit_logs.append("Medium-scale view (+0.20): Surrounding neighborhood land-use strongly corroborates target.")

        # Check for hard negative conflicts in medium scale
        for rule in self.CONTEXT_CONFLICT_RULES:
            target_key = rule["target"]
            is_target_present = any(target_key in t for t in semantic_targets) or (target_key in raw_query_lower)

            if is_target_present:
                for conflict_token in rule["medium_conflict_tokens"]:
                    if conflict_token in med_str:
                        conflict_detected = True
                        conflict_reason = rule["conflict_message"]
                        penalty = rule["penalty"]
                        base_score -= penalty
                        audit_logs.append(f"CRITICAL CONTEXT CONFLICT (-{penalty:.2f}): {conflict_reason}")
                        break
            if conflict_detected:
                break

        # ----------------------------------------------------------------------
        # 3. Large-scale check: Do regional corridors and strategic geography align?
        # ----------------------------------------------------------------------
        large_str = " ".join(msc.large_scale)
        rel_support = 0

        for rel in spatial_relations:
            ref = rel.get("reference", "").lower()
            ref_tokens = ref.split()
            # If relation requested "near highway" or "along highway"
            if any(k in ref for k in ["highway", "road", "expressway"]):
                if any(k in large_str or k in med_str for k in ["highway", "transport", "freight", "artery", "road"]):
                    rel_support += 1
                    base_score += 0.15
                    audit_logs.append("Large-scale view (+0.15): Regional transportation corridor confirmed adjacent to candidate.")
                else:
                    base_score -= 0.15
                    audit_logs.append("Large-scale view (-0.15): Regional highway corridor missing from surrounding strategic geography.")
            elif any(k in ref for k in ["coast", "port", "harbour", "sea"]):
                if any(k in large_str for k in ["coast", "marine", "port", "coastal"]):
                    rel_support += 1
                    base_score += 0.15
                    audit_logs.append("Large-scale view (+0.15): Coastal / maritime buffer confirmed in regional context.")

        # Final context consistency score clamped to [0.0, 1.0]
        context_score = round(max(0.0, min(1.0, base_score)), 3)
        is_verified = (context_score >= self.pass_threshold) and not conflict_detected

        # Update candidate state
        candidate.verification_flags["context_verified"] = is_verified
        
        # Attach detailed metadata contract
        candidate.metadata["context_verification"] = {
            "is_verified": is_verified,
            "context_score": context_score,
            "conflict_detected": conflict_detected,
            "conflict_reason": conflict_reason,
            "scale_observations": {
                "small_scale": msc.small_scale,
                "medium_scale": msc.medium_scale,
                "large_scale": msc.large_scale,
            },
            "audit_trail": audit_logs,
        }

        return is_verified, context_score, audit_logs

    def verify_candidates(
        self,
        candidates: List[CandidateAOI],
        parsed_query: ParsedQuery,
    ) -> List[CandidateAOI]:
        """Batch execution of multi-scale context verification."""
        for c in candidates:
            self.verify(c, parsed_query)
        return candidates
