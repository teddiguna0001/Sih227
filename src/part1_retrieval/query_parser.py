"""
Deterministic Query Parser for Satellite Imagery Semantic Retrieval.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from src.part1_retrieval.query_schema import ParsedQuery, ChangeType, SpatialRelation
from src.part1_retrieval.change_router import ChangeRouter
from src.part1_retrieval.gazetteer import Gazetteer


class OptionalLLMParser:
    """
    Optional NLP/LLM hook interface.
    CRITICAL RULE: The system NEVER depends entirely on an LLM.
    This module provides an optional refinement adapter while maintaining
    full fallback to deterministic rules.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.is_available = bool(api_key)

    def enhance(self, raw_query: str, parsed: ParsedQuery) -> ParsedQuery:
        """
        Optionally augment parsed query attributes if LLM is enabled,
        without overwriting high-confidence deterministic extractions.
        """
        if not self.is_available:
            return parsed
        # In this foundation tier, rule-based values are preserved
        return parsed


class QueryParser:
    """
    Production-grade deterministic query parser utilizing regex, lexical heuristics,
    domain gazetteers, and semantic change routing for space/defence retrieval.
    """

    def __init__(
        self,
        gazetteer: Optional[Gazetteer] = None,
        change_router: Optional[ChangeRouter] = None,
        llm_parser: Optional[OptionalLLMParser] = None,
    ):
        self.gazetteer = gazetteer or Gazetteer()
        self.change_router = change_router or ChangeRouter()
        self.llm_parser = llm_parser or OptionalLLMParser()

        # Known semantic targets & domain ontology mapping
        self.target_patterns = [
            (r"\bagricultural\s+areas?\b|\bfarmlands?\b|\bcroplands?\b|\bcrops?\b", "agricultural areas"),
            (r"\bindustrial\s+buildings?\b|\bfactories\b|\bindustrial\s+zones?\b|\bmanufacturing\s+plants?\b", "industrial buildings"),
            (r"\bwater\s+bodies\b|\bwater\s+body\b|\blakes?\b|\breservoirs?\b|\bponds?\b|\brivers?\b|\bwetlands?\b", "water bodies"),
            (r"\bhighways?\b|\bexpressways?\b|\binterstates?\b|\bfreeways?\b", "highways"),
            (r"\broad\s+development\b|\broads?\b|\bstreet\s+network\b|\bavenues?\b", "roads"),
            (r"\bvegetation\b|\bforests?\b|\bcanopy\b|\bwoodlands?\b|\bgreeneries\b", "vegetation"),
            (r"\burban\s+areas?\b|\bsettlements?\b|\bresidential\s+areas?\b|\bcities\b|\btowns\b", "urban areas"),
            (r"\bairfields?\b|\brunways?\b|\bairports?\b|\bhangars?\b|\bhelipads?\b", "airfields"),
            (r"\bports?\b|\bharbours?\b|\bshipyards?\b|\bnaval\s+bases?\b|\bdocks?\b", "ports"),
            (r"\bbridges?\b|\bflyovers?\b|\bviaducts?\b", "bridges"),
            (r"\bcoastal\s+regions?\b|\bcoastlines?\b|\bshorelines?\b", "coastal regions"),
            (r"\bbuildings?\b|\bstructures?\b|\bfacilities\b", "buildings"),
        ]

        # Sensor dictionaries
        self.sensor_patterns = [
            (r"\bsentinel-?2\b", "Sentinel-2"),
            (r"\bsentinel-?1\b", "Sentinel-1"),
            (r"\bsentinel\b", "Sentinel"),
            (r"\blandsat-?8\b", "Landsat-8"),
            (r"\blandsat-?9\b", "Landsat-9"),
            (r"\blandsat\b", "Landsat"),
            (r"\bcartosat-?2\b", "Cartosat-2"),
            (r"\bcartosat-?3\b", "Cartosat-3"),
            (r"\bcartosat\b", "Cartosat"),
            (r"\bresourcesat\b", "Resourcesat"),
            (r"\bplanetscope\b|\bplanet\b", "PlanetScope"),
            (r"\bworldview\b", "WorldView"),
            (r"\bmodis\b", "MODIS"),
            (r"\bsar\b", "SAR"),
            (r"\bmultispectral\b", "Multispectral"),
            (r"\bhyperspectral\b", "Hyperspectral"),
        ]

        # Resolution patterns
        self.resolution_patterns = [
            (r"\bvery\s+high\s+resolution\b|\bvhr\b", {"max_resolution_m": 1.0, "category": "VHR"}),
            (r"\bsub-?meter\b", {"max_resolution_m": 1.0, "category": "sub_meter"}),
            (r"\bhigh\s+resolution\b", {"max_resolution_m": 5.0, "category": "high_resolution"}),
            (r"\bmedium\s+resolution\b", {"max_resolution_m": 15.0, "category": "medium_resolution"}),
            (r"\b(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+(?:resolution|gsd)\b", "dynamic_m"),
            (r"\b(?:resolution|gsd)\s+(?:better\s+than|<|under)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\b", "better_than_m"),
        ]

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse raw natural language query deterministically into a typed ParsedQuery schema.
        """
        raw_query = query.strip()
        warnings: List[str] = []
        confidence = 1.0

        # 1. Normalized Semantic Query (strip initial conversational verbs: find, show, get, detect, search)
        normalized = re.sub(
            r"^(?:find|show|locate|detect|identify|search\s+for|retrieve|extract)\s+",
            "",
            raw_query,
            flags=re.IGNORECASE,
        ).strip()
        if not normalized:
            normalized = raw_query

        # 2. Extract Multi-temporal State Transitions ("from X to Y", "forest turned into farmland")
        initial_state, final_state = self._extract_state_transitions(raw_query)

        # 3. Extract Semantic Targets
        semantic_targets = self._extract_semantic_targets(raw_query)

        # 4. Extract Location Names & Geometry Constraints using Gazetteer
        location_names, geometry_constraints, loc_warnings = self._extract_locations(raw_query)
        warnings.extend(loc_warnings)

        # 5. Extract Spatial Relations & Proximity (e.g., "within 2 km of highways", "near Chennai")
        spatial_relations = self._extract_spatial_relations(raw_query, semantic_targets, location_names)

        # 6. Extract Temporal Constraints (Dates & Date Ranges)
        start_date, end_date = self._extract_temporal_constraints(raw_query)

        # 7. Extract Sensor Constraints
        sensor_constraints = self._extract_sensors(raw_query)

        # 8. Extract Resolution Constraints
        resolution_constraints = self._extract_resolution(raw_query)

        # 9. Change Routing
        change_intent, requested_change_type, route_conf, _ = self.change_router.route(
            raw_query=raw_query,
            semantic_targets=semantic_targets,
            initial_state=initial_state,
            final_state=final_state,
        )

        # 10. Generate Warnings and Validate Completeness (RULE 7: never invent constraints)
        if not location_names and not geometry_constraints:
            warnings.append(
                "requires_confirmation: No geographic location or bounding constraint specified; requires AOI selection."
            )
            confidence *= 0.90

        if not start_date and not end_date:
            if change_intent:
                warnings.append(
                    "requires_confirmation: Temporal change query specified without date bounds; default archive comparison will be requested."
                )
                confidence *= 0.88
            else:
                warnings.append("No temporal filter specified; search defaults to latest cloud-free satellite imagery.")

        if not semantic_targets and not change_intent:
            warnings.append("requires_confirmation: No specific semantic target identified in query.")
            confidence *= 0.70

        # Construct ParsedQuery
        parsed = ParsedQuery(
            raw_query=raw_query,
            normalized_semantic_query=normalized,
            semantic_targets=semantic_targets,
            initial_state=initial_state,
            final_state=final_state,
            location_names=location_names,
            geometry_constraints=geometry_constraints,
            spatial_relations=spatial_relations,
            start_date=start_date,
            end_date=end_date,
            sensor_constraints=sensor_constraints,
            resolution_constraints=resolution_constraints,
            change_intent=change_intent,
            requested_change_type=requested_change_type.value if requested_change_type else None,
            confidence=round(confidence * route_conf, 3),
            warnings=warnings,
        )

        # 11. Run logical validation
        validation_issues = parsed.validate()
        if validation_issues:
            parsed.warnings.extend(validation_issues)
            parsed.confidence = max(0.1, round(parsed.confidence * 0.8, 3))

        # 12. Optional LLM enhancement (if enabled, but never depended upon)
        if self.llm_parser.is_available:
            parsed = self.llm_parser.enhance(raw_query, parsed)

        return parsed

    def _extract_semantic_targets(self, query: str) -> List[str]:
        matched: List[Tuple[str, int, int]] = []
        for pattern, canonical in self.target_patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                matched.append((canonical, m.start(), m.end()))

        # Deduplicate and prioritize longer/more specific phrases (e.g., 'industrial buildings' over 'buildings')
        filtered: List[str] = []
        for canonical, start, end in matched:
            # Check if this match is contained within another longer matched span
            is_sub = False
            for other_can, o_start, o_end in matched:
                if other_can != canonical and o_start <= start and end <= o_end and (o_end - o_start > end - start):
                    is_sub = True
                    break
            if not is_sub and canonical not in filtered:
                filtered.append(canonical)

        return filtered

    def _extract_state_transitions(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        # e.g., "from forest to urban areas", "converted from farmland to buildings"
        m1 = re.search(r"\bfrom\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\.|$|,|\bin\b|\bnear\b)", query, re.I)
        if m1:
            init = m1.group(1).strip()
            fin = m1.group(2).strip()
            return init, fin

        m2 = re.search(r"\b([a-zA-Z\s]+?)\s+turned\s+into\s+([a-zA-Z\s]+?)(?:\.|$|,)", query, re.I)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()

        return None, None

    def _extract_locations(self, query: str) -> Tuple[List[str], Optional[Dict[str, Any]], List[str]]:
        matches, ambig_warnings = self.gazetteer.extract_places(query)
        location_names: List[str] = [m.name for m in matches]

        # Additional regex heuristic for prepositional location phrases if not in gazetteer
        # e.g. "near Ladakh", "around Pokhran", "over XYZ"
        prep_matches = re.findall(
            r"\b(?:near|around|over|in|at|close\s+to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
            query,
        )
        for loc in prep_matches:
            # Avoid matching stop words
            if loc.lower() not in ["january", "december", "satellite", "highways", "roads", "india"]:
                if loc not in location_names:
                    location_names.append(loc)

        geometry_constraints: Optional[Dict[str, Any]] = None
        if matches:
            primary = matches[0]
            geometry_constraints = {
                "type": "gazetteer_reference",
                "canonical_name": primary.canonical_name,
                "place_type": primary.place_type,
                "admin_level": primary.admin_level,
                "centroid": {"lon": primary.centroid[0], "lat": primary.centroid[1]},
                "bbox": {
                    "min_lon": primary.bbox[0],
                    "min_lat": primary.bbox[1],
                    "max_lon": primary.bbox[2],
                    "max_lat": primary.bbox[3],
                },
                "default_buffer_km": primary.default_buffer_km,
            }

        return location_names, geometry_constraints, ambig_warnings

    def _extract_spatial_relations(
        self, query: str, semantic_targets: List[str], location_names: List[str]
    ) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        primary_target = semantic_targets[0] if semantic_targets else "features"

        # 1. "within <dist> <unit> of <reference>"
        # e.g., "within 2 km of highways", "within 500 m of coast"
        within_pattern = r"\bwithin\s+(\d+(?:\.\d+)?)\s*(km|kilometers?|m|meters?|miles?)\s+of\s+([a-zA-Z0-9\s_-]+?)(?:\.|$|,|\band\b|\bin\s+\d{4})"
        for m in re.finditer(within_pattern, query, re.IGNORECASE):
            raw_dist = float(m.group(1))
            unit = m.group(2).lower()
            reference = m.group(3).strip()

            # Normalize to kilometers
            if "meter" in unit or unit == "m":
                dist_km = raw_dist / 1000.0
            elif "mile" in unit:
                dist_km = raw_dist * 1.60934
            else:
                dist_km = raw_dist

            relations.append({
                "relation": "within",
                "target": primary_target,
                "reference": reference,
                "distance_km": round(dist_km, 3),
                "buffer_km": round(dist_km, 3),
            })

        # 2. Directional spatial expressions: "north of", "south of", "east of", "west of"
        dir_pattern = r"\b(north\s+of|south\s+of|east\s+of|west\s+of)\s+([a-zA-Z0-9\s_-]+?)(?:\.|$|,|\bin\s+\d{4})"
        for m in re.finditer(dir_pattern, query, re.IGNORECASE):
            dir_type = m.group(1).strip().lower()
            reference = m.group(2).strip()
            relations.append({
                "relation": dir_type,
                "target": primary_target,
                "reference": reference,
                "distance_km": 50.0,
                "buffer_km": 50.0,
            })

        # 3. Fixed proximity expressions: "near coast", "near highway", "near highways"
        fixed_corridor_pattern = r"\bnear\s+(coast|coastline|highway|highways|canal|railway)(?:\.|$|,|\bin\s+\d{4})"
        for m in re.finditer(fixed_corridor_pattern, query, re.IGNORECASE):
            corridor = m.group(1).strip().lower()
            if not any(r["reference"].lower() == corridor for r in relations):
                relations.append({
                    "relation": "near",
                    "target": primary_target,
                    "reference": corridor,
                    "distance_km": 2.0,
                    "buffer_km": 2.0,
                })

        # 4. General "near <reference>", "around <reference>"
        near_pattern = r"\b(near|around)\s+([a-zA-Z0-9\s_-]+?)(?:\.|$|,|\band\b|\bwithin\b|\bin\s+\d{4})"
        for m in re.finditer(near_pattern, query, re.IGNORECASE):
            rel_name = m.group(1).lower()
            reference = m.group(2).strip()
            # If already captured, do not duplicate
            if any(r["reference"].lower() == reference.lower() for r in relations):
                continue
            default_buf = 25.0
            entry = self.gazetteer.lookup(reference)
            if entry:
                default_buf = entry.default_buffer_km

            relations.append({
                "relation": rel_name,
                "target": primary_target,
                "reference": reference,
                "distance_km": default_buf,
                "buffer_km": default_buf,
            })

        # 5. Topological expressions: "adjacent to <ref>", "along <ref>", "inside <ref>", "intersects <ref>"
        topo_pattern = r"\b(adjacent\s+to|along|inside|intersects)\s+([a-zA-Z0-9\s_-]+?)(?:\.|$|,)"
        for m in re.finditer(topo_pattern, query, re.IGNORECASE):
            rel_type = m.group(1).strip().lower()
            reference = m.group(2).strip()
            if any(r["reference"].lower() == reference.lower() for r in relations):
                continue
            dist_km = 1.0 if "adjacent" in rel_type else 0.5
            relations.append({
                "relation": rel_type,
                "target": primary_target,
                "reference": reference,
                "distance_km": dist_km,
                "buffer_km": dist_km,
            })

        return relations

    def _extract_temporal_constraints(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        # 1. "between <year1> and <year2>" e.g., "between 2020 and 2025"
        m_between = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", query, re.IGNORECASE)
        if m_between:
            y1, y2 = m_between.group(1), m_between.group(2)
            return f"{y1}-01-01", f"{y2}-12-31"

        # 2. "from <year1> to <year2>"
        m_from_to = re.search(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", query, re.IGNORECASE)
        if m_from_to:
            y1, y2 = m_from_to.group(1), m_from_to.group(2)
            return f"{y1}-01-01", f"{y2}-12-31"

        # 3. ISO date range "YYYY-MM-DD to YYYY-MM-DD"
        m_iso_range = re.search(r"\b(\d{4}-\d{2}-\d{2})\s*(?:to|and|-)\s*(\d{4}-\d{2}-\d{2})\b", query)
        if m_iso_range:
            return m_iso_range.group(1), m_iso_range.group(2)

        # 4. "in <year>" e.g., "in 2024"
        m_in_year = re.search(r"\bin\s+(\d{4})\b", query, re.IGNORECASE)
        if m_in_year:
            year = m_in_year.group(1)
            return f"{year}-01-01", f"{year}-12-31"

        # 5. "since <year>" or "after <year>"
        m_since = re.search(r"\b(?:since|after)\s+(\d{4})\b", query, re.IGNORECASE)
        if m_since:
            year = m_since.group(1)
            return f"{year}-01-01", None

        # 6. "before <year>"
        m_before = re.search(r"\bbefore\s+(\d{4})\b", query, re.IGNORECASE)
        if m_before:
            year = m_before.group(1)
            return None, f"{year}-12-31"

        return None, None

    def _extract_sensors(self, query: str) -> List[str]:
        sensors: List[str] = []
        for pat, canonical in self.sensor_patterns:
            if re.search(pat, query, re.IGNORECASE):
                if canonical not in sensors:
                    sensors.append(canonical)
        return sensors

    def _extract_resolution(self, query: str) -> Optional[Dict[str, Any]]:
        for pat, info in self.resolution_patterns:
            if info == "dynamic_m":
                m = re.search(pat, query, re.IGNORECASE)
                if m:
                    res_val = float(m.group(1))
                    return {"max_resolution_m": res_val, "category": f"{res_val}m"}
            elif info == "better_than_m":
                m = re.search(pat, query, re.IGNORECASE)
                if m:
                    res_val = float(m.group(1))
                    return {"max_resolution_m": res_val, "category": f"under_{res_val}m"}
            else:
                if re.search(pat, query, re.IGNORECASE):
                    return info  # type: ignore

        return None
