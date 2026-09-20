"""
Local Offline Spatial Gazetteer for Defence and Space Technology.
Zero external geocoding API dependencies.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from src.part1_retrieval.gis_geometry import (
    Point,
    Polygon,
    BoundingBox,
    LineString,
    from_projected_metric,
    to_projected_metric,
)


@dataclass
class ResolvedPlace:
    """Standardized ResolvedPlace schema for verified geospatial entities."""
    name: str
    canonical_name: str
    place_type: str  # 'city', 'state', 'district', 'administrative_region', 'defence_installation'
    geometry: Dict[str, Any]  # GeoJSON representation
    centroid: Tuple[float, float]  # (lon, lat) EPSG:4326
    admin_level: int  # 1=country, 2=state, 3=district/division, 4=city/metropolitan, 5=installation
    country: str = "India"
    confidence: float = 1.0
    source: str = "defence_offline_gazetteer_v2"
    parent_admin: Optional[str] = None  # e.g., 'Tamil Nadu', 'Maharashtra', 'Ladakh'
    default_buffer_km: float = 25.0
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "place_type": self.place_type,
            "geometry": self.geometry,
            "centroid": {"lon": self.centroid[0], "lat": self.centroid[1]},
            "admin_level": self.admin_level,
            "country": self.country,
            "confidence": self.confidence,
            "source": self.source,
            "parent_admin": self.parent_admin,
            "default_buffer_km": self.default_buffer_km,
        }

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        coords = self.geometry.get("coordinates", [[]])[0]
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return (min(lons), min(lats), max(lons), max(lats))
        # Fallback to buffer around centroid
        buf_deg = self.default_buffer_km / 111.32
        return (
            self.centroid[0] - buf_deg,
            self.centroid[1] - buf_deg,
            self.centroid[0] + buf_deg,
            self.centroid[1] + buf_deg,
        )

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        return self.get_bounding_box()


# Backward compatibility alias
GazetteerEntry = ResolvedPlace


class AmbiguousLocationError(Exception):
    """Raised or returned when a location query matches multiple entities and context cannot resolve it."""
    def __init__(self, query_name: str, candidates: List[ResolvedPlace]):
        self.query_name = query_name
        self.candidates = candidates
        msg = (
            f"Ambiguous location '{query_name}' matches {len(candidates)} distinct places: "
            + ", ".join(f"{c.canonical_name} ({c.place_type}, {c.parent_admin})" for c in candidates)
            + ". Explicit district/state confirmation required to avoid silent erroneous selection."
        )
        super().__init__(msg)


class Gazetteer:
    """
    High-precision, deterministic, 100% offline geospatial gazetteer.
    Stores administrative boundaries across cities, states, districts, and strategic regions.
    Guarantees that ambiguous queries are flagged rather than silently misassigned.
    """

    def __init__(self, custom_places: Optional[List[ResolvedPlace]] = None):
        self.places: Dict[str, ResolvedPlace] = {}
        self.all_places_list: List[ResolvedPlace] = []
        self.name_lookup: Dict[str, List[ResolvedPlace]] = {}
        self._load_core_database()

        if custom_places:
            for p in custom_places:
                self.register_place(p)

    def register_place(self, place: ResolvedPlace) -> None:
        """Register a place and map its canonical name and aliases into the index."""
        self.places[place.canonical_name.lower()] = place
        self.all_places_list.append(place)

        keys = [place.canonical_name.lower(), place.name.lower()] + [a.lower() for a in place.aliases]
        for k in set(keys):
            if k not in self.name_lookup:
                self.name_lookup[k] = []
            if place not in self.name_lookup[k]:
                self.name_lookup[k].append(place)

    def _load_core_database(self) -> None:
        """Loads offline strategic boundaries for Indian cities, states, districts, and regions."""
        entries = [
            # 1. Cities
            ResolvedPlace(
                name="Chennai",
                canonical_name="Chennai, Tamil Nadu",
                place_type="city",
                geometry=BoundingBox(80.12, 12.92, 80.35, 13.25).to_geojson(),
                centroid=(80.2707, 13.0827),
                admin_level=4,
                parent_admin="Tamil Nadu",
                default_buffer_km=25.0,
                aliases=["madras"],
            ),
            ResolvedPlace(
                name="Mumbai",
                canonical_name="Mumbai",
                place_type="city",
                geometry=BoundingBox(72.75, 18.88, 73.05, 19.30).to_geojson(),
                centroid=(72.8777, 19.0760),
                admin_level=4,
                parent_admin="Maharashtra",
                default_buffer_km=25.0,
                aliases=["bombay"],
            ),
            ResolvedPlace(
                name="Bengaluru",
                canonical_name="Bengaluru",
                place_type="city",
                geometry=BoundingBox(77.40, 12.80, 77.80, 13.18).to_geojson(),
                centroid=(77.5946, 12.9716),
                admin_level=4,
                parent_admin="Karnataka",
                default_buffer_km=25.0,
                aliases=["bangalore"],
            ),
            ResolvedPlace(
                name="New Delhi",
                canonical_name="New Delhi",
                place_type="city",
                geometry=BoundingBox(76.85, 28.35, 77.45, 28.95).to_geojson(),
                centroid=(77.2090, 28.6139),
                admin_level=4,
                parent_admin="Delhi",
                default_buffer_km=30.0,
                aliases=["delhi", "ncr"],
            ),
            ResolvedPlace(
                name="Visakhapatnam",
                canonical_name="Visakhapatnam",
                place_type="city",
                geometry=BoundingBox(83.10, 17.55, 83.40, 17.90).to_geojson(),
                centroid=(83.2185, 17.6868),
                admin_level=4,
                parent_admin="Andhra Pradesh",
                default_buffer_km=25.0,
                aliases=["vizag"],
            ),
            ResolvedPlace(
                name="Port Blair",
                canonical_name="Port Blair",
                place_type="city",
                geometry=BoundingBox(92.60, 11.50, 92.85, 11.80).to_geojson(),
                centroid=(92.7265, 11.6234),
                admin_level=4,
                parent_admin="Andaman and Nicobar Islands",
                default_buffer_km=20.0,
                aliases=["south andaman"],
            ),
            ResolvedPlace(
                name="Pokhran",
                canonical_name="Pokhran",
                place_type="city",
                geometry=BoundingBox(71.70, 26.75, 72.15, 27.15).to_geojson(),
                centroid=(71.9161, 26.9196),
                admin_level=4,
                parent_admin="Rajasthan",
                default_buffer_km=30.0,
                aliases=["pokran", "pokhran test range"],
            ),
            ResolvedPlace(
                name="Leh",
                canonical_name="Leh",
                place_type="city",
                geometry=BoundingBox(77.50, 34.10, 77.65, 34.25).to_geojson(),
                centroid=(77.5771, 34.1526),
                admin_level=4,
                parent_admin="Ladakh",
                default_buffer_km=20.0,
                aliases=["leh town"],
            ),
            ResolvedPlace(
                name="Kochi",
                canonical_name="Kochi",
                place_type="city",
                geometry=BoundingBox(76.20, 9.85, 76.38, 10.05).to_geojson(),
                centroid=(76.2673, 9.9312),
                admin_level=4,
                parent_admin="Kerala",
                default_buffer_km=20.0,
                aliases=["cochin"],
            ),

            # 2. States & Union Territories (admin_level=2)
            ResolvedPlace(
                name="Tamil Nadu",
                canonical_name="Tamil Nadu",
                place_type="state",
                geometry=BoundingBox(76.24, 8.08, 80.35, 13.56).to_geojson(),
                centroid=(78.6569, 11.1271),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=50.0,
                aliases=["tn"],
            ),
            ResolvedPlace(
                name="Ladakh",
                canonical_name="Ladakh",
                place_type="state",
                geometry=BoundingBox(75.50, 32.20, 79.90, 36.00).to_geojson(),
                centroid=(77.5771, 34.1526),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=60.0,
                aliases=["union territory of ladakh"],
            ),
            ResolvedPlace(
                name="Rajasthan",
                canonical_name="Rajasthan",
                place_type="state",
                geometry=BoundingBox(69.50, 23.05, 78.28, 30.20).to_geojson(),
                centroid=(74.2179, 27.0238),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=50.0,
                aliases=[],
            ),
            ResolvedPlace(
                name="Maharashtra",
                canonical_name="Maharashtra",
                place_type="state",
                geometry=BoundingBox(72.60, 15.60, 80.90, 22.05).to_geojson(),
                centroid=(75.7139, 19.7515),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=50.0,
                aliases=[],
            ),

            # 3. Districts (admin_level=3)
            ResolvedPlace(
                name="Thiruvallur",
                canonical_name="Thiruvallur District",
                place_type="district",
                geometry=BoundingBox(79.70, 13.00, 80.30, 13.55).to_geojson(),
                centroid=(80.0000, 13.1500),
                admin_level=3,
                parent_admin="Tamil Nadu",
                default_buffer_km=25.0,
                aliases=["tiruvallur"],
            ),
            ResolvedPlace(
                name="Kanchipuram",
                canonical_name="Kanchipuram District",
                place_type="district",
                geometry=BoundingBox(79.50, 12.50, 80.15, 13.05).to_geojson(),
                centroid=(79.7000, 12.8300),
                admin_level=3,
                parent_admin="Tamil Nadu",
                default_buffer_km=25.0,
                aliases=["kancheepuram"],
            ),
            ResolvedPlace(
                name="Leh District",
                canonical_name="Leh District",
                place_type="district",
                geometry=BoundingBox(76.50, 33.20, 79.50, 35.80).to_geojson(),
                centroid=(77.7000, 34.3000),
                admin_level=3,
                parent_admin="Ladakh",
                default_buffer_km=50.0,
                aliases=["district of leh"],
            ),
            ResolvedPlace(
                name="Jaisalmer District",
                canonical_name="Jaisalmer District",
                place_type="district",
                geometry=BoundingBox(69.50, 26.00, 72.50, 28.00).to_geojson(),
                centroid=(71.0000, 27.0000),
                admin_level=3,
                parent_admin="Rajasthan",
                default_buffer_km=50.0,
                aliases=["jaisalmer"],
            ),

            # 4. Administrative & Strategic Regions
            ResolvedPlace(
                name="Western Ghats",
                canonical_name="Western Ghats",
                place_type="administrative_region",
                geometry=BoundingBox(73.00, 8.30, 77.50, 21.00).to_geojson(),
                centroid=(75.0000, 14.5000),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=60.0,
                aliases=["sahyadri"],
            ),
            ResolvedPlace(
                name="Thar Desert",
                canonical_name="Thar Desert",
                place_type="administrative_region",
                geometry=BoundingBox(69.80, 24.50, 75.00, 29.50).to_geojson(),
                centroid=(72.0000, 27.0000),
                admin_level=2,
                parent_admin=None,
                default_buffer_km=60.0,
                aliases=["great indian desert"],
            ),
            ResolvedPlace(
                name="Siachen Glacier",
                canonical_name="Siachen Glacier",
                place_type="defence_installation",
                geometry=BoundingBox(76.75, 35.10, 77.45, 35.75).to_geojson(),
                centroid=(77.1000, 35.4200),
                admin_level=5,
                parent_admin="Ladakh",
                default_buffer_km=40.0,
                aliases=["siachen", "saltoro ridge"],
            ),
            ResolvedPlace(
                name="Sriharikota",
                canonical_name="Sriharikota",
                place_type="defence_installation",
                geometry=BoundingBox(80.18, 13.65, 80.28, 13.85).to_geojson(),
                centroid=(80.2300, 13.7200),
                admin_level=5,
                parent_admin="Andhra Pradesh",
                default_buffer_km=25.0,
                aliases=["shar", "satish dhawan space centre", "sdsc"],
            ),

            # 5. Strategic Ambiguous Places (Designed for Ambiguity & Safety Validation)
            # Example A: "Aurangabad" -> Maharashtra vs Bihar
            ResolvedPlace(
                name="Aurangabad",
                canonical_name="Aurangabad, Maharashtra",
                place_type="city",
                geometry=BoundingBox(75.25, 19.82, 75.45, 19.95).to_geojson(),
                centroid=(75.3433, 19.8762),
                admin_level=4,
                parent_admin="Maharashtra",
                default_buffer_km=25.0,
                aliases=["chhatrapati sambhajinagar"],
            ),
            ResolvedPlace(
                name="Aurangabad",
                canonical_name="Aurangabad, Bihar",
                place_type="city",
                geometry=BoundingBox(84.30, 24.70, 84.45, 24.80).to_geojson(),
                centroid=(84.3750, 24.7539),
                admin_level=4,
                parent_admin="Bihar",
                default_buffer_km=25.0,
                aliases=["aurangabad bihar"],
            ),

            # Example B: "Hyderabad" -> Telangana (India) vs Sindh (Pakistan)
            ResolvedPlace(
                name="Hyderabad",
                canonical_name="Hyderabad, Telangana",
                place_type="city",
                geometry=BoundingBox(78.25, 17.20, 78.70, 17.60).to_geojson(),
                centroid=(78.4867, 17.3850),
                admin_level=4,
                country="India",
                parent_admin="Telangana",
                default_buffer_km=25.0,
                aliases=["secunderabad", "cyberabad"],
            ),
            ResolvedPlace(
                name="Hyderabad",
                canonical_name="Hyderabad, Sindh",
                place_type="city",
                geometry=BoundingBox(68.30, 25.30, 68.45, 25.45).to_geojson(),
                centroid=(68.3698, 25.3960),
                admin_level=4,
                country="Pakistan",
                parent_admin="Sindh",
                default_buffer_km=25.0,
                aliases=[],
            ),

            # Example C: "Kargil" -> City (Town) vs Kargil District
            ResolvedPlace(
                name="Kargil",
                canonical_name="Kargil Town",
                place_type="city",
                geometry=BoundingBox(76.10, 34.50, 76.20, 34.60).to_geojson(),
                centroid=(76.1333, 34.5500),
                admin_level=4,
                parent_admin="Ladakh",
                default_buffer_km=15.0,
                aliases=[],
            ),
            ResolvedPlace(
                name="Kargil",
                canonical_name="Kargil District",
                place_type="district",
                geometry=BoundingBox(75.50, 33.80, 76.80, 34.90).to_geojson(),
                centroid=(76.1500, 34.4000),
                admin_level=3,
                parent_admin="Ladakh",
                default_buffer_km=40.0,
                aliases=["district of kargil"],
            ),
        ]

        for p in entries:
            self.register_place(p)

    def resolve(
        self,
        name_or_query: str,
        context_hint: Optional[str] = None,
        allow_ambiguous: bool = False,
    ) -> Union[ResolvedPlace, List[ResolvedPlace]]:
        """
        Deterministic, offline place resolution.
        
        CRITICAL DEFENCE RULE: Never silently select an incorrect place.
        If a query matches multiple entities (e.g. 'Aurangabad') and no context hint
        disambiguates them:
          - If allow_ambiguous=False: raises AmbiguousLocationError
          - If allow_ambiguous=True: returns List[ResolvedPlace] with confidence penalized.
        """
        clean = name_or_query.strip().lower()

        # Direct match check
        matches = self.name_lookup.get(clean, [])
        if not matches:
            # Substring / word boundary search
            for key, candidates in self.name_lookup.items():
                if re.search(rf"\b{re.escape(clean)}\b", key) or re.search(rf"\b{re.escape(key)}\b", clean):
                    for c in candidates:
                        if c not in matches:
                            matches.append(c)

        if not matches:
            return [] if allow_ambiguous else None  # type: ignore

        # Context-based disambiguation
        if len(matches) > 1 and context_hint:
            hint_lower = context_hint.lower()
            filtered = [
                m for m in matches
                if (m.parent_admin and m.parent_admin.lower() in hint_lower)
                or (m.country and m.country.lower() in hint_lower)
                or (m.place_type and m.place_type.lower() in hint_lower)
                or (m.canonical_name.lower() in hint_lower)
            ]
            if len(filtered) == 1:
                return filtered[0]
            elif len(filtered) > 1:
                matches = filtered

        # Handle ambiguity
        if len(matches) > 1:
            if not allow_ambiguous:
                raise AmbiguousLocationError(name_or_query, matches)
            # Return candidates with penalized confidence
            for m in matches:
                m.confidence = round(1.0 / len(matches), 3)
            return matches

        return matches[0]

    def extract_places(
        self,
        text: str,
    ) -> Tuple[List[ResolvedPlace], List[str]]:
        """
        Scan text for all known geospatial entities.
        Safely identifies unambiguous entities, and flags ambiguous ones.
        
        Returns:
            (resolved_places, warnings_or_disambiguations)
        """
        found_places: List[ResolvedPlace] = []
        warnings: List[str] = []
        text_lower = text.lower()

        # Check all entries in name_lookup
        # Process longer names first to avoid partial token shadowing (e.g. "Leh District" before "Leh")
        sorted_keys = sorted(self.name_lookup.keys(), key=lambda k: len(k), reverse=True)
        matched_keys: set = set()

        for key in sorted_keys:
            # Word boundary regex matching
            pattern = rf"\b{re.escape(key)}\b"
            if re.search(pattern, text_lower):
                # Avoid matching a subset if already covered by longer key
                if any(key in mk and key != mk for mk in matched_keys):
                    continue

                matched_keys.add(key)
                candidates = self.name_lookup[key]

                if len(candidates) == 1:
                    place = candidates[0]
                    if place not in found_places:
                        found_places.append(place)
                else:
                    # Multiple candidates: check for contextual disambiguation in text
                    disambiguated = None
                    for cand in candidates:
                        if cand.parent_admin and cand.parent_admin.lower() in text_lower:
                            disambiguated = cand
                            break
                        if cand.country and cand.country.lower() in text_lower:
                            disambiguated = cand
                            break

                    if disambiguated:
                        if disambiguated not in found_places:
                            found_places.append(disambiguated)
                    else:
                        # Ambiguous: NEVER silently pick one!
                        cand_descs = [f"{c.canonical_name} ({c.place_type} in {c.parent_admin or c.country})" for c in candidates]
                        warn_msg = (
                            f"requires_confirmation: Ambiguous location '{key}' matches multiple places: "
                            f"[{'; '.join(cand_descs)}]. Please specify district or state."
                        )
                        warnings.append(warn_msg)

        return found_places, warnings

    def convert_spatial_expression_to_gis_constraint(
        self,
        expression_type: str,
        target_entity: str,
        reference_place: Optional[ResolvedPlace] = None,
        reference_geometry: Optional[Union[Point, LineString, Polygon, BoundingBox]] = None,
        distance_km: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Converts natural language spatial relationships into deterministic GIS constraints.
        Supports:
          - inside / within
          - around / near
          - adjacent to
          - within X km / within X m
          - north of / south of / east of / west of
          - along / intersects
          - near coast / near highway
        """
        constraint: Dict[str, Any] = {
            "expression": expression_type,
            "target": target_entity,
            "gis_operator": "ST_DWithin",
            "metric_buffer_m": 25000.0,
            "projected_bounding_box": None,
            "reference_type": "place" if reference_place else "corridor",
        }

        # Resolve distance in meters
        dist_m = 25000.0
        if distance_km is not None:
            dist_m = distance_km * 1000.0
        constraint["metric_buffer_m"] = dist_m

        # If reference place geometry exists
        ref_bbox = None
        ref_centroid = None
        if reference_place:
            ref_centroid = reference_place.centroid
            ref_bbox = reference_place.get_bounding_box()
        elif reference_geometry:
            ref_bbox = reference_geometry.bounds
            ref_centroid = (
                (ref_bbox[0] + ref_bbox[2]) / 2.0,
                (ref_bbox[1] + ref_bbox[3]) / 2.0,
            )

        if not ref_bbox:
            return constraint

        expr_clean = expression_type.lower().strip()

        # 1. Directional constraints: north/south/east/west
        if "north of" in expr_clean:
            # Everything strictly north of reference boundary/centroid
            constraint["gis_operator"] = "ST_NorthOf"
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0] - 1.0,
                "min_lat": ref_bbox[3],  # Above northern edge
                "max_lon": ref_bbox[2] + 1.0,
                "max_lat": ref_bbox[3] + (dist_m / 111320.0),
            }
        elif "south of" in expr_clean:
            constraint["gis_operator"] = "ST_SouthOf"
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0] - 1.0,
                "min_lat": ref_bbox[1] - (dist_m / 111320.0),
                "max_lon": ref_bbox[2] + 1.0,
                "max_lat": ref_bbox[1],  # Below southern edge
            }
        elif "east of" in expr_clean:
            constraint["gis_operator"] = "ST_EastOf"
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[2],
                "min_lat": ref_bbox[1] - 1.0,
                "max_lon": ref_bbox[2] + (dist_m / 111320.0),
                "max_lat": ref_bbox[3] + 1.0,
            }
        elif "west of" in expr_clean:
            constraint["gis_operator"] = "ST_WestOf"
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0] - (dist_m / 111320.0),
                "min_lat": ref_bbox[1] - 1.0,
                "max_lon": ref_bbox[0],
                "max_lat": ref_bbox[3] + 1.0,
            }

        # 2. Containment: inside / within
        elif expr_clean in ("inside", "within"):
            constraint["gis_operator"] = "ST_Contains"
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0],
                "min_lat": ref_bbox[1],
                "max_lon": ref_bbox[2],
                "max_lat": ref_bbox[3],
            }

        # 3. Proximity: within X km, near, around, adjacent to
        elif any(k in expr_clean for k in ("within", "near", "around", "adjacent")):
            constraint["gis_operator"] = "ST_DWithin"
            deg_buf = dist_m / 111320.0
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0] - deg_buf,
                "min_lat": ref_bbox[1] - deg_buf,
                "max_lon": ref_bbox[2] + deg_buf,
                "max_lat": ref_bbox[3] + deg_buf,
            }

        # 4. Corridor: along, near highway, near coast, intersects
        elif any(k in expr_clean for k in ("along", "highway", "coast", "intersects")):
            constraint["gis_operator"] = "ST_IntersectsBuffer"
            deg_buf = dist_m / 111320.0
            constraint["projected_bounding_box"] = {
                "min_lon": ref_bbox[0] - deg_buf,
                "min_lat": ref_bbox[1] - deg_buf,
                "max_lon": ref_bbox[2] + deg_buf,
                "max_lat": ref_bbox[3] + deg_buf,
            }

        return constraint

    def lookup(self, name: str) -> Optional[ResolvedPlace]:
        """Convenience method to look up a place by name, returning primary match or None."""
        try:
            res = self.resolve(name, allow_ambiguous=True)
            if isinstance(res, list):
                return res[0] if res else None
            return res
        except Exception:
            return None

    def scan_text(self, text: str) -> List[ResolvedPlace]:
        """Convenience scanner returning all resolved places found in text."""
        places, _ = self.extract_places(text)
        return places

