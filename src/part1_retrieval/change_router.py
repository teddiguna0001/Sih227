"""
Change Routing Engine for Satellite Imagery Multi-Temporal Analysis.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import re
from typing import Tuple, List, Optional, Dict
from src.part1_retrieval.query_schema import ChangeType


class ChangeRouter:
    """
    Deterministic rule-based router that classifies user queries into specific
    multi-temporal satellite change categories or static semantic search.
    """

    def __init__(self, custom_rules: Optional[Dict[str, List[str]]] = None):
        # Specific pattern sets ordered by semantic priority
        self.rules: Dict[ChangeType, List[str]] = {
            ChangeType.ROAD_DEVELOPMENT: [
                r"\broad\s+development\b",
                r"\broad\s+construction\b",
                r"\bnew\s+(?:roads?|highways?|expressways?|streets?)\b",
                r"\bhighway\s+expansion\b",
                r"\bpaving\s+of\s+roads?\b",
                r"\broad\s+network\s+expansion\b",
                r"\bcorridor\s+development\b",
                r"\btransport\s+infrastructure\s+growth\b",
            ],
            ChangeType.CONSTRUCTION_CHANGE: [
                r"\bconstruction(?:\s+activity|\s+work)?\b",
                r"\bbuilding\s+construction\b",
                r"\bnew\s+(?:structures?|buildings?|facilities|complexes?|plants?)\b",
                r"\bindustrial\s+(?:development|expansion|growth)\b",
                r"\berected\b",
                r"\bbuilt\s+up\b",
                r"\binfrastructure\s+construction\b",
            ],
            ChangeType.CLEARANCE: [
                r"\bclearance\b",
                r"\bcleared\s+land\b",
                r"\bland\s+clearing\b",
                r"\bearthworks\b",
                r"\bdemolition\b",
                r"\bsite\s+preparation\b",
                r"\brazed\b",
                r"\bflattened\s+land\b",
            ],
            ChangeType.WATER_EXPANSION: [
                r"\bwater\s+expansion\b",
                r"\breservoir\s+expansion\b",
                r"\blake\s+growth\b",
                r"\bflooding\b",
                r"\bflood\s+inundation\b",
                r"\bdam\s+filling\b",
                r"\bwater\s+level\s+rise\b",
                r"\bexpanded\s+water\b",
            ],
            ChangeType.WATER_CONTRACTION: [
                r"\bwater\s+contraction\b",
                r"\bdrying(?:\s+up)?\b",
                r"\bdried\s+up\b",
                r"\bdrought\b",
                r"\breservoir\s+depletion\b",
                r"\blake\s+shrinkage\b",
                r"\bwater\s+recession\b",
                r"\bshrunken\s+water\b",
            ],
            ChangeType.VEGETATION_LOSS: [
                r"\bvegetation\s+loss\b",
                r"\bdeforestation\b",
                r"\btree\s+(?:clearing|felling|cutting)\b",
                r"\bcanopy\s+loss\b",
                r"\bforest\s+degradation\b",
                r"\bloss\s+of\s+(?:greenery|trees|forest|vegetation)\b",
            ],
            ChangeType.URBAN_EXPANSION: [
                r"\burban\s+expansion\b",
                r"\burban\s+sprawl\b",
                r"\bcity\s+growth\b",
                r"\bsettlement\s+expansion\b",
                r"\bbuilt-?up\s+expansion\b",
                r"\bresidential\s+encroachment\b",
            ],
            ChangeType.GENERIC_APPEARANCE: [
                r"\bappeared\b",
                r"\bnewly\s+surfaced\b",
                r"\barose\b",
                r"\bemergence\s+of\b",
                r"\bnew\s+objects?\b",
            ],
            ChangeType.GENERIC_DISAPPEARANCE: [
                r"\bdisappeared\b",
                r"\bvanished\b",
                r"\bremoved\b",
                r"\beliminated\b",
            ],
            ChangeType.GENERIC_CHANGE: [
                r"\bchanged(?:\s+between|\s+over|\s+since|\s+from)?\b",
                r"\bchanges?\b",
                r"\baltered\b",
                r"\bmodifications?\b",
                r"\btransformed\b",
                r"\bdifferences?\b",
                r"\bmulti-?temporal\s+change\b",
            ],
        }

        if custom_rules:
            for k, patterns in custom_rules.items():
                try:
                    ctype = ChangeType(k)
                    self.rules[ctype].extend(patterns)
                except ValueError:
                    pass

    def route(
        self,
        raw_query: str,
        semantic_targets: Optional[List[str]] = None,
        initial_state: Optional[str] = None,
        final_state: Optional[str] = None,
    ) -> Tuple[bool, ChangeType, float, List[str]]:
        """
        Route the query to a change classification or static semantic search.

        Returns:
            (has_change_intent, routed_change_type, confidence, matched_patterns)
        """
        query_lower = raw_query.lower()
        semantic_targets = semantic_targets or []

        # 1. Check explicit state transition (initial_state -> final_state)
        if initial_state and final_state:
            init_l = initial_state.lower()
            fin_l = final_state.lower()
            if "vegetation" in init_l or "forest" in init_l:
                if any(x in fin_l for x in ["road", "highway"]):
                    return True, ChangeType.ROAD_DEVELOPMENT, 0.95, ["state_transition:veg->road"]
                if any(x in fin_l for x in ["urban", "building", "settlement"]):
                    return True, ChangeType.URBAN_EXPANSION, 0.95, ["state_transition:veg->urban"]
                return True, ChangeType.VEGETATION_LOSS, 0.92, ["state_transition:veg->loss"]
            if "water" in init_l and ("land" in fin_l or "dry" in fin_l):
                return True, ChangeType.WATER_CONTRACTION, 0.92, ["state_transition:water->dry"]
            if ("land" in init_l or "dry" in init_l) and "water" in fin_l:
                return True, ChangeType.WATER_EXPANSION, 0.92, ["state_transition:dry->water"]

        # 2. Check pattern matching in order of specificity
        priority_order = [
            ChangeType.ROAD_DEVELOPMENT,
            ChangeType.CLEARANCE,
            ChangeType.WATER_EXPANSION,
            ChangeType.WATER_CONTRACTION,
            ChangeType.VEGETATION_LOSS,
            ChangeType.URBAN_EXPANSION,
            ChangeType.CONSTRUCTION_CHANGE,
            ChangeType.GENERIC_APPEARANCE,
            ChangeType.GENERIC_DISAPPEARANCE,
            ChangeType.GENERIC_CHANGE,
        ]

        for change_type in priority_order:
            patterns = self.rules.get(change_type, [])
            matched = []
            for pat in patterns:
                if re.search(pat, query_lower):
                    matched.append(pat)
            if matched:
                return True, change_type, 0.92, matched

        # 3. Check combined target and generic verbs
        # e.g., "areas that changed" -> GENERIC_CHANGE
        if re.search(r"\bchanged\b|\bchange\b", query_lower):
            return True, ChangeType.GENERIC_CHANGE, 0.85, ["verb:change"]

        # 4. If no change intent detected, default to STATIC_SEMANTIC_SEARCH
        return False, ChangeType.STATIC_SEMANTIC_SEARCH, 1.0, ["static_semantic_query"]
