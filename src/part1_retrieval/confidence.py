"""
Confidence Scoring and Quality Assurance Module.

Computes system-level retrieval confidence, risk factors, verification rates,
and user confirmation flags.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from typing import Dict, List, Any
from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery


class ConfidenceEngine:
    """
    Computes system-level retrieval confidence, verification pass rates,
    and user confirmation flags.
    
    Note: 'confidence' measures pipeline certainty in the retrieved set,
    distinct from the candidate-level 'relevance_score' (ranking metric).
    """

    def evaluate(
        self,
        parsed_query: ParsedQuery,
        candidate_aois: List[CandidateAOI],
    ) -> Dict[str, Any]:
        """
        Evaluate overall confidence across query parsing, candidate quality,
        and multi-stage verification stages (GIS, Context, Segmentation).
        """
        query_conf = parsed_query.confidence

        if not candidate_aois:
            return {
                "overall_confidence": round(query_conf * 0.5, 3),
                "query_parsing_confidence": query_conf,
                "candidate_quality_confidence": 0.0,
                "verification_pass_rate": 0.0,
                "context_verification_pass_rate": 0.0,
                "segmentation_verification_pass_rate": 0.0,
                "gis_verification_pass_rate": 0.0,
                "requires_user_confirmation": True,
                "confirmation_reasons": ["No candidate AOIs met retrieval criteria."],
            }

        num_cands = len(candidate_aois)
        avg_overall = sum(c.overall_score for c in candidate_aois) / num_cands

        # Multi-stage verification pass rates
        gis_passed = sum(1 for c in candidate_aois if c.verification_flags.get("gis_verified", False))
        ctx_passed = sum(1 for c in candidate_aois if c.verification_flags.get("context_verified", False))
        seg_passed = sum(1 for c in candidate_aois if c.verification_flags.get("segmentation_verified", False))

        gis_rate = gis_passed / num_cands
        ctx_rate = ctx_passed / num_cands
        seg_rate = seg_passed / num_cands

        # Combined verification pass rate across all 3 verification pillars
        composite_verif_rate = (gis_rate * 0.40) + (ctx_rate * 0.30) + (seg_rate * 0.30)

        # Overall retrieval confidence
        overall_conf = (query_conf * 0.35) + (avg_overall * 0.40) + (composite_verif_rate * 0.25)

        requires_conf = False
        reasons: List[str] = []

        # Check unconstrained spatial extent
        if not parsed_query.location_names and not parsed_query.geometry_constraints:
            requires_conf = True
            reasons.append("Unconstrained spatial scope: AOI boundary selection recommended before multi-temporal analysis.")

        # Check temporal extent when change intent requested
        if parsed_query.change_intent and not (parsed_query.start_date and parsed_query.end_date):
            requires_conf = True
            reasons.append("Temporal range unconstrained for change detection: comparison baseline needed.")

        # Check if top candidate failed verification
        if candidate_aois:
            top_c = candidate_aois[0]
            top_vf = top_c.verification_flags
            if not top_vf.get("gis_verified", False):
                requires_conf = True
                reasons.append(f"Top-ranked candidate '{top_c.aoi_id}' failed deterministic GIS corridor verification.")
            if not top_vf.get("context_verified", False):
                requires_conf = True
                reasons.append(f"Top-ranked candidate '{top_c.aoi_id}' flagged for context inconsistency.")
            if not top_vf.get("segmentation_verified", False):
                reasons.append(f"Top-ranked candidate '{top_c.aoi_id}' flagged for structural morphology mismatch.")

        return {
            "overall_confidence": round(max(0.0, min(1.0, overall_conf)), 3),
            "query_parsing_confidence": round(query_conf, 3),
            "candidate_quality_confidence": round(avg_overall, 3),
            "verification_pass_rate": round(composite_verif_rate, 3),
            "gis_verification_pass_rate": round(gis_rate, 3),
            "context_verification_pass_rate": round(ctx_rate, 3),
            "segmentation_verification_pass_rate": round(seg_rate, 3),
            "requires_user_confirmation": requires_conf,
            "confirmation_reasons": reasons,
        }
