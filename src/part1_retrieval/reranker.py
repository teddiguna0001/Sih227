"""
Multi-Factor Candidate AOI Reranker Module.

Combines semantic similarity, multi-scale context consistency, structural segmentation,
deterministic GIS compliance, data quality, metadata match, resolution suitability,
and sensor suitability using configurable weights from configs/reranking.yaml.

Result is called 'relevance_score' / 'ranking_score' / 'retrieval_score' (NEVER probability).

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import os
from typing import List, Dict, Optional, Any
from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery

try:
    import yaml
except ImportError:
    yaml = None  # Graceful fallback if PyYAML is absent


class CandidateReranker:
    """
    Explainable Multi-Factor Reranking Engine.
    
    Fuses:
      1. semantic_similarity   (Vision-Language hypersphere similarity)
      2. context_consistency   (Multi-scale: small -> medium -> large consistency)
      3. structural_segmentation (Building footprints, roof areas, NDWI, NDVI)
      4. gis_deterministic    (Exact PostGIS topological & metric corridor distance)
      5. quality_metric        (Cloud cover %, valid pixels, usability flag)
      6. metadata_match        (Temporal alignment, acquisition metadata)
      7. resolution_suitability (Sensor GSD matched to target feature scale)
      8. sensor_suitability    (Optical vs SAR sensor mission suitability)
    """

    DEFAULT_CONFIG_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../configs/reranking.yaml")
    )

    DEFAULT_WEIGHTS = {
        "semantic_similarity": 0.25,
        "context_consistency": 0.20,
        "structural_segmentation": 0.20,
        "gis_deterministic": 0.15,
        "quality_metric": 0.08,
        "metadata_match": 0.04,
        "resolution_suitability": 0.04,
        "sensor_suitability": 0.04,
    }

    DEFAULT_PENALTIES = {
        "gis_constraint_violated": 0.35,
        "context_conflict_detected": 0.30,
        "structural_mismatch_detected": 0.25,
        "cloud_cover_exceeded": 0.25,
        "resolution_unsuitable": 0.20,
    }

    DEFAULT_BONUSES = {
        "all_verifiers_passed": 0.06,
        "high_resolution_boost": 0.03,
        "temporal_exact_match": 0.02,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        penalties: Optional[Dict[str, float]] = None,
        bonuses: Optional[Dict[str, float]] = None,
    ):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()

        # Configurable weights (normalized to sum to 1.0)
        raw_weights = weights or self.config.get("weights") or self.DEFAULT_WEIGHTS
        total_w = sum(raw_weights.values()) or 1.0
        self.weights = {k: v / total_w for k, v in raw_weights.items()}

        self.penalties = penalties or self.config.get("penalties") or self.DEFAULT_PENALTIES
        self.bonuses = bonuses or self.config.get("bonuses") or self.DEFAULT_BONUSES
        self.thresholds = self.config.get("thresholds", {
            "gis_pass_threshold": 0.50,
            "context_pass_threshold": 0.50,
            "structural_pass_threshold": 0.50,
        })
        self.output_cfg = self.config.get("output", {
            "score_field_name": "relevance_score",
            "sort_order": "descending",
        })

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file or falls back to defaults."""
        if yaml and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {
            "weights": self.DEFAULT_WEIGHTS,
            "penalties": self.DEFAULT_PENALTIES,
            "bonuses": self.DEFAULT_BONUSES,
        }

    # =========================================================================
    # FACTOR EVALUATION HELPERS
    # =========================================================================

    def compute_quality_score(self, candidate: CandidateAOI) -> float:
        """Evaluates image data quality: cloud cover, valid pixels, and usability."""
        meta = candidate.metadata or {}
        quality = meta.get("quality", {})

        # Default good quality if unstated
        cloud_pct = float(quality.get("cloud_cover_percent", 5.0))
        valid_pct = float(quality.get("valid_pixel_pct", 98.0))
        usable = quality.get("usable_data_flag", True)

        cloud_score = max(0.0, 1.0 - (cloud_pct / 100.0))
        valid_score = max(0.0, min(1.0, valid_pct / 100.0))
        usability_score = 1.0 if usable else 0.2

        return round(0.5 * cloud_score + 0.3 * valid_score + 0.2 * usability_score, 3)

    def compute_metadata_match(self, candidate: CandidateAOI, parsed_query: ParsedQuery) -> float:
        """Evaluates temporal alignment and target metadata correspondence."""
        score = candidate.temporal_score if candidate.temporal_score > 0 else 0.85
        if parsed_query.start_date and parsed_query.end_date:
            # Check overlap with candidate temporal range
            c_start, c_end = candidate.temporal_range
            if c_start and c_end:
                if c_start <= parsed_query.end_date and c_end >= parsed_query.start_date:
                    score = 1.0
                else:
                    score = 0.40
        return round(score, 3)

    def compute_resolution_suitability(self, candidate: CandidateAOI, parsed_query: ParsedQuery) -> float:
        """Evaluates whether the sensor's GSD is suitable for detecting the target feature scale."""
        gsd = candidate.resolution_m
        targets = [t.lower() for t in (parsed_query.semantic_targets or [])]
        raw_lower = (parsed_query.raw_query or "").lower()

        is_building = any(k in t for t in targets for k in ["building", "roof", "warehouse", "factory"]) or ("industrial" in raw_lower)
        is_linear = any(k in t for t in targets for k in ["highway", "road", "canal", "runway"])
        is_regional = any(k in t for t in targets for k in ["agriculture", "cropland", "forest", "desert", "water", "reservoir"])

        if is_building:
            # Buildings benefit from high resolution (Cartosat-3 0.28m, WorldView <= 1m)
            if gsd <= 1.0:
                return 1.0
            elif gsd <= 10.0:
                return 0.75  # Sentinel-2 (10m) can see large warehouse complexes
            else:
                return 0.35  # Landsat-8 (30m) cannot resolve individual buildings
        elif is_linear:
            if gsd <= 10.0:
                return 1.0
            elif gsd <= 30.0:
                return 0.70
            else:
                return 0.40
        elif is_regional:
            # Regional features are well suited for 10m-30m multispectral sensors
            if gsd <= 30.0:
                return 1.0
            else:
                return 0.80
        return 0.80

    def compute_sensor_suitability(self, candidate: CandidateAOI, parsed_query: ParsedQuery) -> float:
        """Evaluates optical vs SAR sensor mission suitability."""
        sensor = (candidate.sensor or "").lower()
        targets = [t.lower() for t in (parsed_query.semantic_targets or [])]
        raw_lower = (parsed_query.raw_query or "").lower()

        # If user explicitly requested radar / SAR / all-weather
        requested_sensors = [s.lower() for s in (parsed_query.sensor_constraints or [])]
        if requested_sensors:
            if any(s in sensor for s in requested_sensors):
                return 1.0
            return 0.50

        # Flood, coastal water, or all-weather search -> SAR (Sentinel-1) excels
        if any(k in raw_lower for k in ["flood", "monsoon", "all-weather", "radar", "sar"]):
            return 1.0 if "sentinel-1" in sensor or "sar" in sensor else 0.65

        # Detailed rooftop or spectral vegetation analysis -> High-res Optical / Sentinel-2 excels
        if "cartosat" in sensor or "sentinel-2" in sensor or "landsat" in sensor:
            return 1.0
        return 0.80

    # =========================================================================
    # CORE MULTI-FACTOR RERANKING
    # =========================================================================

    def rerank(
        self,
        candidates: List[CandidateAOI],
        parsed_query: ParsedQuery,
    ) -> List[CandidateAOI]:
        """
        Calculates multi-factor relevance_score for each candidate, applies penalties
        and bonuses, generates an explainable audit trail, and sorts candidates descending.
        """
        for cand in candidates:
            meta = cand.metadata or {}

            # 1. Gather component factors
            semantic_score = max(0.0, min(1.0, cand.semantic_score))
            
            ctx_meta = meta.get("context_verification", {})
            context_score = max(0.0, min(1.0, ctx_meta.get("context_score", 0.50)))
            
            struct_meta = meta.get("structural_verification", {})
            structural_score = max(0.0, min(1.0, struct_meta.get("structural_score", 0.50)))

            gis_meta = meta.get("gis_verification", {})
            gis_score = max(0.0, min(1.0, gis_meta.get("gis_score", cand.spatial_score)))

            quality_score = self.compute_quality_score(cand)
            metadata_score = self.compute_metadata_match(cand, parsed_query)
            resolution_score = self.compute_resolution_suitability(cand, parsed_query)
            sensor_score = self.compute_sensor_suitability(cand, parsed_query)

            # 2. Weighted component fusion
            factor_values = {
                "semantic_similarity": semantic_score,
                "context_consistency": context_score,
                "structural_segmentation": structural_score,
                "gis_deterministic": gis_score,
                "quality_metric": quality_score,
                "metadata_match": metadata_score,
                "resolution_suitability": resolution_score,
                "sensor_suitability": sensor_score,
            }

            composite_score = sum(
                factor_values[k] * self.weights.get(k, 0.0)
                for k in factor_values
            )

            # 3. Apply Penalties & Bonuses with explainability
            penalties_applied: List[Dict[str, Any]] = []
            bonuses_applied: List[Dict[str, Any]] = []
            score_delta = 0.0

            # Penalties:
            # - GIS constraint failure
            if not cand.verification_flags.get("gis_verified", True):
                p_val = self.penalties.get("gis_constraint_violated", 0.35)
                score_delta -= p_val
                penalties_applied.append({
                    "factor": "gis_constraint_violated",
                    "delta": -p_val,
                    "reason": "Failed deterministic GIS spatial relation or bounding geometry predicate.",
                })

            # - Context conflict
            if ctx_meta.get("conflict_detected", False):
                p_val = self.penalties.get("context_conflict_detected", 0.30)
                score_delta -= p_val
                penalties_applied.append({
                    "factor": "context_conflict_detected",
                    "delta": -p_val,
                    "reason": ctx_meta.get("conflict_reason", "Multi-scale context conflict detected."),
                })

            # - Structural mismatch
            if struct_meta.get("mismatch_detected", False):
                p_val = self.penalties.get("structural_mismatch_detected", 0.25)
                score_delta -= p_val
                penalties_applied.append({
                    "factor": "structural_mismatch_detected",
                    "delta": -p_val,
                    "reason": struct_meta.get("mismatch_reason", "Structural morphological mismatch detected."),
                })

            # - Cloud cover non-compliance
            if not cand.verification_flags.get("cloud_cover_compliant", True):
                p_val = self.penalties.get("cloud_cover_exceeded", 0.25)
                score_delta -= p_val
                penalties_applied.append({
                    "factor": "cloud_cover_exceeded",
                    "delta": -p_val,
                    "reason": "Cloud cover exceeds query tolerance threshold.",
                })

            # - Resolution non-compliance
            if not cand.verification_flags.get("resolution_compliant", True):
                p_val = self.penalties.get("resolution_unsuitable", 0.20)
                score_delta -= p_val
                penalties_applied.append({
                    "factor": "resolution_unsuitable",
                    "delta": -p_val,
                    "reason": "Ground resolution is insufficient for requested feature recognition.",
                })

            # Bonuses:
            # - Passed all three verifiers (GIS, Context, Structural)
            all_verified = (
                cand.verification_flags.get("gis_verified", False)
                and cand.verification_flags.get("context_verified", False)
                and cand.verification_flags.get("segmentation_verified", False)
            )
            if all_verified:
                b_val = self.bonuses.get("all_verifiers_passed", 0.06)
                score_delta += b_val
                bonuses_applied.append({
                    "factor": "all_verifiers_passed",
                    "delta": b_val,
                    "reason": "Candidate validated across GIS, multi-scale context, and structural verification stages.",
                })

            # - High resolution bonus
            if cand.resolution_m <= 1.0:
                b_val = self.bonuses.get("high_resolution_boost", 0.03)
                score_delta += b_val
                bonuses_applied.append({
                    "factor": "high_resolution_boost",
                    "delta": b_val,
                    "reason": f"High resolution imagery asset ({cand.resolution_m}m GSD).",
                })

            # 4. Final Relevance Score Calculation
            # Strictly referred to as relevance_score / ranking_score / retrieval_score (NOT probability)
            final_relevance = round(max(0.0, min(1.0, composite_score + score_delta)), 3)
            cand.overall_score = final_relevance

            # 5. Build Explainable Reranking Report
            explanation_summary = self._build_explanation_summary(
                cand.aoi_id,
                final_relevance,
                factor_values,
                penalties_applied,
                bonuses_applied,
            )

            cand.metadata["relevance_score"] = final_relevance
            cand.metadata["ranking_score"] = final_relevance
            cand.metadata["retrieval_score"] = final_relevance
            cand.metadata["ranking_explanation"] = {
                "metric_name": "relevance_score",
                "notice": "Relevance score represents multi-factor retrieval ranking, NOT a calibrated probability.",
                "relevance_score": final_relevance,
                "composite_pre_adjustments": round(composite_score, 3),
                "adjustments_delta": round(score_delta, 3),
                "factor_breakdown": {
                    k: {
                        "value": round(factor_values[k], 3),
                        "weight": round(self.weights[k], 3),
                        "weighted_contribution": round(factor_values[k] * self.weights[k], 3),
                    }
                    for k in factor_values
                },
                "penalties_applied": penalties_applied,
                "bonuses_applied": bonuses_applied,
                "summary": explanation_summary,
            }

        # Sort descending by relevance_score
        candidates.sort(key=lambda c: c.overall_score, reverse=True)
        return candidates

    def _build_explanation_summary(
        self,
        aoi_id: str,
        relevance_score: float,
        factors: Dict[str, float],
        penalties: List[Dict[str, Any]],
        bonuses: List[Dict[str, Any]],
    ) -> str:
        """Constructs human-readable rationale for reranking position."""
        parts = [f"AOI '{aoi_id}' achieved relevance_score={relevance_score:.3f}."]
        
        # Highlight top positive contributors
        strong_factors = [
            f"{k} ({v:.2f})"
            for k, v in factors.items()
            if v >= 0.70
        ]
        if strong_factors:
            parts.append(f"Strengths: {', '.join(strong_factors)}.")

        # Highlight penalties
        if penalties:
            reasons = [p["reason"] for p in penalties]
            parts.append(f"Penalties applied: {'; '.join(reasons)}.")

        # Highlight bonuses
        if bonuses:
            b_reasons = [b["reason"] for b in bonuses]
            parts.append(f"Bonuses awarded: {'; '.join(b_reasons)}.")

        return " ".join(parts)
