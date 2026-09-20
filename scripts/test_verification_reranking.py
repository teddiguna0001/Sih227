#!/usr/bin/env python3
"""
CLI & API Helper for Candidate Verification and Multi-Factor Reranking.
Evaluates Context Verification, Structural/Segmentation, Deterministic GIS, and Hard-Negative Scenarios.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.context_verifier import ContextVerifier
from src.part1_retrieval.segmentation_verifier import SegmentationVerifier
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.reranker import CandidateReranker
from src.part1_retrieval.confidence import ConfidenceEngine
from src.part1_retrieval.retrieval_pipeline import RetrievalPipeline


def get_hard_negative_scenarios():
    """Generates the 5 mandatory hard-negative evaluation scenarios."""
    context_verifier = ContextVerifier()
    segmentation_verifier = SegmentationVerifier()
    gis_verifier = GISVerifier()
    reranker = CandidateReranker()

    scenarios = []

    # -------------------------------------------------------------------------
    # Scenario 1: Industrial Warehouse vs. Agricultural Greenhouse
    # -------------------------------------------------------------------------
    q1 = ParsedQuery(
        raw_query="Find large industrial buildings near highway",
        normalized_semantic_query="industrial warehouse logistics park near highway",
        semantic_targets=["large industrial buildings", "warehouses"],
        spatial_relations=[{"relation": "near", "reference": "highway", "distance_km": 2.0}],
        confidence=0.92,
    )
    c1_true = CandidateAOI(
        aoi_id="AOI-IND-01",
        name="Sriperumbudur Industrial Hub",
        bbox=(80.12, 13.00, 80.14, 13.02),
        centroid=(80.13, 13.01),
        area_sqkm=2.4,
        sensor="Cartosat-3",
        resolution_m=0.28,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="industrial_facility",
        semantic_score=0.84,
        spatial_score=0.85,
        metadata={
            "multiscale_context": {
                "small": ["large_roofs", "paved_aprons", "loading_docks"],
                "medium": ["warehouses", "logistics_yards", "industrial_estate"],
                "large": ["transport_corridor", "highway_network", "chennai_nh48_industrial_corridor"],
            },
            "structural_evidence": {
                "building_fraction": 0.35,
                "road_fraction": 0.18,
                "vegetation_fraction": 0.08,
                "mean_roof_area_sqm": 4200.0,
                "max_roof_area_sqm": 9500.0,
                "building_density_per_sqkm": 14.0,
                "semantic_class_probabilities": {"industrial": 0.88, "commercial": 0.10},
            },
        },
    )
    c1_neg = CandidateAOI(
        aoi_id="AOI-GRN-01",
        name="Rural Agricultural Polyhouse Complex",
        bbox=(79.80, 12.80, 79.82, 12.82),
        centroid=(79.81, 12.81),
        area_sqkm=1.8,
        sensor="Cartosat-3",
        resolution_m=0.28,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="industrial_facility",
        semantic_score=0.81,
        spatial_score=0.40,
        metadata={
            "multiscale_context": {
                "small": ["large_roofs", "translucent_panels"],
                "medium": ["farmland", "cropland", "greenhouse", "rural_hamlet"],
                "large": ["rural_basin", "agricultural_belt"],
            },
            "structural_evidence": {
                "building_fraction": 0.22,
                "road_fraction": 0.02,
                "vegetation_fraction": 0.65,
                "mean_roof_area_sqm": 2100.0,
                "max_roof_area_sqm": 3500.0,
                "building_density_per_sqkm": 8.0,
                "semantic_class_probabilities": {"greenhouse": 0.75, "industrial": 0.15},
            },
        },
    )
    pair1 = [c1_neg, c1_true]
    context_verifier.verify_candidates(pair1, q1)
    segmentation_verifier.verify_candidates(pair1, q1)
    gis_verifier.verify_candidates(pair1, q1)
    reranked1 = reranker.rerank(pair1, q1)

    scenarios.append({
        "id": "scenario_1",
        "title": "Industrial Building vs. Agricultural Greenhouse",
        "query": q1.raw_query,
        "true_positive": reranked1[0].to_dict(),
        "hard_negative": reranked1[1].to_dict(),
        "discrimination_margin": round(reranked1[0].overall_score - reranked1[1].overall_score, 3),
        "verifiers_fired": {
            "context_conflict": reranked1[1].metadata["context_verification"]["conflict_detected"],
            "conflict_reason": reranked1[1].metadata["context_verification"]["conflict_reason"],
            "structural_mismatch": reranked1[1].metadata["structural_verification"]["mismatch_detected"],
            "mismatch_reason": reranked1[1].metadata["structural_verification"]["mismatch_reason"],
        },
    })

    # -------------------------------------------------------------------------
    # Scenario 2: Highway vs. Airport Runway
    # -------------------------------------------------------------------------
    q2 = ParsedQuery(
        raw_query="Find highways near Chennai",
        normalized_semantic_query="highways expressway transit corridor Chennai",
        semantic_targets=["highway", "road network"],
        location_names=["Chennai"],
        confidence=0.92,
    )
    c2_true = CandidateAOI(
        aoi_id="AOI-HWY-01",
        name="GST Road Arterial Highway",
        bbox=(80.11, 13.00, 80.13, 13.02),
        centroid=(80.12, 13.01),
        area_sqkm=1.5,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="highway",
        semantic_score=0.86,
        spatial_score=0.88,
        metadata={
            "multiscale_context": {
                "small": ["paved_road_strip", "road_markings", "median_strip"],
                "medium": ["highway_corridor", "interchanges", "access_ramps"],
                "large": ["interstate_artery", "regional_connectivity_grid"],
            },
            "structural_evidence": {
                "road_fraction": 0.42,
                "building_fraction": 0.05,
                "semantic_class_probabilities": {"road_network": 0.90, "urban": 0.10},
            },
        },
    )
    c2_neg = CandidateAOI(
        aoi_id="AOI-RNW-01",
        name="Airport Main Runway Strip",
        bbox=(80.16, 12.98, 80.18, 13.00),
        centroid=(80.17, 12.99),
        area_sqkm=1.8,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="highway",
        semantic_score=0.83,
        spatial_score=0.80,
        metadata={
            "multiscale_context": {
                "small": ["paved_road_strip", "white_threshold_markings"],
                "medium": ["runway", "taxiway", "airport", "airfield", "hangar"],
                "large": ["aviation_complex", "urban_perimeter"],
            },
            "structural_evidence": {
                "road_fraction": 0.38,
                "building_fraction": 0.08,
                "semantic_class_probabilities": {"runway": 0.85, "airport": 0.92},
            },
        },
    )
    pair2 = [c2_neg, c2_true]
    context_verifier.verify_candidates(pair2, q2)
    segmentation_verifier.verify_candidates(pair2, q2)
    gis_verifier.verify_candidates(pair2, q2)
    reranked2 = reranker.rerank(pair2, q2)

    scenarios.append({
        "id": "scenario_2",
        "title": "Civil Highway vs. Airport Runway",
        "query": q2.raw_query,
        "true_positive": reranked2[0].to_dict(),
        "hard_negative": reranked2[1].to_dict(),
        "discrimination_margin": round(reranked2[0].overall_score - reranked2[1].overall_score, 3),
        "verifiers_fired": {
            "context_conflict": reranked2[1].metadata["context_verification"]["conflict_detected"],
            "conflict_reason": reranked2[1].metadata["context_verification"]["conflict_reason"],
            "structural_mismatch": reranked2[1].metadata["structural_verification"]["mismatch_detected"],
            "mismatch_reason": reranked2[1].metadata["structural_verification"]["mismatch_reason"],
        },
    })

    # -------------------------------------------------------------------------
    # Scenario 3: Water Body vs. Cloud/Topographic Shadow
    # -------------------------------------------------------------------------
    q3 = ParsedQuery(
        raw_query="Find water bodies and reservoirs",
        normalized_semantic_query="water body reservoir lake freshwater surface",
        semantic_targets=["water bodies", "reservoirs"],
        confidence=0.95,
    )
    c3_true = CandidateAOI(
        aoi_id="AOI-WTR-01",
        name="Chembarambakkam Reservoir",
        bbox=(80.00, 13.00, 80.04, 13.04),
        centroid=(80.02, 13.02),
        area_sqkm=15.2,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="water_body",
        semantic_score=0.88,
        spatial_score=0.80,
        metadata={
            "multiscale_context": {
                "small": ["specular_water_surface", "shoreline_boundary"],
                "medium": ["water_body", "wetland_buffer", "drainage_catchment"],
                "large": ["river_basin", "coastal_drainage"],
            },
            "structural_evidence": {
                "water_fraction": 0.85,
                "vegetation_fraction": 0.08,
                "spectral_indices": {"ndwi_mean": 0.58, "ndvi_mean": -0.18},
                "semantic_class_probabilities": {"water_body": 0.94},
            },
        },
    )
    c3_neg = CandidateAOI(
        aoi_id="AOI-SHD-01",
        name="Escarpment Ridge Shadow",
        bbox=(79.55, 13.30, 79.57, 13.32),
        centroid=(79.56, 13.31),
        area_sqkm=2.1,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="water_body",
        semantic_score=0.82,
        spatial_score=0.60,
        metadata={
            "multiscale_context": {
                "small": ["dark_low_reflectance_patch"],
                "medium": ["cloud_shadow", "steep_cliff_shadow", "mountain_ridge"],
                "large": ["rocky_highlands"],
            },
            "structural_evidence": {
                "water_fraction": 0.02,
                "vegetation_fraction": 0.15,
                "bare_soil_fraction": 0.80,
                "spectral_indices": {"ndwi_mean": -0.32, "ndvi_mean": 0.05},
                "semantic_class_probabilities": {"shadow": 0.88, "water_body": 0.05},
            },
        },
    )
    pair3 = [c3_neg, c3_true]
    context_verifier.verify_candidates(pair3, q3)
    segmentation_verifier.verify_candidates(pair3, q3)
    gis_verifier.verify_candidates(pair3, q3)
    reranked3 = reranker.rerank(pair3, q3)

    scenarios.append({
        "id": "scenario_3",
        "title": "Open Water Body vs. Mountain/Cloud Shadow",
        "query": q3.raw_query,
        "true_positive": reranked3[0].to_dict(),
        "hard_negative": reranked3[1].to_dict(),
        "discrimination_margin": round(reranked3[0].overall_score - reranked3[1].overall_score, 3),
        "verifiers_fired": {
            "context_conflict": reranked3[1].metadata["context_verification"]["conflict_detected"],
            "conflict_reason": reranked3[1].metadata["context_verification"]["conflict_reason"],
            "structural_mismatch": reranked3[1].metadata["structural_verification"]["mismatch_detected"],
            "mismatch_reason": reranked3[1].metadata["structural_verification"]["mismatch_reason"],
        },
    })

    # -------------------------------------------------------------------------
    # Scenario 4: Agriculture vs. Bare Land / Dry Clearing
    # -------------------------------------------------------------------------
    q4 = ParsedQuery(
        raw_query="Find agricultural cropland",
        normalized_semantic_query="agricultural cropland farming cultivation fertile fields",
        semantic_targets=["agricultural cropland", "farming"],
        confidence=0.91,
    )
    c4_true = CandidateAOI(
        aoi_id="AOI-AGR-01",
        name="Kanchipuram Paddy Fields",
        bbox=(79.68, 12.82, 79.72, 12.86),
        centroid=(79.70, 12.84),
        area_sqkm=8.4,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="cropland",
        semantic_score=0.87,
        spatial_score=0.80,
        metadata={
            "multiscale_context": {
                "small": ["crop_field_plots", "vegetation_canopy"],
                "medium": ["agricultural_farmland", "irrigation_canals"],
                "large": ["fertile_plains", "river_valley"],
            },
            "structural_evidence": {
                "vegetation_fraction": 0.72,
                "bare_soil_fraction": 0.18,
                "spectral_indices": {"ndvi_mean": 0.64, "ndwi_mean": -0.02},
                "semantic_class_probabilities": {"cropland": 0.89},
            },
        },
    )
    c4_neg = CandidateAOI(
        aoi_id="AOI-BAR-01",
        name="Dry Construction Plot Clearing",
        bbox=(79.92, 13.10, 79.94, 13.12),
        centroid=(79.93, 13.11),
        area_sqkm=4.2,
        sensor="Sentinel-2",
        resolution_m=10.0,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="cropland",
        semantic_score=0.80,
        spatial_score=0.75,
        metadata={
            "multiscale_context": {
                "small": ["cleared_earth", "open_parcel"],
                "medium": ["barren_rock", "construction_excavation", "saline_flat"],
                "large": ["arid_plateau"],
            },
            "structural_evidence": {
                "vegetation_fraction": 0.05,
                "bare_soil_fraction": 0.86,
                "spectral_indices": {"ndvi_mean": 0.11, "ndwi_mean": -0.22},
                "semantic_class_probabilities": {"barren": 0.82, "cropland": 0.08},
            },
        },
    )
    pair4 = [c4_neg, c4_true]
    context_verifier.verify_candidates(pair4, q4)
    segmentation_verifier.verify_candidates(pair4, q4)
    gis_verifier.verify_candidates(pair4, q4)
    reranked4 = reranker.rerank(pair4, q4)

    scenarios.append({
        "id": "scenario_4",
        "title": "Vegetative Agriculture vs. Bare Land / Soil",
        "query": q4.raw_query,
        "true_positive": reranked4[0].to_dict(),
        "hard_negative": reranked4[1].to_dict(),
        "discrimination_margin": round(reranked4[0].overall_score - reranked4[1].overall_score, 3),
        "verifiers_fired": {
            "context_conflict": reranked4[1].metadata["context_verification"]["conflict_detected"],
            "conflict_reason": reranked4[1].metadata["context_verification"]["conflict_reason"],
            "structural_mismatch": reranked4[1].metadata["structural_verification"]["mismatch_detected"],
            "mismatch_reason": reranked4[1].metadata["structural_verification"]["mismatch_reason"],
        },
    })

    # -------------------------------------------------------------------------
    # Scenario 5: Residential Roofs vs. Industrial Roofs
    # -------------------------------------------------------------------------
    q5 = ParsedQuery(
        raw_query="Find large industrial factories and warehouses",
        normalized_semantic_query="large industrial factory manufacturing plant warehouse facilities",
        semantic_targets=["large industrial factories", "warehouses"],
        confidence=0.92,
    )
    c5_true = CandidateAOI(
        aoi_id="AOI-FAC-01",
        name="Oragadam Automotive Manufacturing Zone",
        bbox=(79.95, 12.83, 79.98, 12.86),
        centroid=(79.965, 12.845),
        area_sqkm=4.8,
        sensor="Cartosat-3",
        resolution_m=0.28,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="industrial_facility",
        semantic_score=0.89,
        spatial_score=0.82,
        metadata={
            "multiscale_context": {
                "small": ["large_roofs", "assembly_plants", "chimneys"],
                "medium": ["warehouses", "industrial_estate", "logistics_yards"],
                "large": ["transport_corridor", "freight_artery"],
            },
            "structural_evidence": {
                "building_fraction": 0.32,
                "road_fraction": 0.16,
                "mean_roof_area_sqm": 5400.0,
                "max_roof_area_sqm": 16000.0,
                "building_density_per_sqkm": 12.0,
                "semantic_class_probabilities": {"industrial": 0.92},
            },
        },
    )
    c5_neg = CandidateAOI(
        aoi_id="AOI-RES-01",
        name="Tambaram Dense Residential Colony",
        bbox=(80.11, 12.92, 80.13, 12.94),
        centroid=(80.12, 12.93),
        area_sqkm=2.2,
        sensor="Cartosat-3",
        resolution_m=0.28,
        temporal_range=("2024-01-01", "2024-03-30"),
        primary_semantic_class="industrial_facility",
        semantic_score=0.83,
        spatial_score=0.80,
        metadata={
            "multiscale_context": {
                "small": ["small_pitched_roofs", "water_tanks", "rooftop_terraces"],
                "medium": ["dense_residential_colony", "residential_suburb", "apartments"],
                "large": ["metropolitan_urban_zone"],
            },
            "structural_evidence": {
                "building_fraction": 0.48,
                "road_fraction": 0.22,
                "mean_roof_area_sqm": 135.0,
                "max_roof_area_sqm": 240.0,
                "building_density_per_sqkm": 210.0,
                "semantic_class_probabilities": {"residential": 0.86, "industrial": 0.08},
            },
        },
    )
    pair5 = [c5_neg, c5_true]
    context_verifier.verify_candidates(pair5, q5)
    segmentation_verifier.verify_candidates(pair5, q5)
    gis_verifier.verify_candidates(pair5, q5)
    reranked5 = reranker.rerank(pair5, q5)

    scenarios.append({
        "id": "scenario_5",
        "title": "Industrial Warehouse Roofs vs. Dense Residential Roofs",
        "query": q5.raw_query,
        "true_positive": reranked5[0].to_dict(),
        "hard_negative": reranked5[1].to_dict(),
        "discrimination_margin": round(reranked5[0].overall_score - reranked5[1].overall_score, 3),
        "verifiers_fired": {
            "context_conflict": reranked5[1].metadata["context_verification"]["conflict_detected"],
            "conflict_reason": reranked5[1].metadata["context_verification"]["conflict_reason"],
            "structural_mismatch": reranked5[1].metadata["structural_verification"]["mismatch_detected"],
            "mismatch_reason": reranked5[1].metadata["structural_verification"]["mismatch_reason"],
        },
    })

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Candidate Verification and Multi-Factor Reranking Helper")
    parser.add_argument(
        "--action",
        choices=["hard_negatives", "rerank_eval", "pipeline"],
        default="hard_negatives",
    )
    parser.add_argument("--query", type=str, default="industrial area near highway in Chennai")
    parser.add_argument("--candidates", type=int, default=5)

    args = parser.parse_args()

    if args.action == "hard_negatives":
        scenarios = get_hard_negative_scenarios()
        print(json.dumps({"status": "success", "scenarios": scenarios}, indent=2))
    elif args.action == "pipeline":
        pipeline = RetrievalPipeline()
        res = pipeline.run(args.query, max_candidates=args.candidates)
        print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
