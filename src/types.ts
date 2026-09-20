export interface ParsedQueryData {
  raw_query: string;
  normalized_semantic_query: string;
  semantic_targets: string[];
  initial_state: string | null;
  final_state: string | null;
  location_names: string[];
  geometry_constraints: {
    type: string;
    canonical_name: string;
    centroid: { lon: number; lat: number };
    bbox: { min_lon: number; min_lat: number; max_lon: number; max_lat: number };
    default_buffer_km: number;
  } | null;
  spatial_relations: Array<{
    relation: string;
    target: string;
    reference: string;
    distance_km?: number;
    buffer_km?: number;
  }>;
  start_date: string | null;
  end_date: string | null;
  sensor_constraints: string[];
  resolution_constraints: {
    max_resolution_m?: number;
    category?: string;
  } | null;
  change_intent: boolean;
  requested_change_type: string | null;
  confidence: number;
  warnings: string[];
}

export interface CandidateAOIData {
  aoi_id: string;
  name: string;
  bbox: [number, number, number, number];
  centroid: [number, number];
  area_sqkm: number;
  sensor: string;
  resolution_m: number;
  temporal_range: [string, string];
  primary_semantic_class: string;
  detected_change_type: string | null;
  semantic_score: number;
  spatial_score: number;
  temporal_score: number;
  overall_score: number;
  verification_flags: {
    gis_verified: boolean;
    context_verified: boolean;
    segmentation_verified: boolean;
    resolution_compliant: boolean;
    cloud_cover_compliant: boolean;
  };
  metadata: Record<string, any>;
}

export interface RetrievalResultData {
  parsed_query: ParsedQueryData;
  candidate_aois: CandidateAOIData[];
  confidence_assessment: {
    overall_confidence: number;
    query_parsing_confidence: number;
    candidate_quality_confidence: number;
    verification_pass_rate: number;
    requires_user_confirmation: boolean;
    confirmation_reasons?: string[];
  };
  execution_trace: string[];
}
