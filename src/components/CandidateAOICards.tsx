import React, { useState } from "react";
import { CandidateAOIData } from "../types";
import {
  Award,
  MapPin,
  Radio,
  Calendar,
  Layers,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  ShieldCheck,
} from "lucide-react";

interface CandidateAOICardsProps {
  candidates: CandidateAOIData[];
  trace: string[];
}

export const CandidateAOICards: React.FC<CandidateAOICardsProps> = ({ candidates, trace }) => {
  const [expandedAoi, setExpandedAoi] = useState<string | null>(null);

  const toggleExpand = (id: string) => {
    setExpandedAoi((prev) => (prev === id ? null : id));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
            <Award className="w-4 h-4 text-emerald-600" />
            Retrieved &amp; Reranked Candidate Areas of Interest (AOIs)
          </h2>
          <p className="text-xs text-slate-500">
            Fused ranking via multi-scale context consistency, structural morphology, and deterministic GIS metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            Candidates: {candidates.length}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {candidates.map((aoi, index) => {
          const v = aoi.verification_flags;
          const meta = aoi.metadata || {};
          const explanation = meta.ranking_explanation;
          const ctxVerif = meta.context_verification;
          const structVerif = meta.structural_verification;
          const gisVerif = meta.gis_verification;
          const isExpanded = expandedAoi === aoi.aoi_id;

          // Deterministic GIS distance summary
          const gisPredicate = gisVerif?.predicates?.[0];

          return (
            <div
              key={aoi.aoi_id}
              className={`bg-white border rounded-xl p-4 shadow-xs flex flex-col justify-between transition-all ${
                isExpanded ? "border-amber-400 ring-1 ring-amber-400/20" : "border-slate-200 hover:border-slate-300"
              }`}
            >
              <div>
                {/* Header: Rank + Relevance Score */}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-bold text-slate-900 px-2 py-0.5 bg-slate-100 rounded">
                    Rank #{index + 1} &bull; {aoi.aoi_id}
                  </span>
                  <div className="flex items-center gap-1.5" title="Relevance ranking score (not a probability)">
                    <span className="text-[11px] text-slate-400 font-mono">relevance_score:</span>
                    <span
                      className={`text-xs font-bold font-mono px-2 py-0.5 rounded border ${
                        aoi.overall_score >= 0.70
                          ? "text-emerald-700 bg-emerald-50 border-emerald-200"
                          : aoi.overall_score >= 0.40
                          ? "text-amber-700 bg-amber-50 border-amber-200"
                          : "text-rose-700 bg-rose-50 border-rose-200"
                      }`}
                    >
                      {aoi.overall_score.toFixed(3)}
                    </span>
                  </div>
                </div>

                <h3 className="text-sm font-bold text-slate-900 leading-snug mb-1">
                  {aoi.name}
                </h3>

                {/* Spatial / Sensor Metadata */}
                <div className="space-y-1.5 text-xs text-slate-600 mb-3 pt-2 border-t border-slate-100">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1 text-slate-500">
                      <MapPin className="w-3 h-3 text-rose-500" /> Centroid:
                    </span>
                    <span className="font-mono text-slate-800">
                      [{aoi.centroid[0].toFixed(4)}, {aoi.centroid[1].toFixed(4)}]
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1 text-slate-500">
                      <Radio className="w-3 h-3 text-blue-500" /> Sensor / GSD:
                    </span>
                    <span className="font-medium text-slate-800">
                      {aoi.sensor} ({aoi.resolution_m}m)
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1 text-slate-500">
                      <Calendar className="w-3 h-3 text-amber-500" /> Acquisition:
                    </span>
                    <span className="font-mono text-slate-700 text-[11px]">
                      {aoi.temporal_range[0]} to {aoi.temporal_range[1]}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Footprint Area:</span>
                    <span className="font-medium text-slate-800">{aoi.area_sqkm} km&sup2;</span>
                  </div>
                </div>

                {/* Verification Flags Section */}
                <div className="pt-2 border-t border-slate-100">
                  <div className="text-[11px] font-semibold text-slate-500 mb-1.5 flex items-center justify-between">
                    <span>Verification Pillars:</span>
                    {gisPredicate && (
                      <span className="text-[10px] font-mono text-slate-400">
                        dist: {gisPredicate.measured_dist_m}m
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-1.5 text-[11px]">
                    <div
                      className={`px-1.5 py-1 rounded text-center font-medium flex items-center justify-center gap-1 border ${
                        v.context_verified
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : "bg-rose-50 text-rose-700 border-rose-200"
                      }`}
                      title={ctxVerif?.conflict_reason || "Multi-scale context status"}
                    >
                      {v.context_verified ? <CheckCircle2 className="w-3 h-3 text-emerald-600" /> : <AlertTriangle className="w-3 h-3 text-rose-600" />}
                      Context
                    </div>

                    <div
                      className={`px-1.5 py-1 rounded text-center font-medium flex items-center justify-center gap-1 border ${
                        v.segmentation_verified
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : "bg-rose-50 text-rose-700 border-rose-200"
                      }`}
                      title={structVerif?.mismatch_reason || "Morphological structure status"}
                    >
                      {v.segmentation_verified ? <CheckCircle2 className="w-3 h-3 text-emerald-600" /> : <XCircle className="w-3 h-3 text-rose-600" />}
                      Structural
                    </div>

                    <div
                      className={`px-1.5 py-1 rounded text-center font-medium flex items-center justify-center gap-1 border ${
                        v.gis_verified
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : "bg-rose-50 text-rose-700 border-rose-200"
                      }`}
                      title="Deterministic GIS metric corridor compliance"
                    >
                      {v.gis_verified ? <CheckCircle2 className="w-3 h-3 text-emerald-600" /> : <XCircle className="w-3 h-3 text-rose-600" />}
                      GIS Metric
                    </div>
                  </div>
                </div>

                {/* Conflict / Mismatch warning badges if any */}
                {ctxVerif?.conflict_detected && (
                  <div className="mt-2 text-[11px] bg-rose-50 border border-rose-200 text-rose-800 p-2 rounded-lg flex items-start gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-600 shrink-0 mt-0.5" />
                    <span>{ctxVerif.conflict_reason}</span>
                  </div>
                )}
                {structVerif?.mismatch_detected && !ctxVerif?.conflict_detected && (
                  <div className="mt-2 text-[11px] bg-rose-50 border border-rose-200 text-rose-800 p-2 rounded-lg flex items-start gap-1.5">
                    <XCircle className="w-3.5 h-3.5 text-rose-600 shrink-0 mt-0.5" />
                    <span>{structVerif.mismatch_reason}</span>
                  </div>
                )}

                {/* Expandable Explainable Reranking Drawer */}
                {isExpanded && explanation && (
                  <div className="mt-3 pt-3 border-t border-slate-200 text-xs space-y-2.5 bg-slate-50 -mx-4 -mb-4 p-4 rounded-b-xl">
                    <div className="flex items-center gap-1.5 text-slate-800 font-semibold text-xs">
                      <ShieldCheck className="w-4 h-4 text-emerald-600" />
                      Explainable Factor Breakdown (Sum to 1.0)
                    </div>

                    {/* Factor table */}
                    <div className="space-y-1 bg-white p-2.5 rounded-lg border border-slate-200 text-[11px] font-mono">
                      {Object.entries(explanation.factor_breakdown || {}).map(([factor, item]: [string, any]) => (
                        <div key={factor} className="flex items-center justify-between text-slate-600">
                          <span className="truncate max-w-[140px] text-slate-500">{factor}:</span>
                          <span className="text-slate-800">
                            {item.value.toFixed(2)} &times; {item.weight.toFixed(2)} ={" "}
                            <strong className="text-slate-900">+{item.weighted_contribution.toFixed(3)}</strong>
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Multi-Scale Observations */}
                    {ctxVerif?.scale_observations && (
                      <div className="space-y-1 text-[11px]">
                        <span className="font-semibold text-slate-700">Multi-Scale Context Hierarchy:</span>
                        <div className="bg-white p-2 rounded-lg border border-slate-200 space-y-1 text-[10px] font-mono">
                          <div>
                            <strong className="text-blue-700">Small (Local):</strong> {ctxVerif.scale_observations.small_scale?.join(", ") || "None"}
                          </div>
                          <div>
                            <strong className="text-amber-700">Medium (Neighborhood):</strong> {ctxVerif.scale_observations.medium_scale?.join(", ") || "None"}
                          </div>
                          <div>
                            <strong className="text-purple-700">Large (Corridor):</strong> {ctxVerif.scale_observations.large_scale?.join(", ") || "None"}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Structural Metrics */}
                    {structVerif?.metrics && structVerif.metrics.mean_roof_area_sqm > 0 && (
                      <div className="space-y-1 text-[11px]">
                        <span className="font-semibold text-slate-700">Structural Morphology Evidence:</span>
                        <div className="bg-white p-2 rounded-lg border border-slate-200 grid grid-cols-2 gap-1 text-[10px] font-mono">
                          <div>Mean Roof: {structVerif.metrics.mean_roof_area_sqm} m&sup2;</div>
                          <div>Max Roof: {structVerif.metrics.max_roof_area_sqm} m&sup2;</div>
                          <div>Bldg Density: {structVerif.metrics.building_density_per_sqkm}/km&sup2;</div>
                          <div>Bldg Fraction: {(structVerif.metrics.building_fraction * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    )}

                    {/* Reranker Explanation Summary */}
                    <div className="text-[11px] text-slate-600 italic bg-amber-50/70 border border-amber-200/80 p-2 rounded-lg">
                      {explanation.summary}
                    </div>
                  </div>
                )}
              </div>

              {/* Expand/Collapse Toggle Button */}
              <button
                onClick={() => toggleExpand(aoi.aoi_id)}
                className="mt-3 w-full py-1.5 px-2.5 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg flex items-center justify-center gap-1 transition-colors cursor-pointer"
              >
                {isExpanded ? (
                  <>
                    <ChevronUp className="w-3.5 h-3.5" /> Hide Explainable Breakdown
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-3.5 h-3.5" /> Explain Reranking &amp; Multi-Scale Context
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Execution Trace Collapse */}
      {trace.length > 0 && (
        <details className="bg-slate-900 text-slate-300 rounded-xl p-4 text-xs font-mono">
          <summary className="cursor-pointer font-bold text-amber-400 hover:text-amber-300">
            View Internal Retrieval Pipeline Execution Trace ({trace.length} stages)
          </summary>
          <div className="mt-2.5 space-y-1 pl-2 border-l border-slate-700 text-slate-400">
            {trace.map((step, idx) => (
              <div key={idx} className="py-0.5">
                <span className="text-emerald-400 font-bold">&gt;&gt;</span> {step}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
};
