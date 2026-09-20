import React from "react";
import { ParsedQueryData } from "../types";
import {
  AlertTriangle,
  Calendar,
  CheckCircle,
  Clock,
  Compass,
  Layers,
  MapPin,
  Maximize2,
  RefreshCw,
  Sliders,
  Target,
} from "lucide-react";

interface ParsedSchemaViewerProps {
  parsed: ParsedQueryData;
}

export const ParsedSchemaViewer: React.FC<ParsedSchemaViewerProps> = ({ parsed }) => {
  const getChangeBadgeColor = (changeType: string | null) => {
    switch (changeType) {
      case "ROAD_DEVELOPMENT":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "CONSTRUCTION_CHANGE":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "CLEARANCE":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "WATER_EXPANSION":
        return "bg-cyan-100 text-cyan-800 border-cyan-200";
      case "WATER_CONTRACTION":
        return "bg-rose-100 text-rose-800 border-rose-200";
      case "VEGETATION_LOSS":
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
      case "URBAN_EXPANSION":
        return "bg-indigo-100 text-indigo-800 border-indigo-200";
      case "GENERIC_CHANGE":
        return "bg-purple-100 text-purple-800 border-purple-200";
      case "STATIC_SEMANTIC_SEARCH":
        return "bg-slate-100 text-slate-800 border-slate-200";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-slate-700" />
            ParsedQuery Contract Schema
          </h2>
          <p className="text-xs text-slate-500">
            Validated against SIH26227 specifications &bull; Zero constraint fabrication
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 text-xs font-mono">
            <span className="text-slate-500">Confidence:</span>
            <span className="font-bold text-slate-900">{(parsed.confidence * 100).toFixed(1)}%</span>
          </div>

          <div
            className={`px-2.5 py-1 rounded-md text-xs font-semibold border ${getChangeBadgeColor(
              parsed.requested_change_type
            )}`}
          >
            {parsed.requested_change_type || "UNKNOWN"}
          </div>
        </div>
      </div>

      {/* Grid of Parsed Attributes */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Semantic Targets */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1.5">
            <Target className="w-3.5 h-3.5 text-blue-600" />
            Semantic Targets
          </div>
          {parsed.semantic_targets.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {parsed.semantic_targets.map((t) => (
                <span
                  key={t}
                  className="px-2 py-0.5 text-xs font-medium rounded-sm bg-blue-50 text-blue-700 border border-blue-200"
                >
                  {t}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-xs text-slate-400 italic">None specified</span>
          )}
        </div>

        {/* Location & Geometry */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1.5">
            <MapPin className="w-3.5 h-3.5 text-rose-600" />
            Locations & Gazetteer
          </div>
          {parsed.location_names.length > 0 ? (
            <div>
              <div className="flex flex-wrap gap-1 mb-1">
                {parsed.location_names.map((l) => (
                  <span
                    key={l}
                    className="px-2 py-0.5 text-xs font-medium rounded-sm bg-rose-50 text-rose-700 border border-rose-200"
                  >
                    {l}
                  </span>
                ))}
              </div>
              {parsed.geometry_constraints && (
                <div className="text-[11px] text-slate-500 font-mono">
                  BBox: [{parsed.geometry_constraints.bbox.min_lon}, {parsed.geometry_constraints.bbox.min_lat}, ... ]
                </div>
              )}
            </div>
          ) : (
            <span className="text-xs text-slate-400 italic">None specified (null)</span>
          )}
        </div>

        {/* Temporal Constraints */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1.5">
            <Calendar className="w-3.5 h-3.5 text-emerald-600" />
            Temporal Bounds
          </div>
          {parsed.start_date || parsed.end_date ? (
            <div className="text-xs font-mono font-medium text-slate-800">
              {parsed.start_date || "Any"} &rarr; {parsed.end_date || "Present"}
            </div>
          ) : (
            <span className="text-xs text-slate-400 italic">None specified (null)</span>
          )}
        </div>

        {/* Spatial Relations */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1.5">
            <Compass className="w-3.5 h-3.5 text-purple-600" />
            Spatial Relations
          </div>
          {parsed.spatial_relations.length > 0 ? (
            <div className="space-y-1">
              {parsed.spatial_relations.map((sr, idx) => (
                <div key={idx} className="text-xs font-mono text-purple-900 bg-purple-50 px-1.5 py-0.5 rounded border border-purple-200">
                  {sr.relation} {sr.distance_km ? `${sr.distance_km} km of ` : ""}
                  <span className="font-semibold">{sr.reference}</span>
                </div>
              ))}
            </div>
          ) : (
            <span className="text-xs text-slate-400 italic">None specified (null)</span>
          )}
        </div>
      </div>

      {/* Secondary Specs: Sensors, Resolution, State Transitions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs">
        <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
          <span className="font-medium text-slate-500">Sensor Constraints:</span>{" "}
          {parsed.sensor_constraints.length > 0 ? (
            <span className="font-semibold text-slate-800">{parsed.sensor_constraints.join(", ")}</span>
          ) : (
            <span className="text-slate-400 italic">None (null)</span>
          )}
        </div>

        <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
          <span className="font-medium text-slate-500">Resolution Constraints:</span>{" "}
          {parsed.resolution_constraints ? (
            <span className="font-semibold text-slate-800">
              GSD &le; {parsed.resolution_constraints.max_resolution_m}m ({parsed.resolution_constraints.category})
            </span>
          ) : (
            <span className="text-slate-400 italic">None (null)</span>
          )}
        </div>

        <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
          <span className="font-medium text-slate-500">Change Intent:</span>{" "}
          <span
            className={`font-semibold ${
              parsed.change_intent ? "text-amber-700" : "text-slate-700"
            }`}
          >
            {parsed.change_intent ? "True (Multi-Temporal Analysis)" : "False (Static Semantic Search)"}
          </span>
        </div>
      </div>

      {/* Warnings & Rule 7 Verification Checks */}
      {parsed.warnings.length > 0 && (
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <div className="text-xs font-semibold text-amber-900 flex items-center gap-1.5 mb-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            Quality Assurance Warnings & Requires Confirmation (Rule 7 Validation):
          </div>
          <ul className="space-y-1">
            {parsed.warnings.map((w, idx) => (
              <li key={idx} className="text-xs text-amber-800 flex items-start gap-1.5">
                <span className="text-amber-500 font-bold">&bull;</span>
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
