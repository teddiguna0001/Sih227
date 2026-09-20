import React from "react";
import { FolderTree, CheckCircle2, Shield, Satellite, Database, Code } from "lucide-react";

export const ArchitectureOverview: React.FC = () => {
  const fileNodes = [
    { path: "src/part1_retrieval/query_schema.py", desc: "ParsedQuery dataclass (14 fields), ChangeType enum & validation rules" },
    { path: "src/part1_retrieval/query_parser.py", desc: "Deterministic regex & rule parser with optional LLM hook & Rule 7 guardrails" },
    { path: "src/part1_retrieval/change_router.py", desc: "11-class multi-temporal change routing engine & priority dispatch" },
    { path: "src/part1_retrieval/gazetteer.py", desc: "Strategic geospatial gazetteer with EPSG:4326 coords & buffer radii" },
    { path: "src/part1_retrieval/metadata_filter.py", desc: "Temporal, bounding box, sensor, cloud cover (<20%) & GSD filtering" },
    { path: "src/part1_retrieval/vector_search.py", desc: "Vector search foundation interface ready for RemoteCLIP / GeoRSCLIP / Clay" },
    { path: "src/part1_retrieval/candidate_generator.py", desc: "Candidate AOI generator synthesizing sector patches & initial scoring" },
    { path: "src/part1_retrieval/gis_verifier.py", desc: "Haversine distance, corridor buffer & topological verification" },
    { path: "src/part1_retrieval/context_verifier.py", desc: "Land-cover co-occurrence and regional contextual plausibility verification" },
    { path: "src/part1_retrieval/segmentation_verifier.py", desc: "Pixel-level NDWI/NDVI & building footprint mask verification interface" },
    { path: "src/part1_retrieval/reranker.py", desc: "Multi-criteria fusion reranker (semantic, spatial, temporal & bonuses/penalties)" },
    { path: "src/part1_retrieval/confidence.py", desc: "System-level confidence engine & requires_confirmation flagging" },
    { path: "src/part1_retrieval/retrieval_pipeline.py", desc: "End-to-end Part 1 retrieval orchestrator coordinating all modules" },
    { path: "src/contracts/candidate_aoi.py", desc: "CandidateAOI contract dataclass with scores & verification flags" },
    { path: "configs/*.yaml", desc: "Declarative configs (query_parser, retrieval, reranking, gazetteer)" },
    { path: "scripts/test_query.py", desc: "Benchmark verification runner for the 5 mandated queries & CLI parser" },
    { path: "scripts/run_part1.py", desc: "Standalone CLI executable running full Part 1 retrieval pipeline" },
    { path: "tests/test_query_parser.py", desc: "Comprehensive 10-case unittest suite validating all parser rules" },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2.5">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
            <FolderTree className="w-4 h-4 text-slate-700" />
            Module Architecture & Implemented File Structure
          </h2>
          <p className="text-xs text-slate-500">
            Modular, typed, documented, and runnable independently per SIH26227 specification
          </p>
        </div>
        <span className="text-xs px-2.5 py-1 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold flex items-center gap-1">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> All 18 Modules Created
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
        {fileNodes.map((fn) => (
          <div
            key={fn.path}
            className="p-2 bg-slate-50 border border-slate-200 rounded-md flex items-start gap-2 hover:bg-slate-100 transition-colors"
          >
            <Code className="w-3.5 h-3.5 text-slate-500 mt-0.5 shrink-0" />
            <div>
              <div className="font-mono font-bold text-slate-900">{fn.path}</div>
              <div className="text-slate-600 text-[11px]">{fn.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
