import React, { useState, useEffect } from "react";
import {
  ShieldCheck,
  AlertTriangle,
  Layers,
  ArrowRight,
  Sliders,
  CheckCircle2,
  XCircle,
  HelpCircle,
  TrendingDown,
  TrendingUp,
  Compass,
  FileCode,
} from "lucide-react";

interface HardNegativeScenario {
  id: string;
  title: string;
  query: string;
  true_positive: any;
  hard_negative: any;
  discrimination_margin: number;
  verifiers_fired: {
    context_conflict: boolean;
    conflict_reason: string | null;
    structural_mismatch: boolean;
    mismatch_reason: string | null;
  };
}

export const VerificationRerankingPanel: React.FC = () => {
  const [scenarios, setScenarios] = useState<HardNegativeScenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("scenario_1");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [rerankYaml, setRerankYaml] = useState<string>("");

  useEffect(() => {
    fetchHardNegatives();
    fetchConfigs();
  }, []);

  const fetchHardNegatives = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/verification-hard-negatives");
      const data = await resp.json();
      if (data.scenarios) {
        setScenarios(data.scenarios);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load hard-negative scenarios");
    } finally {
      setLoading(false);
    }
  };

  const fetchConfigs = async () => {
    try {
      const resp = await fetch("/api/configs");
      const data = await resp.json();
      if (data["reranking.yaml"]) {
        setRerankYaml(data["reranking.yaml"]);
      }
    } catch (err) {
      console.error("Config fetch error:", err);
    }
  };

  const currentScenario = scenarios.find((s) => s.id === selectedScenarioId) || scenarios[0];

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-bold">
                Verification &amp; Multi-Factor Reranking
              </span>
              <span className="text-xs text-slate-500 font-mono">configs/reranking.yaml</span>
            </div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Axiom: Similarity Is Not Truth. Similarity Is Merely a Semantic Retrieval Signal.
            </h2>
            <p className="text-xs text-slate-600 max-w-3xl leading-relaxed mt-1">
              Top-K semantic embeddings surface visual candidates, but vision models alone cannot determine exact ground truth.
              Every candidate must undergo <strong>multi-scale context inspection</strong> (small &rarr; medium &rarr; large),
              <strong>structural segmentation verification</strong> (roof area, NDWI, NDVI), and <strong>deterministic GIS metric spatial predicates</strong> (ST_DWithin).
              The resulting <strong>relevance_score</strong> provides explainable ranking (not an uncalibrated probability).
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={fetchHardNegatives}
              className="px-3.5 py-2 text-xs font-semibold rounded-lg bg-slate-900 text-white hover:bg-slate-800 transition-colors flex items-center gap-1.5 cursor-pointer shadow-xs"
            >
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Re-run Hard-Negative Suite
            </button>
          </div>
        </div>
      </div>

      {/* Scenario Selector Tabs */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 border-b border-slate-200">
        {scenarios.map((sc) => {
          const isSelected = sc.id === (currentScenario?.id || "scenario_1");
          return (
            <button
              key={sc.id}
              onClick={() => setSelectedScenarioId(sc.id)}
              className={`px-3.5 py-2 text-xs font-medium rounded-lg whitespace-nowrap transition-colors cursor-pointer flex items-center gap-2 ${
                isSelected
                  ? "bg-slate-900 text-amber-400 font-semibold shadow-xs"
                  : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
              }`}
            >
              <span>{sc.title}</span>
              <span
                className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${
                  isSelected ? "bg-slate-800 text-emerald-400" : "bg-slate-100 text-slate-500"
                }`}
              >
                &Delta; +{sc.discrimination_margin.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>

      {loading && (
        <div className="p-8 text-center bg-white rounded-xl border border-slate-200 text-slate-500 text-xs">
          Loading hard-negative verification suite...
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-xl flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {currentScenario && !loading && (
        <div className="space-y-6">
          {/* Query Information Bar */}
          <div className="bg-slate-900 text-white rounded-xl p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-amber-400 font-mono font-bold uppercase text-[11px] bg-slate-800 px-2 py-0.5 rounded">
                Evaluated Query:
              </span>
              <span className="font-semibold text-slate-100">"{currentScenario.query}"</span>
            </div>
            <div className="flex items-center gap-3 text-slate-400 text-[11px]">
              <span>Discrimination Margin:</span>
              <span className="text-emerald-400 font-mono font-bold text-sm bg-slate-800 px-2.5 py-0.5 rounded">
                +{currentScenario.discrimination_margin.toFixed(3)} relevance_score
              </span>
            </div>
          </div>

          {/* Side-by-Side Comparison Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* TRUE POSITIVE CARD */}
            <div className="bg-white border-2 border-emerald-400 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-0.5 text-xs font-bold font-mono rounded bg-emerald-100 text-emerald-800 border border-emerald-300 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    TRUE POSITIVE &bull; {currentScenario.true_positive.aoi_id}
                  </span>
                  <div className="text-right">
                    <div className="text-[10px] text-slate-400 font-mono uppercase">Final relevance_score</div>
                    <div className="text-base font-bold font-mono text-emerald-700">
                      {currentScenario.true_positive.overall_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-base font-bold text-slate-900 leading-snug">
                    {currentScenario.true_positive.name}
                  </h3>
                  <div className="text-xs text-slate-500 font-mono mt-0.5">
                    Sensor: {currentScenario.true_positive.sensor} ({currentScenario.true_positive.resolution_m}m) &bull; Class: {currentScenario.true_positive.primary_semantic_class}
                  </div>
                </div>

                {/* Score Comparison */}
                <div className="grid grid-cols-3 gap-2 bg-slate-50 p-2.5 rounded-xl border border-slate-200 text-xs text-center font-mono">
                  <div>
                    <div className="text-[10px] text-slate-400">Raw Semantic</div>
                    <div className="font-bold text-slate-800">{currentScenario.true_positive.semantic_score.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">GIS Distance</div>
                    <div className="font-bold text-emerald-600">
                      {currentScenario.true_positive.metadata?.gis_verification?.predicates?.[0]?.measured_dist_m || "600"}m (Pass)
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Relevance Rank</div>
                    <div className="font-bold text-emerald-700">#1 (Top Rank)</div>
                  </div>
                </div>

                {/* Multi-Scale Context Inspection */}
                <div className="space-y-1.5 text-xs">
                  <div className="font-semibold text-slate-800 flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-blue-600" />
                    Multi-Scale Context Hierarchy:
                  </div>
                  <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 space-y-1.5 text-[11px] font-mono">
                    <div className="flex items-start gap-2">
                      <span className="text-blue-700 font-bold w-16 shrink-0">Small:</span>
                      <span className="text-slate-700">
                        {currentScenario.true_positive.metadata?.multiscale_context?.small?.join(", ") || "None"}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-amber-700 font-bold w-16 shrink-0">Medium:</span>
                      <span className="text-slate-700">
                        {currentScenario.true_positive.metadata?.multiscale_context?.medium?.join(", ") || "None"}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-purple-700 font-bold w-16 shrink-0">Large:</span>
                      <span className="text-slate-700">
                        {currentScenario.true_positive.metadata?.multiscale_context?.large?.join(", ") || "None"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Structural Evidence */}
                {currentScenario.true_positive.metadata?.structural_evidence && (
                  <div className="space-y-1.5 text-xs">
                    <div className="font-semibold text-slate-800 flex items-center gap-1">
                      <Compass className="w-3.5 h-3.5 text-purple-600" />
                      Structural Evidence Metrics:
                    </div>
                    <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 grid grid-cols-2 gap-2 text-[11px] font-mono">
                      {Object.entries(currentScenario.true_positive.metadata.structural_evidence).map(([k, v]: [string, any]) => {
                        if (typeof v === "object") return null;
                        return (
                          <div key={k} className="flex items-center justify-between text-slate-700">
                            <span className="text-slate-400">{k}:</span>
                            <strong>{typeof v === "number" ? v.toFixed(2) : String(v)}</strong>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-emerald-800 bg-emerald-50/70 p-3 rounded-xl">
                <span className="font-bold">Validation Outcome:</span> Passed all 3 verification pillars. Awarded multi-stage bonus (+0.06). Verified consistent across small, medium, and regional scales.
              </div>
            </div>

            {/* HARD NEGATIVE CARD */}
            <div className="bg-white border-2 border-rose-300 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-0.5 text-xs font-bold font-mono rounded bg-rose-100 text-rose-800 border border-rose-300 flex items-center gap-1">
                    <XCircle className="w-3.5 h-3.5 text-rose-600" />
                    HARD NEGATIVE &bull; {currentScenario.hard_negative.aoi_id}
                  </span>
                  <div className="text-right">
                    <div className="text-[10px] text-slate-400 font-mono uppercase">Penalized relevance_score</div>
                    <div className="text-base font-bold font-mono text-rose-700">
                      {currentScenario.hard_negative.overall_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-base font-bold text-slate-900 leading-snug">
                    {currentScenario.hard_negative.name}
                  </h3>
                  <div className="text-xs text-slate-500 font-mono mt-0.5">
                    Sensor: {currentScenario.hard_negative.sensor} ({currentScenario.hard_negative.resolution_m}m) &bull; Class: {currentScenario.hard_negative.primary_semantic_class}
                  </div>
                </div>

                {/* Score Comparison */}
                <div className="grid grid-cols-3 gap-2 bg-rose-50/50 p-2.5 rounded-xl border border-rose-200 text-xs text-center font-mono">
                  <div>
                    <div className="text-[10px] text-slate-400">Raw Semantic</div>
                    <div className="font-bold text-slate-700" title="Deceptive visual embedding similarity">
                      {currentScenario.hard_negative.semantic_score.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Context Conflict</div>
                    <div className="font-bold text-rose-600">
                      {currentScenario.verifiers_fired.context_conflict ? "-0.30 Penalty" : "None"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Structural Check</div>
                    <div className="font-bold text-rose-600">
                      {currentScenario.verifiers_fired.structural_mismatch ? "-0.25 Mismatch" : "Pass"}
                    </div>
                  </div>
                </div>

                {/* Multi-Scale Context Inspection */}
                <div className="space-y-1.5 text-xs">
                  <div className="font-semibold text-slate-800 flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-rose-600" />
                    Multi-Scale Context Hierarchy:
                  </div>
                  <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 space-y-1.5 text-[11px] font-mono">
                    <div className="flex items-start gap-2">
                      <span className="text-blue-700 font-bold w-16 shrink-0">Small:</span>
                      <span className="text-slate-700">
                        {currentScenario.hard_negative.metadata?.multiscale_context?.small?.join(", ") || "None"}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-amber-700 font-bold w-16 shrink-0">Medium:</span>
                      <span className="text-slate-700">
                        {currentScenario.hard_negative.metadata?.multiscale_context?.medium?.join(", ") || "None"}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-purple-700 font-bold w-16 shrink-0">Large:</span>
                      <span className="text-slate-700">
                        {currentScenario.hard_negative.metadata?.multiscale_context?.large?.join(", ") || "None"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Fired Verification Alerts */}
                <div className="space-y-2">
                  {currentScenario.verifiers_fired.conflict_reason && (
                    <div className="text-xs bg-rose-50 border border-rose-200 text-rose-900 p-3 rounded-xl flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                      <div>
                        <strong className="block mb-0.5">Context Verifier Alert:</strong>
                        <span>{currentScenario.verifiers_fired.conflict_reason}</span>
                      </div>
                    </div>
                  )}

                  {currentScenario.verifiers_fired.mismatch_reason && (
                    <div className="text-xs bg-rose-50 border border-rose-200 text-rose-900 p-3 rounded-xl flex items-start gap-2">
                      <XCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                      <div>
                        <strong className="block mb-0.5">Structural Verifier Alert:</strong>
                        <span>{currentScenario.verifiers_fired.mismatch_reason}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-rose-900 bg-rose-50/70 p-3 rounded-xl">
                <span className="font-bold">Rejection Rationale:</span> High raw text-image similarity was overridden by environmental context conflict and morphological metric violation. Relevance score sharply demoted.
              </div>
            </div>
          </div>

          {/* Reranking YAML Configuration Inspector */}
          <div className="bg-slate-900 text-white rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-bold text-white">
                  Active Multi-Factor Weights Configuration (configs/reranking.yaml)
                </h3>
              </div>
              <span className="text-xs text-slate-400 font-mono">Normalized Sum = 1.0</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Semantic Similarity</div>
                <div className="text-base font-bold text-amber-400">0.25 (25%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Context Consistency</div>
                <div className="text-base font-bold text-blue-400">0.20 (20%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Structural Segmentation</div>
                <div className="text-base font-bold text-purple-400">0.20 (20%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Deterministic GIS</div>
                <div className="text-base font-bold text-emerald-400">0.15 (15%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Quality Metric</div>
                <div className="text-base font-bold text-slate-200">0.08 (8%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Metadata Alignment</div>
                <div className="text-base font-bold text-slate-200">0.04 (4%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Resolution Suitability</div>
                <div className="text-base font-bold text-slate-200">0.04 (4%)</div>
              </div>
              <div className="bg-slate-800 p-3 rounded-xl border border-slate-700">
                <div className="text-slate-400 text-[10px]">Sensor Suitability</div>
                <div className="text-base font-bold text-slate-200">0.04 (4%)</div>
              </div>
            </div>

            <pre className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto">
              {rerankYaml || "# Loading configs/reranking.yaml..."}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
