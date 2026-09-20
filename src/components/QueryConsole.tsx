import React from "react";
import { Search, Play, ArrowRight, Sparkles, Filter } from "lucide-react";

interface QueryConsoleProps {
  query: string;
  setQuery: (q: string) => void;
  onParse: () => void;
  onRetrieve: () => void;
  loading: boolean;
}

export const BENCHMARK_QUERIES = [
  "Find agricultural areas near Chennai.",
  "Find industrial buildings within 2 km of highways.",
  "Show water bodies in 2024.",
  "Find areas that changed between 2020 and 2025.",
  "Find road development.",
  "Show Sentinel-2 imagery of Pokhran in 2023.",
  "Find vegetation loss in Ladakh.",
  "Find land clearance near Port Blair.",
];

export const QueryConsole: React.FC<QueryConsoleProps> = ({
  query,
  setQuery,
  onParse,
  onRetrieve,
  loading,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-3">
        <label htmlFor="satellite-query-input" className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <Search className="w-4 h-4 text-slate-600" />
          Natural Language Semantic Query (Part 1 Ingress)
        </label>
        <span className="text-xs text-slate-500 font-mono">
          Deterministic Rule Engine &bull; Non-LLM Dependent
        </span>
      </div>

      <div className="relative flex items-center gap-2">
        <input
          id="satellite-query-input"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !loading) {
              onRetrieve();
            }
          }}
          placeholder="e.g. Find industrial buildings within 2 km of highways..."
          className="w-full px-4 py-2.5 text-sm bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-900 focus:bg-white text-slate-900 placeholder:text-slate-400 font-mono"
        />

        <button
          id="parse-only-btn"
          onClick={onParse}
          disabled={loading || !query.trim()}
          className="px-4 py-2.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded-lg transition-colors shrink-0 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5"
          title="Run query parser and change router"
        >
          <Filter className="w-3.5 h-3.5" />
          Parse Only
        </button>

        <button
          id="execute-pipeline-btn"
          onClick={onRetrieve}
          disabled={loading || !query.trim()}
          className="px-5 py-2.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors shrink-0 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5 shadow-xs"
          title="Run full Part 1 pipeline (parser, verifiers, reranker)"
        >
          {loading ? (
            <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <Play className="w-3.5 h-3.5 fill-current text-emerald-400" />
          )}
          Retrieve AOIs
        </button>
      </div>

      {/* Preset Queries */}
      <div className="mt-4 pt-3 border-t border-slate-100">
        <div className="text-xs font-medium text-slate-500 mb-2 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-amber-500" />
          Benchmark and Sample Space-Technology Queries:
        </div>
        <div className="flex flex-wrap gap-1.5">
          {BENCHMARK_QUERIES.map((bq, i) => (
            <button
              key={bq}
              id={`preset-query-${i}`}
              onClick={() => {
                setQuery(bq);
              }}
              className={`text-xs px-2.5 py-1 rounded-md border text-left transition-colors cursor-pointer ${
                query === bq
                  ? "bg-slate-900 text-white border-slate-900"
                  : "bg-slate-50 hover:bg-slate-100 text-slate-700 border-slate-200"
              }`}
            >
              {i < 5 ? (
                <span className="font-bold text-amber-600 mr-1.5 font-mono">Q{i + 1}</span>
              ) : null}
              {bq}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
