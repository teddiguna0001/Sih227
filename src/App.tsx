/**
 * Problem ID: SIH26227 / SH227
 * Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
 * Organization: Ministry of Defence | Theme: Space Technology
 * PART 1: Semantic Retrieval Interactive Console
 */

import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { QueryConsole, BENCHMARK_QUERIES } from "./components/QueryConsole";
import { ParsedSchemaViewer } from "./components/ParsedSchemaViewer";
import { CandidateAOICards } from "./components/CandidateAOICards";
import { TestRunnerModal } from "./components/TestRunnerModal";
import { ConfigViewerModal } from "./components/ConfigViewerModal";
import { ArchitectureOverview } from "./components/ArchitectureOverview";
import { GISGazetteerPanel } from "./components/GISGazetteerPanel";
import { SemanticVectorPanel } from "./components/SemanticVectorPanel";
import { VerificationRerankingPanel } from "./components/VerificationRerankingPanel";
import { ParsedQueryData, CandidateAOIData, RetrievalResultData } from "./types";
import { Satellite, Shield, Terminal, ArrowUpRight, CheckCircle2 } from "lucide-react";

export default function App() {
  const [query, setQuery] = useState<string>("Find agricultural areas near Chennai.");
  const [loading, setLoading] = useState<boolean>(false);
  const [parsed, setParsed] = useState<ParsedQueryData | null>(null);
  const [candidates, setCandidates] = useState<CandidateAOIData[]>([]);
  const [trace, setTrace] = useState<string[]>([]);
  const [testsModalOpen, setTestsModalOpen] = useState<boolean>(false);
  const [configsModalOpen, setConfigsModalOpen] = useState<boolean>(false);
  const [testsPassed, setTestsPassed] = useState<boolean>(true);
  const [activeView, setActiveView] = useState<"results" | "verification" | "vector_search" | "gis_gazetteer" | "architecture">("results");

  // Initial load: parse default query
  useEffect(() => {
    executeRetrieval("Find agricultural areas near Chennai.");
  }, []);

  const executeParseOnly = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const resp = await fetch("/api/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await resp.json();
      setParsed(data);
      setCandidates([]);
      setTrace([`Direct Query Parser executed at ${new Date().toLocaleTimeString()}`]);
    } catch (err) {
      console.error("Parse error:", err);
    } finally {
      setLoading(false);
    }
  };

  const executeRetrieval = async (targetQuery?: string) => {
    const q = targetQuery || query;
    if (!q.trim()) return;
    setLoading(true);
    try {
      const resp = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, candidates: 5 }),
      });
      const data: RetrievalResultData = await resp.json();
      setParsed(data.parsed_query);
      setCandidates(data.candidate_aois);
      setTrace(data.execution_trace);
    } catch (err) {
      console.error("Retrieval error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex flex-col font-sans">
      {/* Header */}
      <Header
        onOpenTests={() => setTestsModalOpen(true)}
        onOpenConfigs={() => setConfigsModalOpen(true)}
        testPassed={testsPassed}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* Banner with Objective & Context */}
        <div className="bg-slate-900 text-white rounded-2xl p-5 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400 bg-slate-800 px-2.5 py-0.5 rounded">
                Part 1 Foundation
              </span>
              <span className="text-xs text-slate-400">
                Space Technology &bull; Ministry of Defence
              </span>
            </div>
            <h2 className="text-base font-bold text-white tracking-tight">
              Satellite Imagery Semantic Ingress & Deterministic Query Engine
            </h2>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Accepts free-form natural language queries, extracts semantic targets, dates,
              spatial distances, sensors, and gazetteer references without inventing constraints.
              Routes queries into 11 distinct multi-temporal change types or static searches.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setActiveView("results")}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeView === "results"
                  ? "bg-amber-400 text-slate-950 shadow-xs"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Interactive Console
            </button>
            <button
              onClick={() => setActiveView("verification")}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeView === "verification"
                  ? "bg-amber-400 text-slate-950 shadow-xs"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Verification &amp; Reranking
            </button>
            <button
              onClick={() => setActiveView("vector_search")}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeView === "vector_search"
                  ? "bg-amber-400 text-slate-950 shadow-xs"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Vector Search &amp; Grouping
            </button>
            <button
              onClick={() => setActiveView("gis_gazetteer")}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeView === "gis_gazetteer"
                  ? "bg-amber-400 text-slate-950 shadow-xs"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              GIS &amp; Gazetteer Engine
            </button>
            <button
              onClick={() => setActiveView("architecture")}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeView === "architecture"
                  ? "bg-amber-400 text-slate-950 shadow-xs"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Module Blueprint
            </button>
          </div>
        </div>

        {/* View Switching */}
        {activeView === "results" && (
          <div className="space-y-6">
            <QueryConsole
              query={query}
              setQuery={(q) => {
                setQuery(q);
                // Optional auto-retrieve on preset click
                executeRetrieval(q);
              }}
              onParse={executeParseOnly}
              onRetrieve={() => executeRetrieval()}
              loading={loading}
            />

            {parsed && <ParsedSchemaViewer parsed={parsed} />}

            {candidates.length > 0 && (
              <CandidateAOICards candidates={candidates} trace={trace} />
            )}
          </div>
        )}

        {activeView === "verification" && <VerificationRerankingPanel />}

        {activeView === "vector_search" && <SemanticVectorPanel />}

        {activeView === "gis_gazetteer" && <GISGazetteerPanel />}

        {activeView === "architecture" && <ArchitectureOverview />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        Problem ID: SIH26227 / SH227 &bull; Ministry of Defence &bull; Space Technology &bull; Part 1: Semantic Retrieval
      </footer>

      {/* Modals */}
      <TestRunnerModal
        isOpen={testsModalOpen}
        onClose={() => setTestsModalOpen(false)}
        onTestsCompleted={(passed) => setTestsPassed(passed)}
      />

      <ConfigViewerModal
        isOpen={configsModalOpen}
        onClose={() => setConfigsModalOpen(false)}
      />
    </div>
  );
}
