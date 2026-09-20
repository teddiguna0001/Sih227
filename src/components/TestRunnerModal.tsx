import React, { useState } from "react";
import { X, Play, Terminal, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

interface TestRunnerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTestsCompleted?: (passed: boolean) => void;
}

export const TestRunnerModal: React.FC<TestRunnerModalProps> = ({
  isOpen,
  onClose,
  onTestsCompleted,
}) => {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<{
    unittest?: { output: string; passed: boolean };
    benchmark?: { output: string; passed: boolean };
  } | null>(null);

  if (!isOpen) return null;

  const handleRunTests = async () => {
    setRunning(true);
    try {
      const resp = await fetch("/api/tests");
      const data = await resp.json();
      setResults(data);
      if (onTestsCompleted) {
        onTestsCompleted(Boolean(data.unittest?.passed && data.benchmark?.passed));
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-white border border-slate-300 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-slate-800" />
            <h2 className="text-sm font-bold text-slate-900">
              Unit Test Suite & Benchmark Runner (SIH26227 Part 1)
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded-md transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 overflow-y-auto flex-1 space-y-4">
          <p className="text-xs text-slate-600">
            Executes <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-slate-800">python3 -m unittest discover tests</code> and{" "}
            <code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-slate-800">python3 scripts/test_query.py</code> directly against the codebase.
          </p>

          <div className="flex items-center gap-3">
            <button
              id="execute-all-tests-btn"
              onClick={handleRunTests}
              disabled={running}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
            >
              {running ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-400" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current text-emerald-400" />
              )}
              {running ? "Executing Tests..." : "Run All Tests Now"}
            </button>

            {results && (
              <div className="flex items-center gap-2 text-xs font-semibold">
                {results.unittest?.passed && results.benchmark?.passed ? (
                  <span className="inline-flex items-center gap-1 text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-md">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> All Tests & Benchmarks Passed (100%)
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-rose-700 bg-rose-50 border border-rose-200 px-2.5 py-1 rounded-md">
                    <AlertCircle className="w-3.5 h-3.5 text-rose-600" /> Issues Detected
                  </span>
                )}
              </div>
            )}
          </div>

          {results?.benchmark && (
            <div>
              <h3 className="text-xs font-bold text-slate-700 mb-1">
                Benchmark Queries (scripts/test_query.py):
              </h3>
              <pre className="bg-slate-900 text-slate-200 p-3 rounded-lg text-[11px] font-mono whitespace-pre-wrap overflow-x-auto max-h-60">
                {results.benchmark.output}
              </pre>
            </div>
          )}

          {results?.unittest && (
            <div>
              <h3 className="text-xs font-bold text-slate-700 mb-1">
                Unittest Discovery Suite (tests/test_query_parser.py):
              </h3>
              <pre className="bg-slate-900 text-emerald-400 p-3 rounded-lg text-[11px] font-mono whitespace-pre-wrap overflow-x-auto">
                {results.unittest.output}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-100 cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
