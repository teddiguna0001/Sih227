import React from "react";
import { Satellite, Shield, Terminal, CheckCircle2, FileText } from "lucide-react";

interface HeaderProps {
  onOpenTests: () => void;
  onOpenConfigs: () => void;
  testPassed?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ onOpenTests, onOpenConfigs, testPassed }) => {
  return (
    <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-slate-900 text-amber-400 flex items-center justify-center shrink-0 shadow-xs">
            <Satellite className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold px-2 py-0.5 rounded-sm bg-slate-100 text-slate-700 border border-slate-200">
                Problem ID: SIH26227 / SH227
              </span>
              <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-sm">
                <Shield className="w-3 h-3" /> Ministry of Defence
              </span>
            </div>
            <h1 className="text-lg font-bold text-slate-900 tracking-tight">
              Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
            </h1>
            <p className="text-xs text-slate-500">
              Theme: Space Technology &bull; PART 1: Semantic Retrieval Engine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start md:self-auto">
          <button
            id="open-configs-btn"
            onClick={onOpenConfigs}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 rounded-md transition-colors cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5 text-slate-600" />
            Configs (YAML)
          </button>
          <button
            id="run-tests-header-btn"
            onClick={onOpenTests}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-md transition-colors shadow-xs cursor-pointer"
          >
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            Run Test Suite
            {testPassed && <CheckCircle2 className="w-3 h-3 text-emerald-400 ml-0.5" />}
          </button>
        </div>
      </div>
    </header>
  );
};
