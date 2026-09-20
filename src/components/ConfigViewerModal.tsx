import React, { useState, useEffect } from "react";
import { X, FileCode, Check } from "lucide-react";

interface ConfigViewerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ConfigViewerModal: React.FC<ConfigViewerModalProps> = ({ isOpen, onClose }) => {
  const [configs, setConfigs] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState<string>("query_parser.yaml");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetch("/api/configs")
        .then((r) => r.json())
        .then((data) => setConfigs(data))
        .catch(console.error);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const currentContent = configs[activeTab] || "# Loading...";

  const handleCopy = () => {
    navigator.clipboard.writeText(currentContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
      <div className="bg-white border border-slate-300 rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <FileCode className="w-5 h-5 text-slate-800" />
            <h2 className="text-sm font-bold text-slate-900">
              System Configuration Files (configs/*.yaml)
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded-md transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Header */}
        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-100 px-4 py-2">
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {Object.keys(configs).map((cfgName) => (
              <button
                key={cfgName}
                onClick={() => setActiveTab(cfgName)}
                className={`px-3 py-1.5 rounded-md text-xs font-mono transition-colors cursor-pointer ${
                  activeTab === cfgName
                    ? "bg-white text-slate-900 font-bold shadow-xs border border-slate-200"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                configs/{cfgName}
              </button>
            ))}
          </div>

          <button
            onClick={handleCopy}
            className="inline-flex items-center gap-1 px-2.5 py-1 text-xs text-slate-700 bg-white border border-slate-200 rounded hover:bg-slate-50 cursor-pointer"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-600" /> : null}
            {copied ? "Copied" : "Copy YAML"}
          </button>
        </div>

        {/* Code Body */}
        <div className="p-4 overflow-y-auto flex-1 bg-slate-950">
          <pre className="text-emerald-400 font-mono text-xs whitespace-pre-wrap leading-relaxed">
            {currentContent}
          </pre>
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
