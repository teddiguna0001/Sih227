/**
 * Semantic Vector Search & Spatial Clustering Panel.
 * Problem ID: SIH26227 / SH227
 * Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
 * Organization: Ministry of Defence | Theme: Space Technology
 */

import React, { useState, useEffect } from "react";
import {
  Layers,
  Search,
  Sparkles,
  Sliders,
  Database,
  Network,
  Maximize2,
  Minimize2,
  Compass,
  AlertTriangle,
  Info,
  Calendar,
  Satellite,
  CheckCircle,
  Copy,
  ChevronDown,
  ChevronRight,
  Filter,
} from "lucide-react";

interface ConstituentTile {
  tile_id: string;
  similarity_score: number;
  geometry: any;
  datetime: string;
  metadata: any;
  quality: any;
  sensor: string;
  resolution: number;
}

interface CandidateRegion {
  aoi_id: string;
  name: string;
  bbox: [number, number, number, number];
  centroid: [number, number];
  area_sqkm: number;
  sensor: string;
  resolution_m: number;
  temporal_range: [string, string];
  primary_semantic_class: string;
  semantic_score: number;
  spatial_score: number;
  temporal_score: number;
  overall_score: number;
  metadata: {
    tile_id: string;
    similarity_score: number;
    geometry: any;
    datetime: string;
    quality: any;
    tile_ids: string[];
    tile_count: number;
    duplicate_count: number;
    mean_similarity: number;
    max_similarity: number;
    constituent_tiles: ConstituentTile[];
    similarity_is_hypothesis: boolean;
    hypothesis_notice: string;
    cluster_method: string;
    raw_similarities: Record<string, number>;
  };
}

interface EmbeddingCompareResult {
  concept: string;
  tags: string[];
  cosine_similarity: number;
}

export function SemanticVectorPanel() {
  const [query, setQuery] = useState<string>("Find agricultural areas near Chennai.");
  const [loading, setLoading] = useState<boolean>(false);
  const [candidates, setCandidates] = useState<CandidateRegion[]>([]);
  const [archiveStats, setArchiveStats] = useState<any>(null);
  const [embeddingMatches, setEmbeddingMatches] = useState<EmbeddingCompareResult[]>([]);
  const [expandedAoi, setExpandedAoi] = useState<string | null>(null);

  // Tunable Top-K limits and grouping thresholds
  const [initialTopK, setInitialTopK] = useState<number>(100);
  const [verificationTopK, setVerificationTopK] = useState<number>(50);
  const [finalTopK, setFinalTopK] = useState<number>(10);
  const [iouDupThresh, setIouDupThresh] = useState<number>(0.85);
  const [iouOverlapThresh, setIouOverlapThresh] = useState<number>(0.25);
  const [distThreshKm, setDistThreshKm] = useState<number>(5.0);

  useEffect(() => {
    fetchArchiveStats();
    executeSearch();
  }, []);

  const fetchArchiveStats = async () => {
    try {
      const resp = await fetch("/api/vector-archive-stats");
      const data = await resp.json();
      setArchiveStats(data);
    } catch (err) {
      console.error("Failed to fetch archive stats:", err);
    }
  };

  const executeSearch = async (customQuery?: string) => {
    const q = customQuery || query;
    setLoading(true);
    try {
      // 1. Fetch Candidates with Spatial Clustering
      const resp = await fetch("/api/semantic-candidates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          top_k: finalTopK,
          config: {
            initial_top_k: initialTopK,
            verification_top_k: verificationTopK,
            final_top_k: finalTopK,
            iou_duplicate_threshold: iouDupThresh,
            iou_overlap_threshold: iouOverlapThresh,
            centroid_distance_km_threshold: distThreshKm,
          },
        }),
      });
      const data = await resp.json();
      setCandidates(data.candidates || []);
      if (data.candidates && data.candidates.length > 0) {
        setExpandedAoi(data.candidates[0].aoi_id);
      }

      // 2. Fetch Embedding Space alignment comparison
      const embResp = await fetch(`/api/embedding-compare?query=${encodeURIComponent(q)}`);
      const embData = await embResp.json();
      setEmbeddingMatches(embData.similarities || []);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner: Core Scientific Principle */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-white">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-2xs font-bold uppercase tracking-wider rounded bg-amber-400 text-slate-950">
                Layer 4
              </span>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                <Network className="w-4 h-4 text-amber-400" />
                Vision-Language Semantic Retrieval & Spatial Grouping
              </h3>
            </div>
            <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
              Enforces joint text-image embedding space (RemoteCLIP ViT-B/32, 512 dimensions).
              Groups overlapping tiles and temporal revisits via IoU &amp; centroid clustering into
              coherent Candidate Geographic Regions.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-slate-800/80 px-3 py-2 rounded-lg border border-slate-700/60 shrink-0">
            <div className="text-right">
              <div className="text-2xs text-slate-400 font-mono">ARCHIVE TILES</div>
              <div className="text-sm font-bold text-amber-400 font-mono">
                {archiveStats?.total_tiles || 80} Indexed
              </div>
            </div>
            <div className="h-8 w-px bg-slate-700" />
            <div className="text-right">
              <div className="text-2xs text-slate-400 font-mono">SPACE DIM</div>
              <div className="text-sm font-bold text-sky-400 font-mono">512-D L2</div>
            </div>
          </div>
        </div>

        {/* Hypothesis Notice Banner */}
        <div className="mt-3 flex items-center gap-2 bg-amber-500/10 border border-amber-500/30 px-3 py-2 rounded-lg text-2xs text-amber-300">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
          <span>
            <strong>Scientific Axiom:</strong> Similarity is <em>NOT</em> truth. Vector similarity is strictly a semantic retrieval hypothesis signal that triggers downstream GIS, context, and segmentation verifiers.
          </span>
        </div>
      </div>

      {/* Query Bar and Preset Prompts */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
        <label className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
          <Search className="w-3.5 h-3.5 text-slate-500" />
          Natural Language Semantic Query
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && executeSearch()}
            className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-400 font-mono"
            placeholder="e.g. Find industrial buildings and factories along highways near Chennai."
          />
          <button
            onClick={() => executeSearch()}
            disabled={loading}
            className="px-4 py-2 bg-amber-400 hover:bg-amber-500 text-slate-950 font-semibold text-xs rounded-lg transition-colors flex items-center gap-1.5 shrink-0 cursor-pointer disabled:opacity-50"
          >
            {loading ? (
              <span className="animate-spin inline-block w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            Run Semantic Retrieval
          </button>
        </div>

        {/* Preset Query Chips */}
        <div className="flex items-center gap-1.5 flex-wrap pt-1">
          <span className="text-2xs font-semibold text-slate-400 uppercase tracking-wider mr-1">
            Test Queries:
          </span>
          {[
            "Find agricultural areas near Chennai.",
            "Find industrial buildings and factories near Mumbai harbour.",
            "Find desert earthworks and testing facilities in Pokhran.",
            "Find mountain passes and strategic highways in Ladakh.",
            "Find naval docks and coastal shipyards near Visakhapatnam.",
          ].map((preset) => (
            <button
              key={preset}
              onClick={() => {
                setQuery(preset);
                executeSearch(preset);
              }}
              className="text-2xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-md border border-slate-200 transition-colors cursor-pointer"
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Top-K Configuration Tuning & Cross-Modal Embedding Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Top-K & Clustering Parameters */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-slate-600" />
                Pipeline Top-K &amp; Spatial Tuning
              </h4>
              <span className="text-2xs font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                configs/retrieval.yaml
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Initial ANN Vector Retrieval (Top-K)</span>
                  <span className="font-mono font-bold text-slate-900">{initialTopK}</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="150"
                  step="10"
                  value={initialTopK}
                  onChange={(e) => setInitialTopK(Number(e.target.value))}
                  className="w-full accent-amber-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Verification Filter Clamping (Top-K)</span>
                  <span className="font-mono font-bold text-slate-900">{verificationTopK}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={verificationTopK}
                  onChange={(e) => setVerificationTopK(Number(e.target.value))}
                  className="w-full accent-amber-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Final Candidate Regions (Final Top-K)</span>
                  <span className="font-mono font-bold text-slate-900">{finalTopK}</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="25"
                  step="1"
                  value={finalTopK}
                  onChange={(e) => setFinalTopK(Number(e.target.value))}
                  className="w-full accent-amber-500 cursor-pointer"
                />
              </div>

              <div className="border-t border-slate-100 pt-3">
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>IoU Duplicate Grouping Threshold</span>
                  <span className="font-mono font-bold text-slate-900">{iouDupThresh}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={iouDupThresh}
                  onChange={(e) => setIouDupThresh(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>IoU Overlap Clustering Threshold</span>
                  <span className="font-mono font-bold text-slate-900">{iouOverlapThresh}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={iouOverlapThresh}
                  onChange={(e) => setIouOverlapThresh(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 mb-1">
                  <span>Centroid Proximity Radius (km)</span>
                  <span className="font-mono font-bold text-slate-900">{distThreshKm} km</span>
                </div>
                <input
                  type="range"
                  min="2.0"
                  max="15.0"
                  step="1.0"
                  value={distThreshKm}
                  onChange={(e) => setDistThreshKm(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
              </div>

              <button
                onClick={() => executeSearch()}
                className="w-full mt-2 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded font-semibold text-2xs transition-colors cursor-pointer"
              >
                Apply Parameters &amp; Re-cluster
              </button>
            </div>
          </div>

          {/* Cross-Modal Embedding Space Alignment Table */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
            <h4 className="text-xs font-bold text-slate-900 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-amber-500" />
                Vision-Language Cross-Modal Alignment
              </span>
              <span className="text-2xs font-mono text-emerald-600 font-semibold">
                RemoteCLIP ViT
              </span>
            </h4>
            <p className="text-2xs text-slate-500 leading-normal">
              Cosine similarity between query text embedding and synthetic image spectral/semantic features:
            </p>

            <div className="space-y-2 text-xs">
              {embeddingMatches.map((item, idx) => {
                const pct = Math.max(0, Math.min(100, item.cosine_similarity * 100));
                const isTop = idx === 0 && item.cosine_similarity > 0.45;
                return (
                  <div
                    key={item.concept}
                    className={`p-2 rounded-lg border ${
                      isTop
                        ? "bg-emerald-50/60 border-emerald-300"
                        : "bg-slate-50 border-slate-200"
                    }`}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-semibold text-slate-800 capitalize">
                        {item.concept}
                      </span>
                      <span
                        className={`font-mono font-bold ${
                          isTop ? "text-emerald-700" : "text-slate-600"
                        }`}
                      >
                        {item.cosine_similarity.toFixed(4)}
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isTop ? "bg-emerald-500" : "bg-slate-400"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="mt-1 text-3xs text-slate-400 truncate">
                      Tags: {item.tags.join(", ")}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Clustered Candidate Geographic Regions */}
        <div className="lg:col-span-8 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-amber-500" />
                Candidate Geographic Regions ({candidates.length} Regions Formed)
              </h3>
              <p className="text-xs text-slate-500">
                Spatially clustered from {initialTopK} retrieved tiles. Grouped by IoU overlap &amp; proximity.
              </p>
            </div>
            <div className="text-2xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200">
              Pipeline: Query &rarr; RemoteCLIP &rarr; MetadataFilter &rarr; ANN &rarr; Clusters
            </div>
          </div>

          {candidates.length === 0 ? (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-8 text-center text-slate-500">
              <Info className="w-8 h-8 text-slate-400 mx-auto mb-2" />
              <p className="text-sm font-semibold">No candidate geographic regions matched.</p>
              <p className="text-xs text-slate-400 mt-1">
                Try widening your query or adjusting the similarity threshold in the parameters panel.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {candidates.map((cand, idx) => {
                const isExpanded = expandedAoi === cand.aoi_id;
                const tileCount = cand.metadata.tile_count || 1;
                const dupCount = cand.metadata.duplicate_count || 0;

                return (
                  <div
                    key={cand.aoi_id}
                    className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden transition-all hover:border-slate-300"
                  >
                    {/* Header bar */}
                    <div
                      onClick={() => setExpandedAoi(isExpanded ? null : cand.aoi_id)}
                      className="p-4 cursor-pointer flex items-start justify-between gap-4 bg-slate-50/50 hover:bg-slate-50"
                    >
                      <div className="flex items-start gap-3">
                        <button className="mt-0.5 text-slate-400 hover:text-slate-600">
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </button>
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="px-2 py-0.5 text-2xs font-bold font-mono bg-slate-900 text-amber-400 rounded">
                              {cand.aoi_id}
                            </span>
                            <span className="font-semibold text-slate-900 text-sm">
                              {cand.name}
                            </span>
                            <span className="px-2 py-0.5 text-3xs font-semibold rounded bg-blue-100 text-blue-800">
                              {tileCount} {tileCount === 1 ? "tile" : "tiles"} clustered
                            </span>
                            {dupCount > 0 && (
                              <span className="px-2 py-0.5 text-3xs font-semibold rounded bg-purple-100 text-purple-800">
                                {dupCount} duplicate/revisit {dupCount === 1 ? "pass" : "passes"}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-4 text-xs text-slate-500 mt-1.5 font-mono">
                            <span className="flex items-center gap-1">
                              <Compass className="w-3 h-3 text-slate-400" />
                              Centroid: ({cand.centroid[0].toFixed(3)}, {cand.centroid[1].toFixed(3)})
                            </span>
                            <span>Area: {cand.area_sqkm} km&sup2;</span>
                            <span className="flex items-center gap-1">
                              <Calendar className="w-3 h-3 text-slate-400" />
                              {cand.temporal_range[0]} &rarr; {cand.temporal_range[1]}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Scores Badge Cluster */}
                      <div className="flex items-center gap-2 shrink-0">
                        <div className="text-right">
                          <div className="text-3xs font-semibold text-slate-400 uppercase tracking-wider">
                            Retrieval Score
                          </div>
                          <div className="text-sm font-bold font-mono text-slate-900">
                            {cand.overall_score.toFixed(3)}
                          </div>
                        </div>
                        <div className="bg-amber-100 text-amber-900 px-2 py-1 rounded text-2xs font-mono font-bold">
                          Sim: {cand.semantic_score.toFixed(3)}
                        </div>
                      </div>
                    </div>

                    {/* Expanded Detail View: Constituent Tiles & GeoJSON Bounds */}
                    {isExpanded && (
                      <div className="p-4 border-t border-slate-100 bg-white space-y-4">
                        {/* Spatial Extents */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-slate-50 p-3 rounded-lg border border-slate-200/80">
                          <div>
                            <span className="text-slate-400 text-2xs font-semibold uppercase block">
                              Union Bounding Box (EPSG:4326)
                            </span>
                            <span className="font-mono text-slate-800 text-xs">
                              [{cand.bbox[0]}, {cand.bbox[1]}, {cand.bbox[2]}, {cand.bbox[3]}]
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-400 text-2xs font-semibold uppercase block">
                              Dominant Sensor &amp; Best GSD
                            </span>
                            <span className="font-mono text-slate-800 text-xs">
                              {cand.sensor} &bull; {cand.resolution_m}m ground resolution
                            </span>
                          </div>
                        </div>

                        {/* Constituent Tiles Table */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <h5 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                              <Satellite className="w-3.5 h-3.5 text-slate-600" />
                              Constituent Indexed Tiles in Region ({cand.metadata.constituent_tiles.length})
                            </h5>
                            <span className="text-2xs text-slate-400">
                              Method: {cand.metadata.cluster_method}
                            </span>
                          </div>

                          <div className="overflow-x-auto border border-slate-200 rounded-lg">
                            <table className="w-full text-left text-xs">
                              <thead className="bg-slate-100/75 border-b border-slate-200 text-slate-600 font-semibold text-2xs">
                                <tr>
                                  <th className="py-2 px-3">Tile ID</th>
                                  <th className="py-2 px-3">Similarity (Hypothesis)</th>
                                  <th className="py-2 px-3">Sensor</th>
                                  <th className="py-2 px-3">Resolution</th>
                                  <th className="py-2 px-3">Timestamp (UTC)</th>
                                  <th className="py-2 px-3">Cloud %</th>
                                  <th className="py-2 px-3">Usable</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100 font-mono">
                                {cand.metadata.constituent_tiles.map((tile) => (
                                  <tr key={tile.tile_id} className="hover:bg-slate-50/80">
                                    <td className="py-2 px-3 font-semibold text-slate-900">
                                      {tile.tile_id}
                                    </td>
                                    <td className="py-2 px-3">
                                      <span className="text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                                        {tile.similarity_score.toFixed(4)}
                                      </span>
                                    </td>
                                    <td className="py-2 px-3 text-slate-600">{tile.sensor}</td>
                                    <td className="py-2 px-3 text-slate-600">{tile.resolution}m</td>
                                    <td className="py-2 px-3 text-slate-600 text-2xs">
                                      {tile.datetime}
                                    </td>
                                    <td className="py-2 px-3 text-slate-600">
                                      {tile.quality.cloud_cover_percent}%
                                    </td>
                                    <td className="py-2 px-3">
                                      {tile.quality.usable_data_flag ? (
                                        <span className="text-emerald-600 flex items-center gap-1 text-2xs">
                                          <CheckCircle className="w-3 h-3" /> Yes
                                        </span>
                                      ) : (
                                        <span className="text-red-500 flex items-center gap-1 text-2xs">
                                          <AlertTriangle className="w-3 h-3" /> High Cloud
                                        </span>
                                      )}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Raw Candidate Metadata JSON toggle */}
                        <div className="pt-2">
                          <details className="text-2xs text-slate-500">
                            <summary className="cursor-pointer font-semibold hover:text-slate-800">
                              View Candidate Geographic Region Contract JSON
                            </summary>
                            <pre className="mt-2 bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto font-mono text-3xs leading-relaxed">
                              {JSON.stringify(cand, null, 2)}
                            </pre>
                          </details>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
