/**
 * Problem ID: SIH26227 / SH227
 * Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
 * Organization: Ministry of Defence | Theme: Space Technology
 *
 * GIS, Gazetteer, & Metadata Filter Visual Inspection Panel.
 */

import React, { useState, useEffect } from "react";
import {
  MapPin,
  Compass,
  Layers,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Filter,
  Search,
  Database,
  Satellite,
  ArrowRight,
  Maximize2,
  RefreshCw,
  Sliders,
} from "lucide-react";

interface PlaceResult {
  status: "resolved" | "ambiguous" | "error";
  place?: any;
  candidates?: any[];
  error?: string;
  query?: string;
}

interface FilterResult {
  archive_total: number;
  surviving_count: number;
  audit_trail: {
    initial_tile_count: number;
    dropped_geographic: number;
    dropped_temporal: number;
    dropped_sensor: number;
    dropped_resolution: number;
    dropped_quality: number;
    dropped_coverage: number;
    surviving_tile_count: number;
  };
  surviving_tiles: any[];
  all_tiles: any[];
}

interface GISEvalResult {
  predicates: Record<string, boolean>;
  corridors: Record<string, any>;
  coastlines: Record<string, any>;
}

export function GISGazetteerPanel() {
  const [activeTab, setActiveTab] = useState<"gazetteer" | "filter" | "gis">("gazetteer");

  // 1. Gazetteer State
  const [gazetteerQuery, setGazetteerQuery] = useState("Chennai");
  const [contextHint, setContextHint] = useState("");
  const [placeResult, setPlaceResult] = useState<PlaceResult | null>(null);
  const [loadingGazetteer, setLoadingGazetteer] = useState(false);

  // 2. Metadata Filter State
  const [filterQuery, setFilterQuery] = useState("Find agricultural areas near Chennai in 2024 using Sentinel-2");
  const [maxCloud, setMaxCloud] = useState(15);
  const [filterResult, setFilterResult] = useState<FilterResult | null>(null);
  const [loadingFilter, setLoadingFilter] = useState(false);

  // 3. GIS Eval State
  const [gisResult, setGisResult] = useState<GISEvalResult | null>(null);
  const [testDistMeters, setTestDistMeters] = useState(1000);
  const [testLat, setTestLat] = useState(13.064);
  const [testLon, setTestLon] = useState(80.20);
  const [evalLog, setEvalLog] = useState<string | null>(null);

  // Auto-run default gazetteer query on load
  useEffect(() => {
    runGazetteerLookup("Chennai", "");
    runMetadataFilter("Find agricultural areas near Chennai in 2024 using Sentinel-2", 15);
    runGISEval();
  }, []);

  const runGazetteerLookup = async (queryName: string, hint: string) => {
    setLoadingGazetteer(true);
    try {
      const url = `/api/gazetteer?query=${encodeURIComponent(queryName)}&context=${encodeURIComponent(hint)}`;
      const res = await fetch(url);
      const data = await res.json();
      setPlaceResult(data);
    } catch (err: any) {
      setPlaceResult({ status: "error", error: err.message });
    } finally {
      setLoadingGazetteer(false);
    }
  };

  const runMetadataFilter = async (q: string, cloudPct: number) => {
    setLoadingFilter(true);
    try {
      const res = await fetch("/api/metadata-filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          filter: { max_cloud_percent: cloudPct },
        }),
      });
      const data = await res.json();
      setFilterResult(data);
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoadingFilter(false);
    }
  };

  const runGISEval = async () => {
    try {
      const res = await fetch("/api/gis-eval");
      const data = await res.json();
      setGisResult(data);
    } catch (err: any) {
      console.error(err);
    }
  };

  return (
    <div id="gis-gazetteer-panel" className="space-y-6">
      {/* Sub-nav tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2">
          <button
            id="tab-btn-gazetteer"
            onClick={() => setActiveTab("gazetteer")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
              activeTab === "gazetteer"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-white hover:bg-slate-100 text-slate-700 border border-slate-200"
            }`}
          >
            <Compass className="w-3.5 h-3.5 text-amber-400" />
            Offline Strategic Gazetteer
          </button>
          <button
            id="tab-btn-filter"
            onClick={() => setActiveTab("filter")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
              activeTab === "filter"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-white hover:bg-slate-100 text-slate-700 border border-slate-200"
            }`}
          >
            <Filter className="w-3.5 h-3.5 text-blue-400" />
            6-Stage Metadata Filtering
          </button>
          <button
            id="tab-btn-gis"
            onClick={() => setActiveTab("gis")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
              activeTab === "gis"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-white hover:bg-slate-100 text-slate-700 border border-slate-200"
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            Deterministic Metric GIS Verifier
          </button>
        </div>

        <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          100% Offline &bull; Pure Python &bull; Zero External APIs
        </div>
      </div>

      {/* TAB 1: GAZETTEER */}
      {activeTab === "gazetteer" && (
        <div className="space-y-5">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Compass className="w-4 h-4 text-amber-500" />
                  Offline Strategic Gazetteer & Ambiguity Prevention
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Pre-indexed Indian strategic cities, states, borders, and installations. Enforces strict disambiguation without silent errors.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-slate-400 font-medium mr-1">Presets:</span>
                {[
                  { label: "Chennai", query: "Chennai", hint: "" },
                  { label: "Madras (Alias)", query: "Madras", hint: "" },
                  { label: "Aurangabad (Ambiguous!)", query: "Aurangabad", hint: "" },
                  { label: "Aurangabad (MH)", query: "Aurangabad", hint: "Maharashtra" },
                  { label: "Aurangabad (Bihar)", query: "Aurangabad", hint: "Bihar" },
                  { label: "Siachen Glacier", query: "Siachen", hint: "" },
                  { label: "Pokhran", query: "Pokhran", hint: "" },
                ].map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setGazetteerQuery(item.query);
                      setContextHint(item.hint);
                      runGazetteerLookup(item.query, item.hint);
                    }}
                    className={`text-[11px] px-2.5 py-1 rounded-md border font-mono transition-colors cursor-pointer ${
                      item.label.includes("Ambiguous!")
                        ? "bg-amber-50 text-amber-800 border-amber-300 hover:bg-amber-100"
                        : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Inputs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="md:col-span-2 space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Location Query / Entity Name
                </label>
                <div className="relative">
                  <input
                    id="input-gazetteer-query"
                    type="text"
                    value={gazetteerQuery}
                    onChange={(e) => setGazetteerQuery(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg focus:outline-hidden focus:ring-2 focus:ring-slate-900"
                    placeholder="e.g. Chennai, Aurangabad, Pokhran..."
                  />
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Context Hint (Optional Disambiguation)
                </label>
                <input
                  id="input-gazetteer-hint"
                  type="text"
                  value={contextHint}
                  onChange={(e) => setContextHint(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg focus:outline-hidden focus:ring-2 focus:ring-slate-900"
                  placeholder="e.g. Maharashtra, Bihar, Tamil Nadu"
                />
              </div>
            </div>

            <div className="flex justify-end">
              <button
                id="btn-resolve-place"
                onClick={() => runGazetteerLookup(gazetteerQuery, contextHint)}
                disabled={loadingGazetteer || !gazetteerQuery.trim()}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-semibold rounded-lg flex items-center gap-2 cursor-pointer transition-colors"
              >
                {loadingGazetteer ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Compass className="w-3.5 h-3.5 text-amber-400" />
                )}
                Resolve Strategic Geometry
              </button>
            </div>
          </div>

          {/* Place Result Card */}
          {placeResult && (
            <div className="space-y-3">
              {placeResult.status === "ambiguous" && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-amber-900 space-y-3">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-amber-950">
                        Defence Safety Guard: Ambiguous Strategic Location Detected
                      </h4>
                      <p className="text-xs mt-1 text-amber-800 leading-relaxed">
                        {placeResult.error}
                      </p>
                    </div>
                  </div>

                  <div className="bg-white/80 rounded-lg p-3 border border-amber-200 space-y-2">
                    <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
                      Candidate Entities Requiring Confirmation ({placeResult.candidates?.length}):
                    </span>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {placeResult.candidates?.map((c, i) => (
                        <div
                          key={i}
                          className="bg-white p-2.5 rounded border border-slate-200 flex items-center justify-between text-xs"
                        >
                          <div>
                            <span className="font-bold text-slate-900">{c.canonical_name}</span>
                            <span className="text-slate-500 block text-[11px]">
                              Admin Level {c.admin_level} &bull; {c.parent_admin} &bull; Centroid: [{c.centroid.lon.toFixed(4)}, {c.centroid.lat.toFixed(4)}]
                            </span>
                          </div>
                          <button
                            onClick={() => {
                              setContextHint(c.parent_admin);
                              runGazetteerLookup(gazetteerQuery, c.parent_admin);
                            }}
                            className="px-2.5 py-1 text-[11px] font-semibold bg-slate-900 hover:bg-slate-800 text-white rounded cursor-pointer transition-colors"
                          >
                            Disambiguate
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {placeResult.status === "resolved" && placeResult.place && (
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      <div>
                        <h4 className="text-sm font-bold text-slate-900">
                          {placeResult.place.canonical_name}
                        </h4>
                        <span className="text-[11px] text-slate-500 font-mono">
                          Source: {placeResult.place.source} &bull; Confidence: {placeResult.place.confidence * 100}%
                        </span>
                      </div>
                    </div>
                    <span className="text-xs font-mono uppercase bg-emerald-50 text-emerald-800 px-2.5 py-1 rounded-md border border-emerald-200">
                      Resolved Place
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                      <span className="text-[10px] text-slate-500 font-mono uppercase block">Place Type</span>
                      <span className="text-xs font-bold text-slate-900 capitalize">{placeResult.place.place_type}</span>
                    </div>
                    <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                      <span className="text-[10px] text-slate-500 font-mono uppercase block">Admin Hierarchy</span>
                      <span className="text-xs font-bold text-slate-900">
                        Level {placeResult.place.admin_level} ({placeResult.place.parent_admin})
                      </span>
                    </div>
                    <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                      <span className="text-[10px] text-slate-500 font-mono uppercase block">Centroid (Lon, Lat)</span>
                      <span className="text-xs font-mono font-bold text-slate-900">
                        {placeResult.place.centroid.lon.toFixed(4)}, {placeResult.place.centroid.lat.toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                      <span className="text-[10px] text-slate-500 font-mono uppercase block">Default Buffer</span>
                      <span className="text-xs font-bold text-slate-900">{placeResult.place.default_buffer_km} km</span>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <span className="text-[11px] font-mono text-slate-600 font-bold uppercase">
                      Strategic EPSG:4326 Bounding Box & GeoJSON Geometry
                    </span>
                    <pre className="bg-slate-900 text-slate-200 p-3 rounded-lg text-[11px] font-mono overflow-x-auto max-h-48">
                      {JSON.stringify(placeResult.place.geometry, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: METADATA FILTERING */}
      {activeTab === "filter" && (
        <div className="space-y-5">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Filter className="w-4 h-4 text-blue-500" />
                6-Stage Deterministic Archive Filter
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Prunes the satellite archive prior to dense vector search via 6 cascaded stages: AOI &rarr; Temporal &rarr; Sensor &rarr; Resolution &rarr; Quality &rarr; Coverage.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="md:col-span-2 space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Ingress Natural Language Query
                </label>
                <input
                  id="input-filter-query"
                  type="text"
                  value={filterQuery}
                  onChange={(e) => setFilterQuery(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg focus:outline-hidden focus:ring-2 focus:ring-slate-900"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                    Max Cloud Cover Threshold
                  </label>
                  <span className="text-xs font-mono font-bold text-slate-800">{maxCloud}%</span>
                </div>
                <input
                  id="range-max-cloud"
                  type="range"
                  min="0"
                  max="50"
                  step="5"
                  value={maxCloud}
                  onChange={(e) => setMaxCloud(Number(e.target.value))}
                  className="w-full accent-slate-900 cursor-pointer"
                />
              </div>
            </div>

            <div className="flex justify-end">
              <button
                id="btn-run-filter"
                onClick={() => runMetadataFilter(filterQuery, maxCloud)}
                disabled={loadingFilter}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-semibold rounded-lg flex items-center gap-2 cursor-pointer transition-colors"
              >
                {loadingFilter ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Filter className="w-3.5 h-3.5 text-blue-400" />}
                Execute 6-Stage Filter Funnel
              </button>
            </div>
          </div>

          {/* Funnel Stats */}
          {filterResult && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
                <div className="bg-slate-900 text-white p-3 rounded-xl">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Initial Tiles</span>
                  <span className="text-lg font-bold">{filterResult.audit_trail.initial_tile_count}</span>
                </div>
                <div className="bg-rose-50 border border-rose-200 text-rose-900 p-3 rounded-xl">
                  <span className="text-[10px] text-rose-600 uppercase font-mono block">Geo Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_geographic}</span>
                </div>
                <div className="bg-amber-50 border border-amber-200 text-amber-900 p-3 rounded-xl">
                  <span className="text-[10px] text-amber-600 uppercase font-mono block">Time Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_temporal}</span>
                </div>
                <div className="bg-slate-50 border border-slate-200 text-slate-700 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 uppercase font-mono block">Sensor Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_sensor}</span>
                </div>
                <div className="bg-slate-50 border border-slate-200 text-slate-700 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 uppercase font-mono block">Res Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_resolution}</span>
                </div>
                <div className="bg-blue-50 border border-blue-200 text-blue-900 p-3 rounded-xl">
                  <span className="text-[10px] text-blue-600 uppercase font-mono block">Cloud Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_quality}</span>
                </div>
                <div className="bg-slate-50 border border-slate-200 text-slate-700 p-3 rounded-xl">
                  <span className="text-[10px] text-slate-500 uppercase font-mono block">Cover Drop</span>
                  <span className="text-lg font-bold">-{filterResult.audit_trail.dropped_coverage}</span>
                </div>
                <div className="bg-emerald-900 text-white p-3 rounded-xl shadow-xs">
                  <span className="text-[10px] text-emerald-300 uppercase font-mono block">Surviving</span>
                  <span className="text-lg font-bold">{filterResult.surviving_count}</span>
                </div>
              </div>

              {/* Surviving Tiles List */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <Satellite className="w-4 h-4 text-emerald-600" />
                  Surviving Catalog Tiles ({filterResult.surviving_tiles.length} of {filterResult.archive_total} total)
                </h4>

                <div className="space-y-2">
                  {filterResult.surviving_tiles.map((tile, i) => (
                    <div
                      key={i}
                      className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-900 font-mono">{tile.tile_id}</span>
                          <span className="text-[10px] px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-semibold">
                            {tile.sensor} ({tile.resolution}m GSD)
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-500 flex flex-wrap gap-x-3">
                          <span>Acquired: {tile.datetime}</span>
                          <span>Cloud: {tile.quality.cloud_cover_percent}%</span>
                          <span>Valid Pixels: {tile.quality.valid_pixel_pct}%</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5 flex-wrap">
                        {tile.semantic_features.map((feat: string, fi: number) => (
                          <span key={fi} className="px-2 py-0.5 bg-white border border-slate-200 rounded text-[11px] text-slate-700">
                            {feat}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: DETERMINISTIC GIS VERIFIER */}
      {activeTab === "gis" && (
        <div className="space-y-5">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                PostGIS-Compatible Deterministic Spatial Predicates
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Evaluates ST_DWithin, ST_Buffer, ST_Contains, and metric distance constraints using local tangent projection without online geocoders.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Test Point Longitude
                </label>
                <input
                  id="input-gis-lon"
                  type="number"
                  step="0.001"
                  value={testLon}
                  onChange={(e) => setTestLon(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Test Point Latitude
                </label>
                <input
                  id="input-gis-lat"
                  type="number"
                  step="0.001"
                  value={testLat}
                  onChange={(e) => setTestLat(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider">
                  Distance Threshold (Meters)
                </label>
                <div className="flex gap-1.5">
                  {[500, 1000, 2000].map((d) => (
                    <button
                      key={d}
                      onClick={() => setTestDistMeters(d)}
                      className={`flex-1 py-2 text-xs font-mono font-bold rounded-lg border cursor-pointer ${
                        testDistMeters === d
                          ? "bg-slate-900 text-white border-slate-900"
                          : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100"
                      }`}
                    >
                      {d}m
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs font-mono space-y-1 text-slate-700">
              <span className="font-bold text-slate-900 block">Baseline Verification Vectors:</span>
              <div>&bull; NH48 Grand Southern Trunk Industrial Highway Corridor (5 vertices)</div>
              <div>&bull; Coromandel Coastline Polyline (6 vertices)</div>
            </div>
          </div>

          {/* GIS Results */}
          {gisResult && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  Deterministic Predicate Verification
                </h4>

                <div className="space-y-2">
                  {Object.entries(gisResult.predicates).map(([key, passed], idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-between text-xs font-mono"
                    >
                      <span className="text-slate-800">{key}</span>
                      <span
                        className={`px-2 py-0.5 rounded font-bold ${
                          passed ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
                        }`}
                      >
                        {passed ? "PASS (True)" : "FAIL (False)"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-3">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <Database className="w-4 h-4 text-blue-600" />
                  Corridor GeoJSON Polylines
                </h4>
                <pre className="bg-slate-900 text-slate-200 p-3 rounded-lg text-[11px] font-mono overflow-x-auto max-h-56">
                  {JSON.stringify(gisResult.corridors, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
