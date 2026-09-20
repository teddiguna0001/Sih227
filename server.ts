import express from "express";
import path from "path";
import fs from "fs";
import { exec } from "child_process";
import { promisify } from "util";
import { createServer as createViteServer } from "vite";

const execAsync = promisify(exec);
const PORT = 3000;

async function startServer() {
  const app = express();
  app.use(express.json());

  // API 1: Health check
  app.get("/api/health", (req, res) => {
    res.json({
      status: "ok",
      problem_id: "SIH26227 / SH227",
      title: "Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery",
      theme: "Space Technology",
      organization: "Ministry of Defence",
    });
  });

  // API 2: Parse Query using Python query_parser
  app.post("/api/parse", async (req, res) => {
    try {
      const query = (req.body.query || "").replace(/"/g, '\\"');
      if (!query.trim()) {
        return res.status(400).json({ error: "Query is required" });
      }

      const { stdout } = await execAsync(`python3 scripts/test_query.py --query "${query}" --json`);
      const parsedData = JSON.parse(stdout);
      res.json(parsedData);
    } catch (error: any) {
      console.error("Query parse error:", error);
      res.status(500).json({
        error: error.message || "Failed to parse query",
        details: error.stderr || error.stdout,
      });
    }
  });

  // API 3: Run full Part 1 Retrieval Pipeline
  app.post("/api/retrieve", async (req, res) => {
    try {
      const query = (req.body.query || "").replace(/"/g, '\\"');
      const candidates = req.body.candidates || 5;

      const { stdout } = await execAsync(`python3 scripts/run_part1.py --query "${query}" --candidates ${candidates} --json`);
      const resultData = JSON.parse(stdout);
      res.json(resultData);
    } catch (error: any) {
      console.error("Retrieval pipeline error:", error);
      res.status(500).json({
        error: error.message || "Failed to execute retrieval pipeline",
        details: error.stderr || error.stdout,
      });
    }
  });

  // API 4: Run unit tests and benchmark
  app.get("/api/tests", async (req, res) => {
    try {
      const [unitTestRes, benchmarkRes] = await Promise.all([
        execAsync("python3 -m unittest discover tests").catch((e) => ({ stdout: e.stdout, stderr: e.stderr })),
        execAsync("python3 scripts/test_query.py").catch((e) => ({ stdout: e.stdout, stderr: e.stderr })),
      ]);

      res.json({
        unittest: {
          output: unitTestRes.stdout || unitTestRes.stderr,
          passed: (unitTestRes.stderr || unitTestRes.stdout).includes("OK"),
        },
        benchmark: {
          output: benchmarkRes.stdout || benchmarkRes.stderr,
          passed: (benchmarkRes.stdout || "").includes("ALL 5 TESTS PASSED"),
        },
      });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // API 5: Load YAML configurations
  app.get("/api/configs", (req, res) => {
    try {
      const configFiles = ["query_parser.yaml", "retrieval.yaml", "reranking.yaml", "gazetteer.yaml"];
      const configs: Record<string, string> = {};

      for (const file of configFiles) {
        const filePath = path.join(process.cwd(), "configs", file);
        if (fs.existsSync(filePath)) {
          configs[file] = fs.readFileSync(filePath, "utf-8");
        }
      }

      res.json(configs);
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // API 6: Offline Gazetteer listing & resolution
  app.get("/api/gazetteer", async (req, res) => {
    try {
      const q = req.query.query ? String(req.query.query).replace(/"/g, '\\"') : "";
      const ctx = req.query.context ? String(req.query.context).replace(/"/g, '\\"') : "";
      const action = q ? "gazetteer_resolve" : "gazetteer_list";

      const cmd = `PYTHONPATH=. python3 scripts/test_gis.py --action ${action} --query "${q}" --context "${ctx}"`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 7: Satellite Archive Metadata Filter
  app.post("/api/metadata-filter", async (req, res) => {
    try {
      const query = (req.body.query || "").replace(/"/g, '\\"');
      const filterJson = JSON.stringify(req.body.filter || {}).replace(/"/g, '\\"');
      const cmd = `PYTHONPATH=. python3 scripts/test_gis.py --action metadata_filter --query "${query}" --filter_json "${filterJson}"`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 8: Deterministic GIS Spatial Evaluation
  app.get("/api/gis-eval", async (req, res) => {
    try {
      const cmd = `PYTHONPATH=. python3 scripts/test_gis.py --action gis_eval`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 9: Semantic Vector Search & Top-K Candidates (Vision-Language Pipeline)
  app.post("/api/semantic-candidates", async (req, res) => {
    try {
      const query = (req.body.query || "Find agricultural areas near Chennai.").replace(/"/g, '\\"');
      const topK = parseInt(req.body.top_k || "10", 10);
      const configJson = JSON.stringify(req.body.config || {}).replace(/"/g, '\\"');
      const cmd = `PYTHONPATH=. python3 scripts/test_semantic_retrieval.py --action candidates --query "${query}" --top_k ${topK} --config_json "${configJson}"`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 10: Raw ANN Vector Search on Vision-Language Hypersphere
  app.post("/api/vector-search", async (req, res) => {
    try {
      const query = (req.body.query || "industrial buildings").replace(/"/g, '\\"');
      const topK = parseInt(req.body.top_k || "10", 10);
      const cmd = `PYTHONPATH=. python3 scripts/test_semantic_retrieval.py --action search --query "${query}" --top_k ${topK}`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 11: Cross-Modal Vision-Language Alignment Inspector
  app.get("/api/embedding-compare", async (req, res) => {
    try {
      const query = (req.query.query ? String(req.query.query) : "agricultural cropland").replace(/"/g, '\\"');
      const cmd = `PYTHONPATH=. python3 scripts/test_semantic_retrieval.py --action embedding_compare --query "${query}"`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 12: Satellite Imagery Tile Archive Stats
  app.get("/api/vector-archive-stats", async (req, res) => {
    try {
      const cmd = `PYTHONPATH=. python3 scripts/test_semantic_retrieval.py --action archive_stats`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // API 13: Candidate Verification & Multi-Factor Reranking Hard-Negative Scenarios
  app.get("/api/verification-hard-negatives", async (req, res) => {
    try {
      const cmd = `PYTHONPATH=. python3 scripts/test_verification_reranking.py --action hard_negatives`;
      const { stdout } = await execAsync(cmd);
      res.json(JSON.parse(stdout));
    } catch (error: any) {
      res.status(500).json({ error: error.message, details: error.stderr || error.stdout });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Satellite Semantic Retrieval Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
