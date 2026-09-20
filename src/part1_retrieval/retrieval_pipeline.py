"""
End-to-End Semantic Retrieval Pipeline (Part 1 Foundation).

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.contracts.candidate_aoi import CandidateAOI
from src.part1_retrieval.query_schema import ParsedQuery
from src.part1_retrieval.query_parser import QueryParser
from src.part1_retrieval.change_router import ChangeRouter
from src.part1_retrieval.gazetteer import Gazetteer
from src.part1_retrieval.metadata_filter import MetadataFilter
from src.part1_retrieval.candidate_generator import CandidateGenerator
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.context_verifier import ContextVerifier
from src.part1_retrieval.segmentation_verifier import SegmentationVerifier
from src.part1_retrieval.reranker import CandidateReranker
from src.part1_retrieval.confidence import ConfidenceEngine
from src.part1_retrieval.vector_search import FoundationVectorSearch


@dataclass
class RetrievalResult:
    """
    Standard response contract from the Part 1 Semantic Retrieval pipeline.
    """
    parsed_query: ParsedQuery
    candidate_aois: List[CandidateAOI]
    confidence_assessment: Dict[str, Any]
    execution_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parsed_query": self.parsed_query.to_dict(),
            "candidate_aois": [c.to_dict() for c in self.candidate_aois],
            "confidence_assessment": self.confidence_assessment,
            "execution_trace": self.execution_trace,
        }


class RetrievalPipeline:
    """
    Primary orchestrator coordinating Query Parsing, Change Routing,
    Gazetteer Mapping, Candidate Generation, Multi-Stage Verification,
    Reranking, and Confidence Scoring.
    """

    def __init__(
        self,
        gazetteer: Optional[Gazetteer] = None,
        query_parser: Optional[QueryParser] = None,
        change_router: Optional[ChangeRouter] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        gis_verifier: Optional[GISVerifier] = None,
        context_verifier: Optional[ContextVerifier] = None,
        segmentation_verifier: Optional[SegmentationVerifier] = None,
        reranker: Optional[CandidateReranker] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
    ):
        self.gazetteer = gazetteer or Gazetteer()
        self.change_router = change_router or ChangeRouter()
        self.query_parser = query_parser or QueryParser(
            gazetteer=self.gazetteer,
            change_router=self.change_router,
        )
        self.metadata_filter = MetadataFilter()
        self.candidate_generator = candidate_generator or CandidateGenerator(gazetteer=self.gazetteer)
        self.gis_verifier = gis_verifier or GISVerifier()
        self.context_verifier = context_verifier or ContextVerifier()
        self.segmentation_verifier = segmentation_verifier or SegmentationVerifier()
        self.reranker = reranker or CandidateReranker()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.vector_search = FoundationVectorSearch()

    def run(self, query: str, max_candidates: int = 5) -> RetrievalResult:
        """
        Execute full Part 1 Semantic Retrieval pipeline on natural language query.
        """
        trace: List[str] = []
        trace.append(f"Stage 1 [Ingress]: Received query '{query}'")

        # Step 1: Query Parsing & Change Routing
        parsed = self.query_parser.parse(query)
        trace.append(
            f"Stage 2 [Query Parsing]: Identified targets={parsed.semantic_targets}, "
            f"locations={parsed.location_names}, temporal=({parsed.start_date} to {parsed.end_date})"
        )
        trace.append(
            f"Stage 3 [Change Router]: change_intent={parsed.change_intent}, "
            f"routed_type={parsed.requested_change_type}"
        )

        # Step 2: Candidate AOI Generation (Top-K semantic candidates)
        candidates = self.candidate_generator.generate_candidates(parsed, max_candidates=max_candidates)
        trace.append(f"Stage 4 [Candidate Generation]: Retrieved Top-{len(candidates)} semantic candidate AOIs")

        # Step 3: Multi-Scale Context Verification (Small -> Medium -> Large)
        candidates = self.context_verifier.verify_candidates(candidates, parsed)
        trace.append("Stage 5 [Multi-Scale Context Verification]: Evaluated small/medium/large spatial consistency")

        # Step 4: Structural / Segmentation Verification (Morphology, NDWI/NDVI, Roof Areas)
        candidates = self.segmentation_verifier.verify_candidates(candidates, parsed)
        trace.append("Stage 6 [Structural & Segmentation Verification]: Evaluated pixel-level footprints & spectral evidence")

        # Step 5: Deterministic GIS Verification (Projected Metric Corridor Distance & Topology)
        candidates = self.gis_verifier.verify_candidates(candidates, parsed)
        trace.append("Stage 7 [Deterministic GIS Verification]: Enforced exact metric spatial predicates (ST_DWithin/ST_Intersects)")

        # Step 6: Multi-Factor Reranking (Relevance Score Fusion & Explainability)
        candidates = self.reranker.rerank(candidates, parsed)
        trace.append("Stage 8 [Multi-Factor Reranking]: Computed relevance_score and explainable ranking breakdown")

        # Step 7: Confidence Assessment & Quality Assurance
        confidence_report = self.confidence_engine.evaluate(parsed, candidates)
        trace.append(
            f"Stage 9 [Confidence Engine]: System retrieval confidence={confidence_report['overall_confidence']}"
        )

        return RetrievalResult(
            parsed_query=parsed,
            candidate_aois=candidates,
            confidence_assessment=confidence_report,
            execution_trace=trace,
        )
