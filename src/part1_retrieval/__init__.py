"""
Part 1: Semantic Retrieval Module.

Problem ID: SIH26227 / SH227
Title: Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
Organization: Ministry of Defence | Theme: Space Technology
"""

from src.part1_retrieval.query_schema import ParsedQuery, ChangeType, SpatialRelation
from src.part1_retrieval.query_parser import QueryParser, OptionalLLMParser
from src.part1_retrieval.change_router import ChangeRouter
from src.part1_retrieval.gazetteer import Gazetteer, GazetteerEntry, ResolvedPlace, AmbiguousLocationError
from src.part1_retrieval.metadata_filter import MetadataFilter, IndexedTile
from src.part1_retrieval.vector_search import (
    VectorSearchIndex,
    FoundationVectorSearch,
    SemanticVectorSearch,
    TileSearchResult,
    VisionLanguageEmbeddingProvider,
    MockRemoteCLIPEmbeddingProvider,
    ANNBackend,
    generate_mock_archive,
)
from src.part1_retrieval.candidate_generator import (
    CandidateGenerator,
    RetrievalConfig,
    SpatialClusterer,
)
from src.part1_retrieval.gis_verifier import GISVerifier
from src.part1_retrieval.context_verifier import ContextVerifier
from src.part1_retrieval.segmentation_verifier import SegmentationVerifier
from src.part1_retrieval.reranker import CandidateReranker
from src.part1_retrieval.confidence import ConfidenceEngine
from src.part1_retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult

__all__ = [
    "ParsedQuery",
    "ChangeType",
    "SpatialRelation",
    "QueryParser",
    "OptionalLLMParser",
    "ChangeRouter",
    "Gazetteer",
    "GazetteerEntry",
    "MetadataFilter",
    "IndexedTile",
    "VectorSearchIndex",
    "FoundationVectorSearch",
    "SemanticVectorSearch",
    "TileSearchResult",
    "VisionLanguageEmbeddingProvider",
    "MockRemoteCLIPEmbeddingProvider",
    "ANNBackend",
    "generate_mock_archive",
    "CandidateGenerator",
    "RetrievalConfig",
    "SpatialClusterer",
    "GISVerifier",
    "ContextVerifier",
    "SegmentationVerifier",
    "CandidateReranker",
    "ConfidenceEngine",
    "RetrievalPipeline",
    "RetrievalResult",
]
