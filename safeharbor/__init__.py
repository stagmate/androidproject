"""SafeHarbor-RAG: a privacy-first, fully local Chroma DB retrieval layer for Yankee AI agents."""
from .privacy import disable_telemetry

disable_telemetry()
__version__ = "0.1.0"
