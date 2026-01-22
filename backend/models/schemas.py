from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class FileInfo(BaseModel):
    filename: str
    content_type: str
    size: int

class UploadResponse(BaseModel):
    status: str
    message: str
    files_processed: int
    chunks_created: int
    processing_time: float

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None

class SourceDocument(BaseModel):
    source: str
    file_type: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    rows: Optional[str] = None
    content_preview: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    processing_time: float
    tokens_used: Optional[int] = None

class HealthCheck(BaseModel):
    status: str
    is_model_loader: bool
    vector_db_ready: bool
    total_documents: int

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None