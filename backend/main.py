from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import logging
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
from backend.config import settings

# Update imports to relative paths
from .models.schemas import (
    UploadResponse, QueryRequest, QueryResponse, 
    HealthCheck, ErrorResponse, SourceDocument
)
from .file_processors.pdf_processor import PDFProcessor
from .file_processors.excel_processor import ExcelProcessor
from .file_processors.csv_processor import CSVProcessor
from .utils.chunking import TextChunker
from .services.embedding_service import EmbeddingService
from .services.vector_store import VectorStore
from .services.rag_service import SLMService, RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
pdf_processor = PDFProcessor(chunk_size=settings.CHUNK_SIZE)
excel_processor = ExcelProcessor(max_rows_per_chunk=50)
csv_processor = CSVProcessor(max_rows_per_chunk=100)

# Global state for services
embedding_service = None
vector_store = None
slm_service = None
rag_service = None
text_chunker = None

# Replace startup and shutdown events with lifespan context
@asynccontextmanager
def app_lifespan(app: FastAPI):
    logger.info("Starting Financial RAG Backend...")
    
    try:
        global embedding_service, vector_store, slm_service, rag_service, text_chunker

        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)

        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            collection_name=settings.COLLECTION_NAME,
            embedding_service=embedding_service,
            persist_directory=settings.VECTOR_DB_PATH
        )

        # Initialize SLM service
        logger.info("Initializing SLM service (this may take a minute)...")
        slm_service = SLMService(model_name=settings.SLM_MODEL)
        rag_service = RAGService(vector_store, slm_service, settings)

        # Initialize text chunker
        text_chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        logger.info("Backend initialized successfully. Debug mode: %s", settings.DEBUG)

        yield

    finally:
        logger.info("Shutting down Financial RAG Backend...")

        # Cleanup temp directory
        temp_dir = Path(settings.UPLOAD_DIR)
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")

        logger.info("Backend shutdown complete")

# Dependency injection
def get_rag_service():
    """Get RAG service instance"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return rag_service

def get_vector_store():
    """Get vector store instance"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Financial RAG API for PDF, Excel, and CSV documents",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - health check"""
    try:
        vector_db_info = vector_store.get_collection_info() if vector_store else None
        
        return HealthCheck(
            status="healthy" if embedding_service and vector_store else "degraded",
            model_loaded=slm_service is not None,
            vector_db_ready=vector_store is not None,
            total_documents=vector_db_info['document_count'] if vector_db_info else 0
        )
    except Exception:
        return HealthCheck(
            status="unhealthy",
            model_loaded=False,
            vector_db_ready=False,
            total_documents=0
        )

# Get supported file formats
@app.get("/formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB
    }

# Upload files endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(..., description="Upload PDF, Excel, or CSV files")
):
    """Upload and process multiple files"""
    start_time = datetime.now()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = 0
    all_chunks = []
    
    for file in files:
        try:
            # Validate file
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue
            
            # Validate file size
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                logger.warning(f"File too large: {file.filename} ({file_size_mb:.2f} MB)")
                continue
            
            # Create unique filename
            file_id = str(uuid.uuid4())[:8]
            safe_filename = f"{file_id}_{file.filename.replace(' ', '_')}"
            file_path = Path(settings.UPLOAD_DIR) / safe_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Processing file: {file.filename} as {safe_filename}")
            
            # Process based on file type
            if file_extension == '.pdf':
                chunks = pdf_processor.process(str(file_path))
            elif file_extension in ['.xlsx', '.xls']:
                chunks = excel_processor.process(str(file_path))
            elif file_extension == '.csv':
                chunks = csv_processor.process(str(file_path))
            else:
                continue
            
            # Apply semantic chunking
            if text_chunker:
                chunks = text_chunker.create_semantic_chunks(chunks)
            
            all_chunks.extend(chunks)
            processed_files += 1
            
            # Move to processed directory
            processed_path = Path(settings.PROCESSED_DIR) / safe_filename
            shutil.move(file_path, processed_path)
            
            logger.info(f"Processed {file.filename}: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    # Add chunks to vector store
    if all_chunks and vector_store:
        vector_store.add_documents(all_chunks)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return UploadResponse(
        status="success",
        message=f"Processed {processed_files} of {len(files)} files",
        files_processed=processed_files,
        chunks_created=len(all_chunks),
        processing_time=processing_time
    )

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query the document collection"""
    start_time = datetime.now()
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get response from RAG service
        result = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Format sources
        sources = []
        for source in result.get('sources', []):
            sources.append(SourceDocument(
                source=source.get('source', 'Unknown'),
                file_type=source.get('file_type', 'Unknown'),
                page=source.get('page'),
                sheet=source.get('sheet'),
                rows=source.get('rows'),
                content_preview=source.get('content_preview', ''),
                relevance_score=source.get('relevance_score', 0.0)
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            answer=result.get('answer', ''),
            sources=sources,
            processing_time=processing_time,
            tokens_used=result.get('tokens_used')
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Get document statistics
@app.get("/stats")
async def get_statistics(vector_store: VectorStore = Depends(get_vector_store)):
    """Get statistics about the vector store"""
    try:
        info = vector_store.get_collection_info()
        
        return {
            "collection_name": info['collection_name'],
            "document_count": info['document_count'],
            "status": info['status']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Clear documents endpoint
@app.delete("/clear")
async def clear_documents(vector_store: VectorStore = Depends(get_vector_store)):
    """Clear all documents from the vector store"""
    try:
        success = vector_store.clear_collection()
        
        if success:
            return {"status": "success", "message": "All documents cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc) if settings.DEBUG else None
        ).model_dump()
    )

# For development
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )