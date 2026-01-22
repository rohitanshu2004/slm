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
import sys
import os

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

# Update imports to absolute paths
from models.schemas import (
    UploadResponse, QueryRequest, QueryResponse,
    HealthCheck, ErrorResponse, SourceDocument
)
from file_processors.pdf_processor import PDFProcessor
from file_processors.excel_processor import ExcelProcessor
from file_processors.csv_processor import CSVProcessor
from utils.chunking import TextChunker
from services.vector_store import VectorStore
from services.rag_service import RAGService
from exceptions import (
    RAGException, ValidationException, FileProcessingException,
    VectorStoreException, RAGQueryException, OllamaException,
    ServiceInitializationException, NoDocumentsException, EmbeddingException
)

# Verify LangChain imports for Ollama integration
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.documents import Document

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

# Global state for services - ALL INITIALIZED LAZILY
_embeddings = None
_vector_store = None
_llm = None
_rag_service = None
_text_chunker = None
_initialization_lock = asyncio.Lock()
_is_initializing = False
_initialization_error = None

# ============================================================================
# LAZY INITIALIZATION FUNCTIONS
# ============================================================================

async def verify_ollama_models():
    """Verify that required Ollama models are available"""
    import requests

    required_models = {
        'embedding': settings.EMBEDDING_MODEL,
        'generation': settings.SLM_MODEL
    }

    try:
        # Check if Ollama is running
        response = requests.get(f"{settings.OLLAMA_API_URL}/api/tags", timeout=5)
        response.raise_for_status()

        available_models = response.json().get('models', [])
        available_model_names = [model['name'] for model in available_models]

        # Check each required model
        missing_models = []
        for model_type, model_name in required_models.items():
            if not any(model_name in name for name in available_model_names):
                missing_models.append(f"{model_name} ({model_type})")
                logger.warning(f"Required {model_type} model '{model_name}' not found")

        if missing_models:
            error_msg = (
                f"Missing required Ollama models: {', '.join(missing_models)}. "
                f"Please run: ollama pull {settings.EMBEDDING_MODEL} && ollama pull {settings.SLM_MODEL}"
            )
            logger.error(error_msg)
            raise OllamaException(
                message="Required Ollama models not found",
                api_endpoint=settings.OLLAMA_API_URL,
                details={"missing_models": missing_models}
            )

        logger.info(f"✓ Verified embedding model: {settings.EMBEDDING_MODEL}")
        logger.info(f"✓ Verified generation model: {settings.SLM_MODEL}")
        return True

    except requests.exceptions.ConnectionError as e:
        error_msg = (
            "Cannot connect to Ollama API. Please ensure Ollama is running. "
            "Start it with: ollama serve"
        )
        logger.error(error_msg)
        raise OllamaException(
            message="Cannot connect to Ollama API",
            api_endpoint=settings.OLLAMA_API_URL,
            details={"error": str(e)}
        )
    except OllamaException:
        raise
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        raise ServiceInitializationException(
            message="Failed to verify Ollama models",
            service_name="Ollama",
            details={"error": str(e)}
        )

async def _initialize_services():
    """Initialize all services lazily - called only when needed"""
    global _embeddings, _vector_store, _llm, _rag_service, _text_chunker, _is_initializing, _initialization_error
    
    async with _initialization_lock:
        if _rag_service is not None:
            return  # Already initialized
        
        if _is_initializing:
            # Wait for initialization to complete
            await asyncio.sleep(0.1)
            return
        
        _is_initializing = True
        _initialization_error = None
        
        try:
            # Verify Ollama models are available
            logger.info("Verifying Ollama models...")
            await verify_ollama_models()

            # Check Ollama health
            logger.info("Checking Ollama health...")
            try:
                import requests
                response = requests.get(f"{settings.OLLAMA_API_URL}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise OllamaException(
                        message="Ollama is not healthy",
                        api_endpoint=settings.OLLAMA_API_URL
                    )
            except requests.exceptions.RequestException as e:
                raise OllamaException(
                    message="Ollama health check failed",
                    api_endpoint=settings.OLLAMA_API_URL,
                    details={"error": str(e)}
                )

            # Initialize embeddings using LangChain
            logger.info("Initializing embeddings...")
            try:
                _embeddings = OllamaEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    base_url=settings.OLLAMA_API_URL
                )
                
                # Test embeddings with a simple query
                test_embedding = _embeddings.embed_query("test")
                if test_embedding:
                    logger.info(f"Embeddings initialized successfully. Vector dimension: {len(test_embedding)}")
                else:
                    raise EmbeddingException(
                        message="Embeddings returned empty result",
                        model=settings.EMBEDDING_MODEL
                    )
                    
            except Exception as e:
                raise EmbeddingException(
                    message="Failed to initialize embeddings",
                    model=settings.EMBEDDING_MODEL,
                    details={"error": str(e)}
                )

            # Initialize vector store
            logger.info("Initializing vector store...")
            try:
                _vector_store = VectorStore(
                    collection_name=settings.COLLECTION_NAME,
                    persist_directory=settings.VECTOR_DB_PATH
                )
                
                # Set embeddings on vector store if needed
                if hasattr(_vector_store, 'set_embeddings'):
                    logger.info("Setting embeddings on vector store...")
                    _vector_store.set_embeddings(_embeddings)
                elif hasattr(_vector_store, 'embeddings'):
                    logger.info("Setting embeddings attribute on vector store...")
                    _vector_store.embeddings = _embeddings
                    
            except Exception as e:
                raise VectorStoreException(
                    message="Failed to initialize vector store",
                    operation="initialization",
                    details={"error": str(e)}
                )

            # Initialize LLM service using LangChain
            logger.info("Initializing LLM service (this may take a minute)...")
            try:
                _llm = OllamaLLM(
                    model=settings.SLM_MODEL,
                    base_url=settings.OLLAMA_API_URL,
                    temperature=0.1,  # Lower temperature for more factual responses
                    num_predict=512  # Limit response length
                )
                
                # Test LLM with a simple query
                test_response = _llm.invoke("Hello")
                if test_response:
                    logger.info("LLM initialized successfully")
                else:
                    raise ServiceInitializationException(
                        message="LLM returned empty response",
                        service_name="LLM"
                    )
                    
            except Exception as e:
                raise ServiceInitializationException(
                    message="Failed to initialize LLM service",
                    service_name="LLM",
                    details={"error": str(e)}
                )

            # Initialize RAG service
            logger.info("Initializing RAG service...")
            try:
                _rag_service = RAGService(
                    vector_store=_vector_store,
                    llm=_llm,
                    config=settings
                )
            except Exception as e:
                raise ServiceInitializationException(
                    message="Failed to initialize RAG service",
                    service_name="RAGService",
                    details={"error": str(e)}
                )

            # Initialize text chunker
            _text_chunker = TextChunker(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            logger.info("All services initialized successfully")

        except Exception as e:
            _initialization_error = e
            logger.error(f"Service initialization failed: {e}", exc_info=True)
            raise
        finally:
            _is_initializing = False

# ============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# ============================================================================

async def get_rag_service():
    """Get RAG service instance with lazy initialization"""
    global _rag_service, _initialization_error
    
    if _rag_service is None:
        try:
            await _initialize_services()
        except Exception as e:
            if _initialization_error:
                e = _initialization_error
            if isinstance(e, RAGException):
                raise HTTPException(status_code=e.status_code, detail=e.to_dict())
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": "SERVICE_UNAVAILABLE",
                    "message": "RAG service initialization failed",
                    "details": str(e) if settings.DEBUG else None
                }
            )
    
    if _rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    return _rag_service

async def get_vector_store():
    """Get vector store instance with lazy initialization"""
    global _vector_store, _initialization_error
    
    if _vector_store is None:
        try:
            await _initialize_services()
        except Exception as e:
            if _initialization_error:
                e = _initialization_error
            if isinstance(e, RAGException):
                raise HTTPException(status_code=e.status_code, detail=e.to_dict())
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": "SERVICE_UNAVAILABLE",
                    "message": "Vector store initialization failed",
                    "details": str(e) if settings.DEBUG else None
                }
            )
    
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    return _vector_store

def get_text_chunker():
    """Get text chunker instance"""
    global _text_chunker
    return _text_chunker

# ============================================================================
# FASTAPI LIFESPAN (MINIMAL)
# ============================================================================

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Minimal lifespan - only sets up logging and basic checks"""
    logger.info("Starting Financial RAG Backend (lazy initialization)...")
    
    # Ensure required directories exist
    for dir_path in [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.VECTOR_DB_PATH]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Backend startup complete. Services will initialize on first use.")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    yield
    
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

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

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

# ============================================================================
# INSTANT HEALTH ENDPOINT (CRITICAL FIX)
# ============================================================================

@app.get("/health")
async def instant_health_check():
    """Instant health check - does NOT initialize services"""
    return {"status": "ok"}

@app.get("/", response_model=HealthCheck)
async def detailed_health_check():
    """Detailed health check with service status"""
    try:
        # Check services without initializing them
        embedding_status = "not_initialized" if _embeddings is None else "healthy"
        vector_store_status = "not_initialized" if _vector_store is None else "healthy"
        rag_status = "not_initialized" if _rag_service is None else "healthy"
        llm_status = "not_initialized" if _llm is None else "healthy"
        
        # Try to get document count if vector store exists
        doc_count = 0
        if _vector_store is not None:
            try:
                info = _vector_store.get_collection_info()
                doc_count = info.get('document_count', 0)
            except:
                vector_store_status = "error"
        
        return HealthCheck(
            status="healthy" if all(s in ["healthy"] for s in [embedding_status, vector_store_status, rag_status, llm_status]) else "degraded",
            is_model_loader=_rag_service is not None,
            vector_db_ready=_vector_store is not None,
            total_documents=doc_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "unhealthy",
                "is_model_loader": False,
                "vector_db_ready": False,
                "total_documents": 0
            },
            status_code=503
        )

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
    
    try:
        if not files:
            raise ValidationException(
                message="No files provided",
                field="files"
            )
        
        logger.info(f"Processing {len(files)} uploaded files")
        processed_files = 0
        all_chunks = []
        all_docs = []
        
        # Get text chunker (doesn't require heavy initialization)
        text_chunker = get_text_chunker()
        
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
                    raise FileProcessingException(
                        message=f"File size exceeds maximum allowed size",
                        filename=file.filename,
                        file_type=file_extension,
                        details={"max_size_mb": settings.MAX_FILE_SIZE_MB, "actual_size_mb": file_size_mb}
                    )
                
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
                try:
                    if file_extension == '.pdf':
                        chunks = pdf_processor.process(str(file_path))
                    elif file_extension in ['.xlsx', '.xls']:
                        chunks = excel_processor.process(str(file_path))
                    elif file_extension == '.csv':
                        chunks = csv_processor.process(str(file_path))
                    else:
                        continue
                except Exception as e:
                    raise FileProcessingException(
                        message=f"Failed to process {file_extension} file",
                        filename=file.filename,
                        file_type=file_extension,
                        details={"error": str(e)}
                    )
                
                # Apply semantic chunking
                if text_chunker:
                    try:
                        chunks = text_chunker.create_semantic_chunks(chunks)
                    except Exception as e:
                        logger.warning(f"Semantic chunking failed, using original chunks: {e}")
                
                # Convert chunks to Document objects
                for chunk in chunks:
                    if isinstance(chunk, dict) and 'content' in chunk:
                        metadata = chunk.get('metadata', {})
                        metadata['filename'] = safe_filename
                        metadata['original_filename'] = file.filename
                        metadata['file_type'] = file_extension.lstrip('.')
                        metadata['processed_time'] = datetime.now().isoformat()
                        
                        doc = Document(
                            page_content=chunk['content'],
                            metadata=metadata
                        )
                        all_docs.append(doc)
                        all_chunks.append(chunk)
                
                processed_files += 1
                logger.info(f"Processed {file.filename}: {len(chunks)} chunks created")
                
                # Move to processed directory
                processed_path = Path(settings.PROCESSED_DIR) / safe_filename
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(processed_path))
                
            except FileProcessingException:
                raise
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
                raise FileProcessingException(
                    message=f"Unexpected error processing file",
                    filename=file.filename,
                    details={"error": str(e)}
                )
        
        if processed_files == 0:
            raise ValidationException(
                message="No valid files could be processed",
                field="files"
            )
        
        # Initialize vector store if needed and add documents
        if all_docs:
            vector_store = await get_vector_store()
            try:
                vector_store.add_documents(all_docs)
                logger.info(f"Added {len(all_docs)} documents to vector store")
            except Exception as e:
                raise VectorStoreException(
                    message="Failed to add documents to vector store",
                    operation="add_documents",
                    details={"error": str(e), "document_count": len(all_docs)}
                )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return UploadResponse(
            status="success",
            message=f"Processed {processed_files} of {len(files)} files",
            files_processed=processed_files,
            chunks_created=len(all_chunks),
            processing_time=processing_time
        )
    
    except RAGException as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "UPLOAD_ERROR",
            "message": "Unexpected error during file upload",
            "details": str(e) if settings.DEBUG else None
        })

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest
):
    """Query the document collection"""
    start_time = datetime.now()
    
    try:
        if not request.query or not request.query.strip():
            raise ValidationException(
                message="Query cannot be empty",
                field="query"
            )
        
        logger.info(f"Processing query: {request.query[:100]}")
        
        # Get RAG service with lazy initialization
        rag_service = await get_rag_service()
        
        # Check if documents exist in vector store
        try:
            vector_store = await get_vector_store()
            info = vector_store.get_collection_info()
            if info.get('document_count', 0) == 0:
                raise NoDocumentsException()
        except NoDocumentsException:
            raise
        except Exception as e:
            logger.warning(f"Could not verify document count: {e}")
        
        # Get response from RAG service
        try:
            result = rag_service.query(
                query=request.query,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )
        except Exception as e:
            raise RAGQueryException(
                message="Failed to process query",
                query=request.query,
                details={"error": str(e)}
            )
        
        # Format sources
        sources = []
        for source in result.get('sources', []):
            try:
                sources.append(SourceDocument(
                    source=source.get('source', 'Unknown'),
                    file_type=source.get('file_type', 'Unknown'),
                    page=source.get('page'),
                    sheet=source.get('sheet'),
                    rows=source.get('rows'),
                    content_preview=source.get('content', '')[:500],
                    relevance_score=source.get('relevance_score', 0.0)
                ))
            except Exception as e:
                logger.warning(f"Failed to format source: {e}")
                continue
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            answer=result.get('answer', 'No answer could be generated'),
            sources=sources, 
            processing_time=processing_time,
            tokens_used=result.get('tokens_used')
        )
        
    except NoDocumentsException as e:
        logger.warning(f"Query failed: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except (ValidationException, RAGQueryException) as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "QUERY_ERROR",
            "message": "Unexpected error processing query",
            "details": str(e) if settings.DEBUG else None
        })

# Get document statistics
@app.get("/stats")
async def get_statistics():
    """Get statistics about the vector store"""
    try:
        vector_store = await get_vector_store()
        info = vector_store.get_collection_info()
        
        return {
            "collection_name": info['collection_name'],
            "document_count": info['document_count'],
            "status": info['status']
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "STATS_ERROR",
            "message": "Failed to retrieve statistics",
            "details": str(e) if settings.DEBUG else None
        })

# Clear documents endpoint
@app.delete("/clear")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        logger.info("Clearing all documents from vector store")
        vector_store = await get_vector_store()
        success = vector_store.clear_collection()
        
        if success:
            logger.info("Successfully cleared vector store")
            return {"status": "success", "message": "All documents cleared"}
        else:
            raise VectorStoreException(
                message="Failed to clear documents from vector store",
                operation="clear_collection"
            )
    except VectorStoreException as e:
        logger.error(f"Clear operation failed: {e}")
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error during clear: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "CLEAR_ERROR",
            "message": "Failed to clear documents",
            "details": str(e) if settings.DEBUG else None
        })

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if isinstance(exc, RAGException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.DEBUG else None
        }
    )

# For development
if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    for dir_path in [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.VECTOR_DB_PATH]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )