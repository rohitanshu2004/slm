"""
Custom exceptions for the Financial RAG API.
Provides structured error handling throughout the application.
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """Base exception for all RAG-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


class ValidationException(RAGException):
    """Raised when input validation fails"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if field:
            full_details["field"] = field
        if value is not None:
            full_details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=full_details,
            status_code=400
        )


class FileProcessingException(RAGException):
    """Raised when file processing fails"""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if filename:
            full_details["filename"] = filename
        if file_type:
            full_details["file_type"] = file_type
        
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            details=full_details,
            status_code=422
        )


class VectorStoreException(RAGException):
    """Raised when vector store operations fail"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if operation:
            full_details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            details=full_details,
            status_code=500
        )


class RAGQueryException(RAGException):
    """Raised when RAG query processing fails"""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if query:
            full_details["query"] = query[:100] + "..." if len(query) > 100 else query
        
        super().__init__(
            message=message,
            error_code="RAG_QUERY_ERROR",
            details=full_details,
            status_code=500
        )


class OllamaException(RAGException):
    """Raised when Ollama API calls fail"""
    
    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if api_endpoint:
            full_details["api_endpoint"] = api_endpoint
        if model:
            full_details["model"] = model
        
        super().__init__(
            message=message,
            error_code="OLLAMA_API_ERROR",
            details=full_details,
            status_code=503
        )


class ServiceInitializationException(RAGException):
    """Raised when service initialization fails"""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if service_name:
            full_details["service"] = service_name
        
        super().__init__(
            message=message,
            error_code="SERVICE_INITIALIZATION_ERROR",
            details=full_details,
            status_code=503
        )


class NoDocumentsException(RAGException):
    """Raised when no documents are available for querying"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="No documents available in vector store. Please upload documents first.",
            error_code="NO_DOCUMENTS",
            details=details or {},
            status_code=404
        )


class EmbeddingException(RAGException):
    """Raised when embedding operations fail"""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        full_details = details or {}
        if model:
            full_details["model"] = model
        
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            details=full_details,
            status_code=500
        )
