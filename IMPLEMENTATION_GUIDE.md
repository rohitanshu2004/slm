# Financial RAG API - Implementation Guide

This document describes the newly implemented features and improvements to the Financial RAG system.

## 📋 New Features

### 1. Structured Prompt Templates (config.py)

#### What's New
- Added structured `PromptTemplate` from LangChain with professional financial analysis instructions
- Template includes system message, context formatting, and query instructions
- Lazy-loaded prompt template instance for optimal performance

#### Key Features
- **Professional tone** appropriate for financial documents
- **Source citation** instructions for transparency
- **Confidence guidance** for nuanced answers
- **Ambiguity handling** for complex financial queries

#### Usage
```python
from backend.config import settings

# Get the prompt template
prompt = settings.get_prompt_template()

# Template variables: context, question
# Automatically used in RAGService
```

#### Configuration (config.py)
```python
PROMPT_TEMPLATE_STR: str = """You are a financial document analysis assistant..."""

def get_prompt_template(self) -> PromptTemplate:
    """Get or create the prompt template instance"""
    if self._prompt_template_instance is None:
        self._prompt_template_instance = PromptTemplate(
            input_variables=["context", "question"],
            template=self.PROMPT_TEMPLATE_STR,
            template_format="f-string",
            validate_template=True
        )
    return self._prompt_template_instance
```

---

### 2. LangChain Callbacks for Logging & Token Usage (rag_service.py)

#### What's New
- Implemented custom `TokenTrackingCallback` using LangChain's `BaseCallbackHandler`
- Automatic tracking of token usage (prompt tokens, completion tokens, total tokens)
- Comprehensive logging at each stage of the RAG pipeline

#### Key Features
- **Token Usage Tracking**: Automatic collection of token consumption metrics
- **Performance Monitoring**: Elapsed time tracking for each LLM call
- **Multi-level Logging**: Debug, info, and error level logging
- **Error Tracking**: Detailed error logging with context

#### Callback Hooks Implemented
```python
- on_llm_start()      # LLM processing starts
- on_llm_end()        # LLM processing completes
- on_llm_error()      # LLM error occurs
- on_chain_start()    # Chain processing starts
- on_chain_end()      # Chain processing completes
- on_chain_error()    # Chain error occurs
- on_retriever_start() # Retriever search starts
- on_retriever_end()   # Retriever search completes
```

#### Response Structure
```python
{
    "answer": "The answer to the query...",
    "sources": [...],
    "confidence": 0.95,
    "tokens_used": 1250,
    "processing_time": 2.34,
    "callback_info": {
        "prompt_tokens": 500,
        "completion_tokens": 750,
        "llm_calls": 1
    }
}
```

#### Example Logging Output
```
[LLM Start] Query: What is the profit margin of TechCorp?
[LLM End] Call #1 completed in 2.15s for query: What is the profit margin of TechCorp?
[Token Usage] Total: 1250, Prompt: 500, Completion: 750
[Chain End] Completed in 2.34s
[Retriever End] Retrieved 5 documents
```

---

### 3. Custom Exceptions & Error Handling (exceptions.py)

#### New Exception Classes

All custom exceptions inherit from `RAGException` and provide structured error responses.

##### Exception Hierarchy
```
RAGException (base)
├── ValidationException
├── FileProcessingException
├── VectorStoreException
├── RAGQueryException
├── OllamaException
├── ServiceInitializationException
├── NoDocumentsException
└── EmbeddingException
```

#### Exception Features
- **Structured responses** with error codes, messages, and details
- **HTTP status codes** automatically set appropriately
- **Contextual information** captured in details field
- **JSON serializable** via `to_dict()` method

#### Exception Usage Examples

**ValidationException** - Input validation errors
```python
from backend.exceptions import ValidationException

raise ValidationException(
    message="Query cannot be empty",
    field="query"
)
# Response: 400 Bad Request
```

**FileProcessingException** - File processing errors
```python
raise FileProcessingException(
    message="Failed to process CSV file",
    filename="data.csv",
    file_type=".csv",
    details={"error": str(e)}
)
# Response: 422 Unprocessable Entity
```

**VectorStoreException** - Vector store operation errors
```python
raise VectorStoreException(
    message="Failed to add documents to vector store",
    operation="add_documents",
    details={"document_count": 100, "error": str(e)}
)
# Response: 500 Internal Server Error
```

**OllamaException** - Ollama API errors
```python
raise OllamaException(
    message="Cannot connect to Ollama API",
    api_endpoint="http://localhost:11434",
    details={"error": str(e)}
)
# Response: 503 Service Unavailable
```

**RAGQueryException** - Query processing errors
```python
raise RAGQueryException(
    message="Failed to process query",
    query="Sample query",
    details={"error": str(e)}
)
# Response: 500 Internal Server Error
```

**NoDocumentsException** - Empty vector store
```python
raise NoDocumentsException()
# Response: 404 Not Found
# Message: "No documents available in vector store. Please upload documents first."
```

---

### 4. Enhanced Error Handling in Endpoints (main.py)

#### Improvements Made

**Upload Endpoint (`/upload`)**
- ✅ File validation (type, size, extension)
- ✅ Comprehensive error handling with detailed messages
- ✅ Graceful failure handling per file
- ✅ Vector store insertion error handling

**Query Endpoint (`/query`)**
- ✅ Empty query validation
- ✅ No documents check (404 response)
- ✅ Query processing error handling
- ✅ Source formatting error handling
- ✅ Token usage tracking in response

**Stats Endpoint (`/stats`)**
- ✅ Vector store health check
- ✅ Detailed error information in debug mode

**Clear Endpoint (`/delete`)**
- ✅ Safe clearing with error handling
- ✅ Success/failure feedback

**Health Check Endpoint (`/`)**
- ✅ Checks embedding service status
- ✅ Checks vector store status
- ✅ Returns document count
- ✅ Overall system status

#### Service Initialization Error Handling
- ✅ Ollama model verification with specific error messages
- ✅ Ollama health check
- ✅ Embedding service initialization
- ✅ Vector store initialization
- ✅ RAG service initialization

#### Example Error Response Structure
```json
{
    "error": "VALIDATION_ERROR",
    "message": "Query cannot be empty",
    "details": {
        "field": "query"
    },
    "status_code": 400
}
```

---

### 5. Enhanced Frontend with Error Handling (frontend/app.py)

#### New Features

**API Health Checking**
- ✅ Automatic health check on startup
- ✅ User-friendly error messages if API is unavailable
- ✅ Clear instructions for starting services

**File Upload Improvements**
- ✅ File size validation before upload
- ✅ Format validation
- ✅ Progress indication with spinner
- ✅ Detailed upload summary

**Query Error Handling**
- ✅ Connection error detection
- ✅ Timeout handling
- ✅ User-friendly error messages
- ✅ Suggestion for resolution

**Source Display Enhancements**
- ✅ Better formatting
- ✅ Expandable sections
- ✅ Relevance score display
- ✅ Safe rendering with error handling

**Advanced Settings**
- ✅ Top-K results slider
- ✅ Similarity threshold control
- ✅ Expandable advanced options

**Sidebar Features**
- ✅ Supported format information
- ✅ Max file size display
- ✅ Real-time vector store statistics
- ✅ Refresh statistics button

#### User-Friendly Error Messages
```
⏱️ Request timed out. The file might be too large or the server is slow.
❌ Cannot connect to the API server. Make sure it's running on http://localhost:8000
📤 Please upload documents first using the sidebar to start asking questions.
ℹ️ No relevant sources found for this query.
```

---

## 🧪 Testing

### Complete Pipeline Test Script

A comprehensive test script (`test_pipeline.py`) is provided to verify the entire workflow.

#### Running Tests
```bash
# From the project root directory
python test_pipeline.py

# Or with custom API URL
python test_pipeline.py http://custom-host:port
```

#### Test Cases Included

1. **API Health Check** - Verify API connectivity
2. **Supported Formats** - List accepted file formats
3. **Vector Store Stats (Before Upload)** - Initial state
4. **Query Without Documents** - Graceful failure handling
5. **File Upload** - Upload and process test CSV
6. **Vector Store Stats (After Upload)** - Post-upload state
7. **Query with Documents** - Multiple query tests
8. **Empty Query Validation** - Input validation
9. **Clear Vector Store** - Document cleanup

#### Test Output Example
```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   FINANCIAL RAG API - FULL PIPELINE TEST                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

============================================================
TEST 1: API Health Check
============================================================
✅ API is healthy
   Status: healthy
   Model loaded: True
   Vector DB ready: True
   Documents: 5

...

============================================================
TEST SUMMARY
============================================================
✅ PASS: API Health Check
✅ PASS: Supported Formats
✅ PASS: Stats (Before Upload)
✅ PASS: Query Without Documents
✅ PASS: File Upload
✅ PASS: Stats (After Upload)
✅ PASS: Query Documents
✅ PASS: Empty Query Validation
✅ PASS: Clear Vector Store
============================================================
Total: 9/9 tests passed (100.0%)
============================================================
```

---

## 📚 Usage Guide

### Starting the System

1. **Start Ollama** (in a separate terminal)
```bash
ollama serve
```

2. **Start Backend API** (from project root)
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Start Frontend** (in a separate terminal)
```bash
streamlit run frontend/app.py
```

### Workflow

1. **Upload Documents**
   - Navigate to sidebar
   - Select files (PDF, XLSX, XLS, CSV)
   - Click "Process Documents"
   - Wait for processing to complete

2. **Ask Questions**
   - Type your question in the chat input
   - Review the AI response
   - Check sources if needed
   - Continue the conversation

3. **Manage Documents**
   - View statistics in sidebar
   - Clear documents when needed
   - Upload new documents anytime

### API Endpoints

#### Health Check
```
GET /
Response: HealthCheck model with status, model_loaded, vector_db_ready, total_documents
```

#### Get Supported Formats
```
GET /formats
Response: { supported_formats: [...], max_file_size_mb: 50 }
```

#### Upload Files
```
POST /upload
Body: multipart/form-data with files
Response: UploadResponse with processing details
```

#### Query Documents
```
POST /query
Body: { query: str, top_k?: int, similarity_threshold?: float }
Response: QueryResponse with answer, sources, processing_time, tokens_used
```

#### Get Statistics
```
GET /stats
Response: { collection_name, document_count, status }
```

#### Clear Documents
```
DELETE /clear
Response: { status, message }
```

---

## 🔍 Logging & Debugging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General informational messages
- **WARNING**: Warning messages (recoverable issues)
- **ERROR**: Error messages (failures)

### Enabling Debug Mode
Edit `backend/config.py`:
```python
DEBUG: bool = True  # Enable detailed logging
```

### Example Debug Output
```
2026-01-22 10:30:45 - backend.services.rag_service - DEBUG - [Chain Start] Inputs: dict_keys(['query'])
2026-01-22 10:30:45 - backend.services.rag_service - DEBUG - [Retriever Start] Query: What is the profit?
2026-01-22 10:30:46 - backend.services.rag_service - DEBUG - [Retriever End] Retrieved 5 documents
2026-01-22 10:30:47 - backend.services.rag_service - INFO - [Token Usage] Total: 1250, Prompt: 500, Completion: 750
2026-01-22 10:30:47 - backend.services.rag_service - INFO - Query completed in 2.34s - 5 sources retrieved
```

---

## 🐛 Troubleshooting

### Issue: API returns 503 Service Unavailable
**Solution**: Check that Ollama is running and models are available
```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama2:7b
```

### Issue: Query returns "No documents available"
**Solution**: Upload documents first using the sidebar

### Issue: File upload fails
**Possible causes**:
- File is too large (max 50MB)
- Invalid file format (only PDF, XLSX, XLS, CSV supported)
- Vector store is not initialized

**Solution**: Check API logs and try a smaller file

### Issue: Queries are slow
**Possible causes**:
- Large number of documents
- Complex document content
- Ollama model is slow

**Solution**: Adjust similarity_threshold to reduce processing, use fewer top_k results

---

## 📊 Performance Notes

- **Token Usage**: Tracked automatically for each query
- **Processing Time**: Included in all responses
- **Callback Info**: Available for debugging LLM behavior
- **Logging**: All operations logged for audit trail

---

## 🔐 Security Notes

- Input validation on all endpoints
- File type validation
- File size limits
- Debug mode should be disabled in production
- Error details hidden from users in production mode

---

## 📝 Configuration Reference

### Key Settings (backend/config.py)
```python
# Model settings
EMBEDDING_MODEL: str = "nomic-embed-text"
SLM_MODEL: str = "llama2:7b"

# File settings
MAX_FILE_SIZE_MB: int = 50
ALLOWED_EXTENSIONS: list = ['.pdf', '.xlsx', '.xls', '.csv']

# Chunking
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Retrieval
TOP_K_RESULTS: int = 5
SIMILARITY_THRESHOLD: float = 0.7

# Debug
DEBUG: bool = True
```

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in backend output
3. Run the test pipeline to diagnose issues
4. Check API health endpoint (`GET /`)

---

Last Updated: January 22, 2026
