import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate

class Settings:
    # App settings
    APP_NAME: str = "Financial RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # File settings
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = ['.pdf', '.xlsx', '.xls', '.csv']
    UPLOAD_DIR: str = "temp/uploads"
    PROCESSED_DIR: str = "temp/processed"

    # Ollama API settings
    OLLAMA_API_URL: str = "http://localhost:11434"
    OLLAMA_API_KEY: str = ""  # Optional, leave empty if not required

    # Model settings (Ollama model names)
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Ollama embedding model
    SLM_MODEL: str = "llama2:7b"  # Ollama text generation model

    # Vector store settings
    VECTOR_DB_PATH: str = "data/vector_db"
    COLLECTION_NAME: str = "financial_documents"

    # Chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # Structured Prompt Template for RetrievalQA
    PROMPT_TEMPLATE_STR: str = """You are a financial document analysis assistant. Your role is to provide accurate, 
helpful answers based on the provided financial documents. Always cite your sources and be transparent about 
the confidence level of your answers.

Financial Context:
{context}

Question: {question}

Instructions:
- Provide a clear, concise answer based on the context provided
- If the context does not contain sufficient information to answer the question, explicitly state this
- Cite specific sections or pages from the documents when possible
- Maintain a professional tone appropriate for financial analysis
- If there are any ambiguities or caveats, mention them

Answer:"""

    # Lazy-loaded prompt template instance
    _prompt_template_instance = None

    def __init__(self):
        # Load from environment variables if available
        self.DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', self.EMBEDDING_MODEL)
        self.SLM_MODEL = os.getenv('SLM_MODEL', self.SLM_MODEL)
        self.VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', self.VECTOR_DB_PATH)
        self.COLLECTION_NAME = os.getenv('COLLECTION_NAME', self.COLLECTION_NAME)
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', self.CHUNK_OVERLAP))
        self.TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', self.TOP_K_RESULTS))
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', self.SIMILARITY_THRESHOLD))

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

settings = Settings()

# Create directories
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
