from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from config import settings
import logging
from time import time
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# Replace SLMService with Ollama
slm_service = OllamaLLM(
    model=settings.SLM_MODEL,
    base_url=settings.OLLAMA_API_URL
)


class TokenTrackingCallback(BaseCallbackHandler):
    """Callback handler for tracking token usage and logging"""
    
    def __init__(self, query: str = None):
        super().__init__()
        self.query = query
        self.start_time = time()
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts processing"""
        logger.debug(f"[LLM Start] Query: {self.query}")
        logger.debug(f"[LLM Start] Prompts: {prompts[:100]}..." if len(str(prompts)) > 100 else f"[LLM Start] Prompts: {prompts}")
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes processing"""
        self.call_count += 1
        elapsed_time = time() - self.start_time
        logger.info(f"[LLM End] Call #{self.call_count} completed in {elapsed_time:.2f}s for query: {self.query}")
        
        # Extract token information if available
        if response.llm_output:
            if isinstance(response.llm_output, dict):
                self.tokens_used = response.llm_output.get('token_usage', {}).get('total_tokens', 0)
                self.prompt_tokens = response.llm_output.get('token_usage', {}).get('prompt_tokens', 0)
                self.completion_tokens = response.llm_output.get('token_usage', {}).get('completion_tokens', 0)
                
                if self.tokens_used > 0:
                    logger.info(f"[Token Usage] Total: {self.tokens_used}, Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        elapsed_time = time() - self.start_time
        logger.error(f"[LLM Error] After {elapsed_time:.2f}s - {str(error)}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain starts"""
        logger.debug(f"[Chain Start] Inputs: {list(inputs.keys())}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends"""
        elapsed_time = time() - self.start_time
        logger.info(f"[Chain End] Completed in {elapsed_time:.2f}s")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain encounters an error"""
        elapsed_time = time() - self.start_time
        logger.error(f"[Chain Error] After {elapsed_time:.2f}s - {str(error)}")
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> None:
        """Called when retriever starts"""
        logger.debug(f"[Retriever Start] Query: {query}")
    
    def on_retriever_end(self, documents, **kwargs: Any) -> None:
        """Called when retriever ends"""
        logger.debug(f"[Retriever End] Retrieved {len(documents)} documents")


class RAGService:
    def __init__(self, vector_store, llm, config=None):
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or settings
        
        # Use the structured prompt template from config
        self.prompt_template = self.config.get_prompt_template()
        
        # Create chain with prompt template
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RESULTS}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "document_variable_name": "context"
            }
        )
        
        logger.info("RAGService initialized with prompt template and token tracking")

    def query(self, query: str, top_k: Optional[int] = None, 
              similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Process query and generate response with callbacks"""
        start_time = time()
        
        # Use provided values or defaults
        top_k = top_k or self.config.TOP_K_RESULTS
        similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
        
        # Create callback handler for this query
        callback = TokenTrackingCallback(query=query)
        
        try:
            logger.info(f"Starting query processing: '{query}' with top_k={top_k}")
            
            # Invoke the chain with callbacks
            result = self.chain.invoke(
                {"query": query},
                config={"callbacks": [callback]}
            )

            # Extract source documents and metadata
            source_documents = result.get("source_documents", [])
            
            logger.debug(f"Retrieved {len(source_documents)} source documents")
            
            # Filter results by similarity threshold
            filtered_sources = []
            for doc in source_documents:
                # Get relevance score from metadata or calculate based on retrieval
                relevance_score = doc.metadata.get("relevance_score", 0.9)
                
                if relevance_score >= similarity_threshold:
                    filtered_sources.append({
                        "content": doc.page_content[:500],  # Preview first 500 chars
                        "metadata": doc.metadata,
                        "relevance_score": relevance_score
                    })
            
            elapsed_time = time() - start_time
            
            response = {
                "answer": result.get("result", "No answer generated"),
                "sources": filtered_sources,
                "confidence": max([src["relevance_score"] for src in filtered_sources], default=0.0),
                "tokens_used": callback.tokens_used,
                "processing_time": elapsed_time,
                "callback_info": {
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens,
                    "llm_calls": callback.call_count
                }
            }
            
            logger.info(f"Query completed in {elapsed_time:.2f}s - {len(filtered_sources)} sources retrieved")
            return response
        
        except Exception as e:
            elapsed_time = time() - start_time
            logger.error(f"Error processing query after {elapsed_time:.2f}s: {str(e)}", exc_info=True)
            raise