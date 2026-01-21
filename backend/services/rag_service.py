from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Dict, Any, Optional
import logging
from time import time
from ..config import settings

logger = logging.getLogger(__name__)

class SLMService:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SLM model"""
        try:
            logger.info(f"Loading SLM model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("SLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SLM model: {e}")
            # Fallback to a simpler model if primary fails
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if primary fails"""
        try:
            logger.info("Trying fallback model: TinyLlama-1.1B")
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise Exception("Could not load any SLM model")
    
    def generate_response(self, prompt: str, max_length: int = 500) -> str:
        """Generate response using SLM"""
        try:
            start_time = time()
            
            # Generate response
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.1,  # Low temperature for factual responses
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Remove the prompt from response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            processing_time = time() - start_time
            logger.info(f"Generated response in {processing_time:.2f}s")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

class RAGService:
    def __init__(self, vector_store, slm_service, config):
        self.vector_store = vector_store
        self.slm_service = slm_service
        self.config = config
    
    def query(self, query: str, top_k: Optional[int] = None, 
              similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Process query and generate response"""
        start_time = time()
        
        # Use provided values or defaults
        top_k = top_k or self.config.TOP_K_RESULTS
        similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
        
        try:
            # Step 1: Search for relevant documents
            search_results = self.vector_store.search(query, top_k=top_k * 2)
            
            # Step 2: Filter by similarity threshold
            relevant_results = [
                result for result in search_results 
                if result['similarity_score'] >= similarity_threshold
            ]
            
            # Take top k after filtering
            relevant_results = relevant_results[:top_k]
            
            if not relevant_results:
                return {
                    'answer': "I couldn't find relevant information in the provided documents to answer your question.",
                    'sources': [],
                    'processing_time': time() - start_time,
                    'confidence': 0.0
                }
            
            # Step 3: Prepare context
            context_parts = []
            for i, result in enumerate(relevant_results, 1):
                source_info = self._format_source_info(result['metadata'])
                context_parts.append(f"[Source {i}: {source_info}]\n{result['content']}\n")
            
            context = "\n---\n".join(context_parts)
            
            # Step 4: Construct prompt
            prompt = self._construct_prompt(query, context)
            
            # Step 5: Generate response
            response = self.slm_service.generate_response(prompt)
            
            # Step 6: Format sources for response
            sources = []
            for result in relevant_results:
                source = {
                    'source': result['metadata'].get('source', 'Unknown'),
                    'file_type': result['metadata'].get('file_type', 'Unknown'),
                    'page': result['metadata'].get('page_number'),
                    'sheet': result['metadata'].get('sheet_name'),
                    'rows': result['metadata'].get('rows'),
                    'content_preview': result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                }
                sources.append(source)
            
            return {
                'answer': response,
                'sources': sources,
                'processing_time': time() - start_time,
                'confidence': sum(result['similarity_score'] for result in relevant_results) / len(relevant_results)
            }
        
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def _format_source_info(self, metadata) -> str:
        """Format source information for display"""
        source = metadata.get('source', 'Unknown')
        file_type = metadata.get('file_type', 'Unknown')
        page = metadata.get('page_number', 'N/A')
        sheet = metadata.get('sheet_name', 'N/A')
        rows = metadata.get('rows', 'N/A')
        
        return f"Source: {source}, File Type: {file_type}, Page: {page}, Sheet: {sheet}, Rows: {rows}"
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """Construct prompt for SLM from query and context"""
        return f"Q: {query}\n\nContext: {context}\n\nA:"