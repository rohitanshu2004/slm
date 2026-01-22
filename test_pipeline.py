"""
Test script for the Financial RAG API pipeline.
Tests the complete flow: upload -> query -> sources retrieval
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 60

class RAGPipelineTest:
    """Test suite for RAG pipeline"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.test_results = []
    
    def test_api_health(self) -> bool:
        """Test 1: Check API health"""
        logger.info("=" * 60)
        logger.info("TEST 1: API Health Check")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API is healthy")
                logger.info(f"   Status: {data.get('status')}")
                logger.info(f"   Model loaded: {data.get('model_loaded')}")
                logger.info(f"   Vector DB ready: {data.get('vector_db_ready')}")
                logger.info(f"   Documents: {data.get('total_documents')}")
                self.test_results.append(("API Health Check", True))
                return True
            else:
                logger.error(f"❌ API returned status {response.status_code}")
                self.test_results.append(("API Health Check", False))
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to API: {e}")
            self.test_results.append(("API Health Check", False))
            return False
    
    def test_supported_formats(self) -> bool:
        """Test 2: Get supported file formats"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Supported File Formats")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/formats", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Supported formats: {data.get('supported_formats')}")
                logger.info(f"   Max file size: {data.get('max_file_size_mb')} MB")
                self.test_results.append(("Supported Formats", True))
                return True
            else:
                logger.error(f"❌ Failed to get formats: {response.status_code}")
                self.test_results.append(("Supported Formats", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Supported Formats", False))
            return False
    
    def test_stats_before_upload(self) -> bool:
        """Test 3: Get vector store stats before upload"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Vector Store Stats (Before Upload)")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Collection: {data.get('collection_name')}")
                logger.info(f"   Documents: {data.get('document_count')}")
                logger.info(f"   Status: {data.get('status')}")
                self.test_results.append(("Stats (Before Upload)", True))
                return True
            else:
                logger.error(f"❌ Failed to get stats: {response.status_code}")
                self.test_results.append(("Stats (Before Upload)", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Stats (Before Upload)", False))
            return False
    
    def test_query_without_documents(self) -> bool:
        """Test 4: Query when no documents exist (should fail gracefully)"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Query Without Documents (Expected to Fail)")
        logger.info("=" * 60)
        
        try:
            payload = {"query": "Test query"}
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                data = response.json()
                logger.info(f"✅ Correctly rejected empty query")
                logger.info(f"   Error: {data.get('error')}")
                logger.info(f"   Message: {data.get('message')}")
                self.test_results.append(("Query Without Documents", True))
                return True
            else:
                logger.warning(f"⚠️ Unexpectedly succeeded: {response.json()}")
                self.test_results.append(("Query Without Documents", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Query Without Documents", False))
            return False
    
    def create_test_file(self) -> Optional[Path]:
        """Create a simple test CSV file"""
        logger.info("\n" + "=" * 60)
        logger.info("Creating Test File")
        logger.info("=" * 60)
        
        try:
            test_file = Path("test_document.csv")
            content = """Company,Q1_Revenue,Q2_Revenue,Q3_Revenue,Q4_Revenue,Profit_Margin
TechCorp,5000000,5200000,5500000,6000000,0.25
FinanceInc,3000000,3100000,3200000,3500000,0.30
DataSys,2000000,2100000,2300000,2600000,0.22
CloudServ,4000000,4300000,4600000,5000000,0.28"""
            
            with open(test_file, "w") as f:
                f.write(content)
            
            logger.info(f"✅ Test file created: {test_file}")
            return test_file
        except Exception as e:
            logger.error(f"❌ Failed to create test file: {e}")
            return None
    
    def test_file_upload(self, file_path: Path) -> bool:
        """Test 5: Upload test file"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: File Upload")
        logger.info("=" * 60)
        
        try:
            with open(file_path, "rb") as f:
                files = [("files", f)]
                response = requests.post(
                    f"{self.base_url}/upload",
                    files=files,
                    timeout=TIMEOUT
                )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Upload successful")
                logger.info(f"   Status: {data.get('status')}")
                logger.info(f"   Message: {data.get('message')}")
                logger.info(f"   Files processed: {data.get('files_processed')}")
                logger.info(f"   Chunks created: {data.get('chunks_created')}")
                logger.info(f"   Processing time: {data.get('processing_time'):.2f}s")
                self.test_results.append(("File Upload", True))
                return True
            else:
                data = response.json()
                logger.error(f"❌ Upload failed: {response.status_code}")
                logger.error(f"   Error: {data.get('error')}")
                logger.error(f"   Message: {data.get('message')}")
                self.test_results.append(("File Upload", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("File Upload", False))
            return False
        finally:
            # Clean up test file
            try:
                file_path.unlink()
                logger.info("Test file cleaned up")
            except Exception as e:
                logger.warning(f"Could not delete test file: {e}")
    
    def test_stats_after_upload(self) -> bool:
        """Test 6: Get vector store stats after upload"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Vector Store Stats (After Upload)")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                doc_count = data.get('document_count', 0)
                logger.info(f"✅ Stats retrieved")
                logger.info(f"   Collection: {data.get('collection_name')}")
                logger.info(f"   Documents: {doc_count}")
                logger.info(f"   Status: {data.get('status')}")
                
                if doc_count > 0:
                    self.test_results.append(("Stats (After Upload)", True))
                    return True
                else:
                    logger.warning("⚠️ No documents in vector store after upload")
                    self.test_results.append(("Stats (After Upload)", False))
                    return False
            else:
                logger.error(f"❌ Failed to get stats: {response.status_code}")
                self.test_results.append(("Stats (After Upload)", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Stats (After Upload)", False))
            return False
    
    def test_query_with_documents(self) -> bool:
        """Test 7: Query documents"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: Query Documents")
        logger.info("=" * 60)
        
        try:
            queries = [
                "What is the profit margin of TechCorp?",
                "Which company had the highest revenue in Q4?",
                "What are the financial metrics in the data?",
            ]
            
            successful_queries = 0
            for query in queries:
                logger.info(f"\n📝 Query: {query}")
                
                payload = {
                    "query": query,
                    "top_k": 3,
                    "similarity_threshold": 0.5
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json=payload,
                    timeout=TIMEOUT
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('answer', 'No answer')
                    sources = data.get('sources', [])
                    
                    logger.info(f"✅ Query successful (took {elapsed:.2f}s)")
                    logger.info(f"   Answer: {answer[:150]}...")
                    logger.info(f"   Sources found: {len(sources)}")
                    
                    for idx, source in enumerate(sources, 1):
                        logger.info(f"   Source {idx}:")
                        logger.info(f"      File: {source.get('source')}")
                        logger.info(f"      Type: {source.get('file_type')}")
                        logger.info(f"      Relevance: {source.get('relevance_score', 0):.2f}")
                    
                    successful_queries += 1
                else:
                    data = response.json()
                    logger.error(f"❌ Query failed: {data.get('error')}")
            
            if successful_queries == len(queries):
                self.test_results.append(("Query Documents", True))
                return True
            else:
                logger.warning(f"⚠️ Only {successful_queries}/{len(queries)} queries succeeded")
                self.test_results.append(("Query Documents", successful_queries > 0))
                return successful_queries > 0
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Query Documents", False))
            return False
    
    def test_empty_query(self) -> bool:
        """Test 8: Empty query validation"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: Empty Query Validation")
        logger.info("=" * 60)
        
        try:
            payload = {"query": "   "}  # Empty/whitespace query
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                data = response.json()
                logger.info(f"✅ Correctly rejected empty query")
                logger.info(f"   Error: {data.get('error')}")
                self.test_results.append(("Empty Query Validation", True))
                return True
            else:
                logger.error(f"❌ Should have rejected empty query")
                self.test_results.append(("Empty Query Validation", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Empty Query Validation", False))
            return False
    
    def test_clear_documents(self) -> bool:
        """Test 9: Clear vector store"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 9: Clear Vector Store")
        logger.info("=" * 60)
        
        try:
            response = requests.delete(
                f"{self.base_url}/clear",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Documents cleared")
                logger.info(f"   Status: {data.get('status')}")
                logger.info(f"   Message: {data.get('message')}")
                self.test_results.append(("Clear Vector Store", True))
                return True
            else:
                logger.error(f"❌ Failed to clear: {response.status_code}")
                self.test_results.append(("Clear Vector Store", False))
                return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.test_results.append(("Clear Vector Store", False))
            return False
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
        
        logger.info("=" * 60)
        logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        logger.info("=" * 60)
    
    def run_full_pipeline(self):
        """Run complete test pipeline"""
        logger.info("\n\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " " * 58 + "║")
        logger.info("║" + "FINANCIAL RAG API - FULL PIPELINE TEST".center(58) + "║")
        logger.info("║" + " " * 58 + "║")
        logger.info("╚" + "=" * 58 + "╝")
        
        # Run tests
        self.test_api_health()
        self.test_supported_formats()
        self.test_stats_before_upload()
        self.test_query_without_documents()
        
        # Create and upload test file
        test_file = self.create_test_file()
        if test_file:
            self.test_file_upload(test_file)
            time.sleep(2)  # Wait for processing
            
            self.test_stats_after_upload()
            self.test_query_with_documents()
            self.test_empty_query()
            self.test_clear_documents()
        else:
            logger.error("❌ Could not create test file, skipping upload tests")
        
        # Print summary
        self.print_summary()


if __name__ == "__main__":
    import sys
    
    # Optional: Accept custom API URL as argument
    api_url = sys.argv[1] if len(sys.argv) > 1 else API_BASE_URL
    
    logger.info(f"Using API URL: {api_url}")
    
    tester = RAGPipelineTest(base_url=api_url)
    tester.run_full_pipeline()
