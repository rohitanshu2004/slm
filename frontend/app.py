import streamlit as st
import requests
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 300  # 5 minutes

st.set_page_config(
    page_title="Financial Document Q&A", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .error-box {
        background-color: #ffebee;
        border: 1px solid #ef5350;
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .success-box {
        background-color: #e8f5e9;
        border: 1px solid #66bb6a;
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .warning-box {
        background-color: #fff3e0;
        border: 1px solid #ffa726;
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed: {e}")
        return False


def handle_api_error(error: Dict[str, Any], context: str = "Operation") -> None:
    """Display formatted error message"""
    error_code = error.get("error", "UNKNOWN_ERROR")
    message = error.get("message", "An unexpected error occurred")
    details = error.get("details")
            
    error_text = f"**{context} Failed** ({error_code})\n\n{message}"
    if details and isinstance(details, str):
        error_text += f"\n\nDetails: {details}"
    elif details and isinstance(details, dict):
        details_str = "\n".join([f"- {k}: {v}" for k, v in details.items()])
        error_text += f"\n\nDetails:\n{details_str}"
    
    st.error(error_text)
    logger.error(f"{context} error: {error_code} - {message}")


def get_supported_formats() -> Optional[Dict[str, Any]]:
    """Get list of supported file formats from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/formats",
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch supported formats: {e}")
    return None


def upload_files(files, clear_before_upload: bool = True) -> bool:
    """Upload files to API with progress tracking and duplicate prevention"""
    try:
        if not files:
            st.error("Please select at least one file to upload")
            return False

        # Check if upload is already in progress
        if st.session_state.get('upload_in_progress', False):
            st.warning("⚠️ An upload is already in progress. Please wait for it to complete.")
            return False

        # Set upload in progress flag
        st.session_state.upload_in_progress = True

        files_to_upload = [
    (
        "files",
        (
            file.name,
            file.getvalue(),   # <-- BYTES (critical)
            file.type          # <-- MIME type
        )
    )
    for file in files
]


        # Create progress bar and status containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing upload...")

        try:
            url = f"{API_BASE_URL}/upload?clear_before_upload={str(clear_before_upload).lower()}"
            response = requests.post(
                url,
                files=files_to_upload,
                timeout=300  # 5 minute timeout for large uploads
            )

            if response.status_code == 200:
                data = response.json()
                progress_bar.progress(100)
                status_text.text("✅ Upload completed successfully!")

                st.success(f"✅ {data.get('message', 'Files processed successfully!')}")
                st.info(
                    f"📊 **Processing Summary**\n\n"
                    f"- Files processed: {data.get('files_processed', 0)}\n"
                    f"- Chunks created: {data.get('chunks_created', 0)}\n"
                    f"- Processing time: {data.get('processing_time', 0):.2f}s"
                )
                logger.info(f"Successfully processed {data.get('files_processed')} files")
                return True
            elif response.status_code == 409:
                # Upload already in progress
                error_data = response.json()
                st.warning(f"⚠️ {error_data.get('message', 'Another upload is already in progress')}")
                return False
            else:
                error_data = response.json()
                progress_bar.progress(100)
                status_text.text("❌ Upload failed")
                handle_api_error(error_data, "File Upload")
                return False

        except requests.exceptions.Timeout:
            progress_bar.progress(100)
            status_text.text("❌ Upload timed out")
            st.error("⏱️ Request timed out. The files might be too large or the server is overloaded.")
            logger.error("File upload timeout")
            return False
        except requests.exceptions.ConnectionError:
            progress_bar.progress(100)
            status_text.text("❌ Connection failed")
            st.error("❌ Cannot connect to the API server. Make sure it's running on http://localhost:8000")
            logger.error("API connection error during upload")
            return False
        except Exception as e:
            progress_bar.progress(100)
            status_text.text("❌ Upload failed")
            st.error(f"❌ Unexpected error during upload: {str(e)}")
            logger.error(f"Unexpected upload error: {e}", exc_info=True)
            return False
        finally:
            # Clear upload in progress flag
            st.session_state.upload_in_progress = False

    except Exception as e:
        # Clear upload in progress flag even if outer try fails
        st.session_state.upload_in_progress = False
        st.error(f"❌ Unexpected error during upload: {str(e)}")
        logger.error(f"Unexpected upload error: {e}", exc_info=True)
        return False


def query_api(query: str, top_k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Query the API"""
    try:
        payload = {"query": query}
        if top_k:
            payload["top_k"] = top_k
        if similarity_threshold:
            payload["similarity_threshold"] = similarity_threshold
        
        with st.spinner("Searching documents..."):
            response = requests.post(
                f"{API_BASE_URL}/query",
                json=payload,
                timeout=TIMEOUT
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            handle_api_error(error_data, "Query")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Query timed out. Please try again or use a simpler query.")
        logger.error("Query timeout")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Make sure it's running on http://localhost:8000")
        logger.error("API connection error during query")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during query: {str(e)}")
        logger.error(f"Unexpected query error: {e}", exc_info=True)
        return None


def get_vector_store_stats() -> Optional[Dict[str, Any]]:
    """Get statistics about the vector store"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/stats",
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch stats: {e}")
        return None


# Main UI
st.title("💬 Financial Document Q&A System")

# Check API health at startup
if not check_api_health():
    st.error(
        "❌ **API Server Not Available**\n\n"
        "The backend API is not running. Please ensure:\n"
        "1. The backend server is started: `python -m backend.main`\n"
        "2. Ollama is running: `ollama serve`\n"
        "3. Both services are accessible on localhost"
    )
    st.stop()

st.success("✅ API is online and ready")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Document Management")
    
    # Show supported formats
    formats_info = get_supported_formats()
    if formats_info:
        max_size = formats_info.get("max_file_size_mb", "Unknown")
        st.info(f"**Max file size:** {max_size}MB\n\n**Supported formats:** PDF, XLSX, XLS, CSV")
    
    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="You can select multiple files at once"
    )
    
    # Option to clear previous data before uploading
    clear_previous_data = st.checkbox("Clear previous data", value=True, help="Clear vector DB and temporary files before uploading new documents")

    if uploaded_files:
        st.write(f"**Selected files:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🚀 Process Documents", use_container_width=True):
            success = upload_files(uploaded_files, clear_before_upload=clear_previous_data)
            if success:
                st.session_state.documents_processed = True
    
    st.divider()
    
    # Vector store statistics
    st.subheader("📊 Vector Store Status")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        stats = get_vector_store_stats()
        if stats:
            st.metric("Total Documents", stats.get("document_count", 0))
            st.metric("Collection", stats.get("collection_name", "N/A"))
        else:
            st.warning("Could not retrieve statistics")
    else:
        stats = get_vector_store_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", stats.get("document_count", 0))
            with col2:
                st.metric("📦 Status", stats.get("status", "Unknown"))
    
    st.divider()
    
    # Query settings
    st.subheader("⚙️ Query Settings")
    top_k = None
    similarity_threshold = None
    with st.expander("Advanced Options"):
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources = slower but more comprehensive"
        )
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = only most relevant sources"
        )

# Main chat interface
st.subheader("💬 Ask Questions About Your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if documents are available before allowing queries
if "documents_processed" not in st.session_state:
    stats = get_vector_store_stats()
    if stats and stats.get("document_count", 0) == 0:
        st.info("📤 Please upload documents first using the sidebar to start asking questions.")
    elif stats:
        st.session_state.documents_processed = True

# Chat input
if prompt := st.chat_input("Ask about your financial documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from backend
    with st.chat_message("assistant"):
        response_data = query_api(prompt, top_k, similarity_threshold)
        
        if response_data:
            answer = response_data.get('answer', 'No answer generated')
            sources = response_data.get('sources', [])
            processing_time = response_data.get('processing_time', 0)
            
            # Display answer
            st.markdown(answer)
            
            # Display metadata
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ Processing time: {processing_time:.2f}s")
            with col2:
                st.caption(f"📚 Sources found: {len(sources)}")
            
            # Display sources in expander
            if not sources:
                st.info("ℹ️ No relevant sources found for this query.")
            else:
                with st.expander(f"📖 View Sources ({len(sources)})"):
                    for idx, source in enumerate(sources, 1):
                        try:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Source {idx}**")
                                st.write(f"📄 **File:** {source.get('source', 'Unknown')}")
                                st.write(f"📊 **Type:** {source.get('file_type', 'Unknown')}")
                            
                            with col2:
                                score = source.get('relevance_score', 0.0)
                                st.metric("Relevance", f"{score*100:.0f}%")
                            
                            # Optional metadata
                            metadata_parts = []
                            if source.get('page') is not None:
                                metadata_parts.append(f"📖 Page: {source.get('page')}")
                            if source.get('sheet') is not None:
                                metadata_parts.append(f"📋 Sheet: {source.get('sheet')}")
                            if source.get('rows') is not None:
                                metadata_parts.append(f"📏 Rows: {source.get('rows')}")
                            
                            if metadata_parts:
                                st.caption(" | ".join(metadata_parts))
                            
                            # Content preview
                            content = source.get('content_preview', '')
                            if content:
                                st.write("**Preview:**")
                                st.text_area(
                                    f"Content preview for source {idx}",
                                    value=content[:300] + "..." if len(content) > 300 else content,
                                    height=100,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                            
                            st.divider()
                        except Exception as e:
                            st.error(f"Error displaying source {idx}: {str(e)}")
                            logger.error(f"Error rendering source {idx}: {e}", exc_info=True)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
