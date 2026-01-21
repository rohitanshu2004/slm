import streamlit as st
import requests

st.set_page_config(page_title="Financial Document Q&A", layout="wide")

# Sidebar for file upload
with st.sidebar:
    st.title("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'xlsx', 'xls', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                files = [("files", file) for file in uploaded_files]
                response = requests.post(
                    "http://localhost:8000/upload",
                    files=files
                )
                
                if response.status_code == 200:
                    st.success("Documents processed successfully!")
                    st.session_state.documents_processed = True
                else:
                    st.error("Error processing documents")

# Main chat interface
st.title("💬 Financial Document Q&A")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your financial documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": prompt}
            )
            
            if response.status_code == 200:
                data = response.json()

                # Validate response structure
                if not isinstance(data, dict) or 'answer' not in data or 'sources' not in data:
                    st.error("Invalid response structure from backend.")
                    return

                answer = data['answer']
                sources = data.get('sources', [])

                # Display answer
                st.markdown(answer)

                # Display sources in expander
                if not sources:
                    st.info("No sources found.")
                else:
                    with st.expander("View Sources"):
                        for source in sources:
                            try:
                                # Safe access to fields with defaults
                                file_name = source.get('source', 'Unknown')
                                file_type = source.get('file_type', 'Unknown')
                                content_preview = source.get('content_preview', '')
                                relevance_score = source.get('relevance_score', 0.0)
                                page = source.get('page')
                                sheet = source.get('sheet')
                                rows = source.get('rows')

                                # Display source details
                                st.write(f"📄 **File:** {file_name}")
                                st.write(f"📊 **Type:** {file_type}")
                                st.write(f"🎯 **Relevance:** {relevance_score * 100:.1f}%")

                                if page is not None:
                                    st.write(f"📖 **Page:** {page}")
                                if sheet is not None:
                                    st.write(f"📋 **Sheet:** {sheet}")
                                if rows is not None:
                                    st.write(f"📏 **Rows:** {rows}")

                                st.write(f"📝 **Content Preview:** {content_preview}")
                                st.divider()
                            except Exception as e:
                                st.error(f"Error displaying source: {e}")

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("Error getting response")