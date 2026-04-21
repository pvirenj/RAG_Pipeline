import streamlit as st
import requests

# Configure the page
st.set_page_config(
    page_title="StudyAI Chat",
    page_icon="📚",
    layout="centered"
)

st.title("📚 StudyAI RAG Chat")
st.markdown("Ask questions based on the ingested textbooks.")

# ----------------------------------------------------
# Sidebar Configuration
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    API_URL = st.text_input("API URL", value="http://localhost:8000")
    
    # We could fetch available collections via GET /collections,
    # but for simplicity, we provide a text input.
    collection_name = st.text_input("Collection Name", value="learning_knowledge_base")

    st.markdown("---")
    st.markdown("""
    **💡 Tip:** Make sure your FastAPI backend is running!
    ```bash
    python main.py serve
    ```
    """)

# ----------------------------------------------------
# Chat UI Setup
# ----------------------------------------------------
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm StudyAI. Ask me anything about your textbooks.", "sources": []}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message with sources, display them nicely
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.write(f"{idx+1}. {source}")

# ----------------------------------------------------
# Handle User Input
# ----------------------------------------------------
if prompt := st.chat_input("Ask a question..."):
    # 1. Add user message to chat history & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Add placeholder for assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 3. Call the FastAPI backend
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "collection": collection_name,
                        "question": prompt
                    },
                    timeout=120 # Can take a while if LLM is slow or re-ranking is heavy
                )
                
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                
                # Display the answer
                message_placeholder.markdown(answer)
                
                # Display the sources below answer
                if sources:
                    with st.expander("📚 Sources"):
                        for idx, source in enumerate(sources):
                            st.write(f"{idx+1}. {source}")
                            
                # Save assistant response to memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                message_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection Error: Could not connect to {API_URL}. Is the FastAPI server running?"
            message_placeholder.error(error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            message_placeholder.error(error_msg)
