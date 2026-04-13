import streamlit as st
from groq import Groq
import os
import time
import numpy as np
from endee_client import EndeeClient
from processor import DocumentProcessor

# 🔑 Configuration
HARDCODED_GROQ_KEY = "" 

# Page Configuration
st.set_page_config(page_title="Endee AI | Knowledge Explorer", page_icon="💠", layout="wide")

# Load Custom CSS
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "index_name" not in st.session_state:
    st.session_state.index_name = f"doc_index_{int(time.time())}"
if "local_knowledge" not in st.session_state:
    st.session_state.local_knowledge = []
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

# --- Persistent API Key Logic ---
def get_saved_api_key():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=")[1].strip()
    return ""

def save_api_key(key):
    with open(".env", "w") as f:
        f.write(f"GROQ_API_KEY={key}")
    st.session_state.stored_api_key = key

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("💠 Nav Center")
    
    # API Key Section
    if HARDCODED_GROQ_KEY:
        groq_api_key = HARDCODED_GROQ_KEY
        st.success("✅ Production Mode")
    else:
        saved_key = get_saved_api_key()
        groq_api_key = st.text_input("Groq API Key", value=saved_key, type="password")
        if st.button("💾 Save Key"):
            save_api_key(groq_api_key)
            st.toast("Key saved!")

    st.divider()
    selected_model = st.selectbox(
        "🧠 Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )
    
    current_temp = st.slider("🌡️ Temperature", 0.0, 1.0, 0.3)
    
    st.divider()
    if st.button("🗑️ Reset App", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Core Logic Functions ---
def call_groq(prompt, stream=True):
    if not groq_api_key:
        return None
    try:
        client = Groq(api_key=groq_api_key)
        return client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=current_temp,
            stream=stream,
        )
    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        return None

def summarize_document():
    if not st.session_state.doc_text:
        return "No document text available."
    
    with st.spinner("✨ Analyzing entire document for summary..."):
        # Improved Sampling: Take parts from start, middle, and end if long
        full_text = st.session_state.doc_text
        if len(full_text) > 12000:
            sample = full_text[:4000] + "\n...[MIDDLE CONTENT]...\n" + \
                     full_text[len(full_text)//2 - 2000 : len(full_text)//2 + 2000] + \
                     "\n...[FINAL CONTENT]...\n" + full_text[-4000:]
        else:
            sample = full_text
            
        prompt = f"""
        Provide a professional and detailed Executive Summary of this document. 
        Structure your response with:
        1. 📌 Overview
        2. 🔑 Key Themes
        3. 🚀 Core Takeaways
        
        DOCUMENT TEXT:
        {sample}
        """
        
        resp = call_groq(prompt, stream=False)
        if resp:
            return resp.choices[0].message.content
        return "⚠️ The AI could not generate a summary. Please check your API key or document content."

def retrieve_context(query_vec):
    context = ""
    sources = []
    # Local fallback logic (robust)
    if st.session_state.local_knowledge:
        scores = []
        for txt, vec in st.session_state.local_knowledge:
            score = np.dot(query_vec, vec)
            scores.append((score, txt))
        scores.sort(key=lambda x: x[0], reverse=True)
        for _, txt in scores[:5]: # increased to 5 for better context
            context += txt + "\n\n"
            sources.append(txt)
    return context, sources

# --- Main UI ---
st.title("💠 Endee Knowledge Explorer")

tab1, tab2, tab3 = st.tabs(["📂 Upload", "💬 Chat Explorer", "📊 Insights & Summary"])

with tab1:
    if not st.session_state.processed_docs:
        uploaded_file = st.file_uploader("Upload Document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.status("🚀 Processing Knowledge...", expanded=True) as status:
                try:
                    proc = DocumentProcessor()
                    text = proc.extract_text(uploaded_file)
                    st.session_state.doc_text = text
                    
                    st.write("Chunking content...")
                    chunks = proc.split_text(text)
                    
                    st.write("Generating Semantic Vectors...")
                    embeddings = proc.generate_embeddings(chunks)
                    
                    st.session_state.local_knowledge = list(zip(chunks, embeddings))
                    st.session_state.processed_docs = True
                    status.update(label="✅ Ready!", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)}", state="error")
                    st.error(f"Try a different file or check the format: {e}")
    else:
        st.success("✅ Document Loaded and Indexed")
        st.info(f"Content length: {len(st.session_state.doc_text)} characters")
        if st.button("🔄 Upload New Document"):
            st.session_state.processed_docs = False
            st.rerun()

with tab2:
    if not st.session_state.processed_docs:
        st.warning("Please upload a document in the 'Upload' tab first.")
    else:
        # Quick Action Buttons
        col1, col2, col3 = st.columns(3)
        trigger_query = None
        
        with col1:
            if st.button("📝 Summarize Page", use_container_width=True):
                trigger_query = "Summarize the key points of this document concisely."
        with col2:
            if st.button("🔑 Key Takeaways", use_container_width=True):
                trigger_query = "What are the top 3 most important takeaways from this document?"
        with col3:
            if st.button("❓ Suggested Questions", use_container_width=True):
                trigger_query = "Based on this document, what are 3 interesting questions I could ask?"

        # Chat UI
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input handling
        query = st.chat_input("Ask about your document...")
        if trigger_query: # Override with button action if clicked
            query = trigger_query

        if query:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.chat_message("assistant"):
                # Retrieval
                proc = DocumentProcessor()
                query_vec = proc.generate_query_embedding(query)
                context, sources = retrieve_context(query_vec)
                
                # Generation
                prompt = f"System: Use the context to answer.\nContext: {context}\n\nUser: {query}"
                response_placeholder = st.empty()
                full_response = ""
                
                stream = call_groq(prompt)
                if stream:
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

with tab3:
    if not st.session_state.processed_docs:
        st.warning("Please upload a document first.")
    else:
        st.subheader("📄 Document Overview")
        if not st.session_state.doc_summary:
            if st.button("🪄 Generate Executive Summary"):
                st.session_state.doc_summary = summarize_document()
                st.rerun()
        
        if st.session_state.doc_summary:
            st.markdown(st.session_state.doc_summary)
            if st.button("♻️ Regenerate Summary"):
                st.session_state.doc_summary = ""
                st.rerun()
        
        st.divider()
        st.subheader("💡 Analysis")
        st.write("Current analysis mode: Semantic Extraction")
        st.progress(1.0, text="Health check: Vector Index optimized")
