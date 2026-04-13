# 💠 Endee AI | Knowledge Explorer

A production-grade Retrieval-Augmented Generation (RAG) system built for high-speed document intelligence. This project leverages **Endee** (Vector Database), **Groq** (LLM Acceleration), and **Streamlit** (UI) to provide a seamless knowledge discovery experience.

## 🏗️ Architecture Overview

```text
+----------------+      +-------------------+      +------------------+
|  User Upload   | ---> | Document Processor| ---> | Sentence Encoder |
| (PDF/DOCX/TXT) |      | (Chunking/Parsing)|      | (all-MiniLM-L6)  |
+----------------+      +-------------------+      +------------------+
                                                            |
                                                            v
+----------------+      +-------------------+      +------------------+
|  Groq Llama 3  | <--- | Context Retrieval | <--- |  Vector Store    |
| (LLM Inference)|      | (Semantic Search) |      | (Endee/Local)    |
+----------------+      +-------------------+      +------------------+
        |
        v
+----------------+
| Streaming UI   |
| (Streamlit)    |
+----------------+
```

## 🌟 Key Features

- **🚀 Performance**: Sub-50ms search latency using semantic vector indexing.
- **📄 Multi-Format Support**: Native handling for PDF, Word (.docx), and Plain Text.
- **🧩 Modularity**: Clean separation between UI (Streamlit), Logic (LLMService), and Data (Processor).
- **📉 Error Resilience**: Automatic fallback to local vector search if database connection is interrupted.
- **✨ Smart Summary**: Advanced executive summarization using multi-point document sampling.

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **LLM Provider**: Groq (Llama 3.3 70B / 8B)
- **Vector Indexing**: Endee (Scaleable Vector DB) & Local Numpy (Fallback)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Parsing**: PyPDF2, python-docx

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Groq API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🔒 Security & Best Practices

- **Environment Variables**: API keys are handled via a `.env` interface or secure Streamlit inputs.
- **Data Privacy**: Local indexing mode ensures no document data leaves your environment except for inference.
- **Modularity**: Every component is isolated for easy testing and updates.

---
Built with ❤️ for the Endee AI Ecosystem.
