import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import re
import io

class DocumentProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def extract_text(self, uploaded_file):
        filename = uploaded_file.name.lower()
        if filename.endswith('.pdf'):
            return self.extract_text_from_pdf(uploaded_file)
        elif filename.endswith('.docx'):
            return self.extract_text_from_docx(uploaded_file)
        elif filename.endswith('.txt'):
            return uploaded_file.read().decode('utf-8')
        else:
            raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")

    def extract_text_from_pdf(self, pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")

    def extract_text_from_docx(self, docx_file):
        try:
            doc = Document(docx_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Word extraction failed: {str(e)}")

    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        if not text: return []
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-(chunk_overlap):] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_embeddings(self, texts):
        if not texts: return []
        return self.model.encode(texts).tolist()

    def generate_query_embedding(self, query):
        return self.model.encode([query])[0].tolist()
