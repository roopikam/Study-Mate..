import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class StudyMateBackend:
    def _init_(self, embed_model_name="all-MiniLM-L6-v2", llm_model_name="tiiuae/mistral-7b-instruct"):
        # Load embedding model
        self.embedder = SentenceTransformer(embed_model_name)
        
        # Initialize FAISS index (will build after loading docs)
        self.index = None
        self.text_chunks = []  # Keep track of chunks to map back from index
        
        # Load LLM model pipeline for text generation
        self.qa_pipeline = pipeline("text2text-generation", model=llm_model_name, device=0 if self._has_gpu() else -1)
        
    def _has_gpu(self):
        import torch
        return torch.cuda.is_available()
 def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text
    
    def chunk_text(self, text, max_chunk_size=500):
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < max_chunk_size:
                current_chunk += sent + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def build_faiss_index(self, chunks):
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        self.text_chunks = chunks
def semantic_search(self, query, top_k=5):
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        results = [self.text_chunks[idx] for idx in indices[0]]
        return results
    
    def generate_answer(self, question, context_chunks):
        context = "\n\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.qa_pipeline(prompt, max_length=256, do_sample=False)[0]['generated_text']
        return response
