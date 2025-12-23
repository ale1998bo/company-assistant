import re
import os
import json
from google import genai
import numpy as np

# File dove salveremo la memoria
DB_FILE = "vector_store.json"

class SimpleRAG:
    def __init__(self, project_id, region):
        self.client = genai.Client(vertexai=True, project=project_id, location=region)
        self.knowledge_base = [] 
        
        # Al momento della creazione, proviamo a caricare i dati salvati
        self.load_from_disk()

    def chunk_text(self, text, chunk_size=300, overlap=50):
        """Smart Chunking (uguale a prima)"""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = " ".join(current_chunk.split()[-10:]) 
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def get_embedding(self, text):
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values

    def ingest_file(self, filename, content):
        """Crea embedding e SALVA SUBITO su disco"""
        # print(f"Ingestione e salvataggio: {filename}...")
        chunks = self.chunk_text(content)
        
        for chunk in chunks:
            vector = self.get_embedding(chunk)
            self.knowledge_base.append({
                "source": filename,
                "text": chunk,
                "vector": np.array(vector) # In memoria usiamo Numpy
            })
        
        # Dopo aver aggiunto i nuovi chunk, salviamo tutto
        self.save_to_disk()

    def search(self, query, top_k=8):
        if not self.knowledge_base:
            return []

        query_vector = np.array(self.get_embedding(query))
        results = []

        for entry in self.knowledge_base:
            doc_vector = entry["vector"]
            
            dot_product = np.dot(query_vector, doc_vector)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(doc_vector)
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)
            
            results.append({
                "similarity": similarity,
                "text": entry["text"],
                "source": entry["source"]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # --- NUOVE FUNZIONI DI PERSISTENZA ---

    def save_to_disk(self):
        """Salva la knowledge base su file JSON"""
        data_to_save = []
        for item in self.knowledge_base:
            # Creiamo una copia convertendo il vettore Numpy in Lista normale (per il JSON)
            item_copy = item.copy()
            item_copy['vector'] = item['vector'].tolist()
            data_to_save.append(item_copy)
        
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        # print("Database salvato su disco.")

    def load_from_disk(self):
        """Carica il DB se esiste"""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r', encoding='utf-8') as f:
                    data_loaded = json.load(f)
                
                self.knowledge_base = []
                for item in data_loaded:
                    # Riconvertiamo la lista in Numpy Array
                    item['vector'] = np.array(item['vector'])
                    self.knowledge_base.append(item)
                
                print(f"--- DB CARICATO DA DISCO: {len(self.knowledge_base)} chunks pronti ---")
            except Exception as e:
                print(f"Errore nel caricamento DB: {e}")
                self.knowledge_base = []
        else:
            print("--- Nessun DB trovato, si parte da zero ---")