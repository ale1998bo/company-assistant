import os
import glob
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from rag_engine import SimpleRAG
from google.genai import types 

# --- CONFIGURAZIONE ---
PROJECT_ID = "company-assistant-482110"
REGION = "us-central1"
# NOTA: Assicurati che il modello esista. Se 2.5 da errore, usa "gemini-1.5-flash"
MODEL_NAME = "gemini-2.5-flash" 
SIMILARITY_THRESHOLD = 0.4
KNOWLEDGE_FOLDER = "Knowledge Base"

app = Flask(__name__)
rag_system = SimpleRAG(PROJECT_ID, REGION)

# --- MEMORIA DELLA CHAT (History) ---
chat_history = [] 
MAX_HISTORY = 6 

# --- HELPER FUNCTIONS (Spostata QUI in alto!) ---
def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    text_content = ""
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text_content += extracted + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        return text_content
    except Exception as e:
        print(f"Errore lettura {file_path}: {e}")
        return ""

# --- CARICAMENTO DATI ---
def load_initial_data():
    if not os.path.exists(KNOWLEDGE_FOLDER):
        os.makedirs(KNOWLEDGE_FOLDER)
        
    extensions = ['*.md', '*.txt', '*.pdf']
    files_on_disk = []
    for ext in extensions:
        files_on_disk.extend(glob.glob(os.path.join(KNOWLEDGE_FOLDER, ext)))
    
    # 1. Vediamo quali file conosciamo già
    known_files = set()
    for entry in rag_system.knowledge_base:
        known_files.add(entry['source'])
        
    print(f"--- Controllo nuovi file in '{KNOWLEDGE_FOLDER}' ---")
    
    new_files_count = 0
    for file_path in files_on_disk:
        filename = os.path.basename(file_path)
        
        # 2. Se il file è già nel DB, saltiamo!
        if filename in known_files:
            continue
            
        # 3. Se è nuovo, lo processiamo
        print(f"Nuovo file rilevato: {filename} -> Indicizzazione...")
        # ORA FUNZIONA PERCHÉ LA FUNZIONE È GIÀ STATA DEFINITA SOPRA
        content = extract_text_from_file(file_path) 
        if content.strip():
            rag_system.ingest_file(filename, content)
            new_files_count += 1

    if new_files_count == 0:
        print("--- Nessun nuovo file da indicizzare. DB Aggiornato. ---")
    else:
        print(f"--- Indicizzati {new_files_count} nuovi file. ---")

print("--- AVVIO AGENTE IBRIDO (Persistent DB) ---")
# Ora possiamo chiamarla perché tutto è stato definito
load_initial_data()

# --- ROUTES ---
@app.route('/')
def home():
    global chat_history
    chat_history = []
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No filename"}), 400
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(KNOWLEDGE_FOLDER, filename)
        file.save(save_path)
        content = extract_text_from_file(save_path)
        if not content.strip(): return jsonify({"error": "File vuoto"}), 400
        rag_system.ingest_file(filename, content)
        return jsonify({"messaggio": f"Documento '{filename}' appreso!"})

@app.route('/api/esegui-azione', methods=['POST'])
def chat():
    global chat_history
    data = request.json
    user_query = data.get('query', '')
    if not user_query: return jsonify({"messaggio": "Domanda vuota."})

    try:
        # 1. Cerca nel DB
        retrieved_docs = rag_system.search(user_query, top_k=5)
        best_score = retrieved_docs[0]['similarity'] if retrieved_docs else 0.0
        
        mode = "internal"
        response_text = ""
        sources = []
        
        history_str = ""
        for msg in chat_history[-MAX_HISTORY:]:
            role = "UTENTE" if msg['role'] == 'user' else "ASSISTENTE"
            history_str += f"{role}: {msg['content']}\n"

        print(f"Query: '{user_query}' | Score: {best_score:.3f}")

        # --- ROUTER LOGIC ---
        if best_score >= SIMILARITY_THRESHOLD:
            # RAMO RAG INTERNO
            mode = "internal"
            context_str = "\n".join([f"- {d['text']} (Fonte: {d['source']})" for d in retrieved_docs])
            
            prompt = f"""
            Sei un assistente aziendale intelligente.
            
            STORICO CONVERSAZIONE:
            {history_str}
            
            CONTESTO DOCUMENTI TROVATI:
            {context_str}
            
            NUOVA DOMANDA UTENTE: {user_query}
            
            ISTRUZIONI:
            Usa la cronologia per capire il contesto.
            Usa i documenti trovati per rispondere. Cita la fonte se possibile.
            """
            
            resp = rag_system.client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt]
            )
            response_text = resp.text
            sources = list(set([d['source'] for d in retrieved_docs]))

        else:
            # RAMO GOOGLE SEARCH
            mode = "web"
            print("--- Using Google Search ---")
            
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            prompt = f"""
            STORICO CONVERSAZIONE:
            {history_str}
            
            NUOVA DOMANDA UTENTE: {user_query}
            
            Rispondi usando Google Search. Tieni conto dello storico se serve.
            """
            
            resp = rag_system.client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(tools=[google_search_tool])
            )
            response_text = resp.text
            sources = ["Google Search"]

        # --- AGGIORNA HISTORY ---
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response_text})

        return jsonify({
            "messaggio": response_text,
            "fonti": sources,
            "modalita": mode,
            "score": float(best_score)
        })

    except Exception as e:
        print(f"Errore: {e}")
        return jsonify({"messaggio": f"Errore: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, load_dotenv=False)