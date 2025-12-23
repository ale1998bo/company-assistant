import os
import glob
import sys
from pypdf import PdfReader
from rag_engine import SimpleRAG
from google.genai import types

# --- COLORI PER IL TERMINALE ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'   # Per Web Search
    GREEN = '\033[92m'  # Per RAG Interno
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- CONFIGURAZIONE ---
PROJECT_ID = "company-assistant-482110"
REGION = "us-central1"
MODEL_NAME = "gemini-1.5-flash"
SIMILARITY_THRESHOLD = 0.4
KNOWLEDGE_FOLDER = "Knowledge Base"
MAX_HISTORY = 6  # Ricorda gli ultimi 3 scambi (User + AI)

# --- INIZIALIZZAZIONE ---
rag_system = SimpleRAG(PROJECT_ID, REGION)

# --- HELPER FUNCTIONS ---
def extract_text_from_file(file_path):
    """Legge PDF o Testo"""
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
        print(f"{Colors.FAIL}Errore lettura {file_path}: {e}{Colors.ENDC}")
        return ""

def load_initial_data():
    """Caricamento Incrementale (uguale al main.py)"""
    print(f"{Colors.HEADER}--- Avvio Knowledge Base ---{Colors.ENDC}")
    
    if not os.path.exists(KNOWLEDGE_FOLDER):
        os.makedirs(KNOWLEDGE_FOLDER)
        
    extensions = ['*.md', '*.txt', '*.pdf']
    files_on_disk = []
    for ext in extensions:
        files_on_disk.extend(glob.glob(os.path.join(KNOWLEDGE_FOLDER, ext)))

    # 1. Recupera file già noti dal DB caricato
    known_files = set()
    for entry in rag_system.knowledge_base:
        known_files.add(entry['source'])

    print(f"Documenti totali nella cartella: {len(files_on_disk)}")
    
    new_files_count = 0
    for file_path in files_on_disk:
        filename = os.path.basename(file_path)
        
        # 2. Salta se esiste già
        if filename in known_files:
            continue
            
        # 3. Indicizza solo se nuovo
        print(f"Nuovo file trovato: {filename} -> {Colors.WARNING}Indicizzazione in corso...{Colors.ENDC}")
        content = extract_text_from_file(file_path)
        if content.strip():
            rag_system.ingest_file(filename, content)
            new_files_count += 1
            print(f"{Colors.GREEN}Fatto.{Colors.ENDC}")

    if new_files_count == 0:
        print(f"{Colors.GREEN}Tutti i file sono già indicizzati.{Colors.ENDC}\n")
    else:
        print(f"{Colors.GREEN}Aggiunti {new_files_count} nuovi documenti.{Colors.ENDC}\n")

def chat_loop():
    print(f"{Colors.BOLD}--- COMPANY AGENT CLI (Scrivi 'exit' per uscire) ---{Colors.ENDC}")
    
    # Memoria della sessione CLI
    chat_history = []
    
    while True:
        try:
            print("-" * 50)
            user_query = input(f"{Colors.BOLD}Tu: {Colors.ENDC}").strip()
            
            if user_query.lower() in ['exit', 'quit', 'esci']:
                print("Arrivederci!")
                break
            
            if not user_query:
                continue

            print(f"{Colors.HEADER}... Analisi in corso ...{Colors.ENDC}", end="\r")

            # 1. RETRIEVAL
            retrieved_docs = rag_system.search(user_query, top_k=5)
            best_score = retrieved_docs[0]['similarity'] if retrieved_docs else 0.0

            # Costruiamo la stringa della history per il prompt
            history_str = ""
            for msg in chat_history[-MAX_HISTORY:]:
                role = "UTENTE" if msg['role'] == 'user' else "ASSISTENTE"
                history_str += f"{role}: {msg['content']}\n"

            # 2. ROUTER LOGIC
            if best_score >= SIMILARITY_THRESHOLD:
                # --- RAMO INTERNO (RAG) ---
                mode_label = f"{Colors.GREEN}[INTERNAL RAG - Score: {best_score:.2f}]{Colors.ENDC}"
                
                context_str = "\n".join([f"- {d['text']} (Fonte: {d['source']})" for d in retrieved_docs])
                
                prompt = f"""
                Sei un assistente aziendale intelligente.
                
                STORICO CONVERSAZIONE:
                {history_str}
                
                CONTESTO DOCUMENTI TROVATI:
                {context_str}
                
                NUOVA DOMANDA UTENTE: {user_query}
                
                Rispondi usando contesto e storico. Cita le fonti.
                """
                
                resp = rag_system.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt]
                )
                
                print(f"Modalità: {mode_label}")
                print(f"{Colors.GREEN}Agente:{Colors.ENDC} {resp.text}")
                
                # Mostra fonti uniche
                unique_sources = list(set([d['source'] for d in retrieved_docs]))
                print(f"{Colors.BOLD}Fonti:{Colors.ENDC} {unique_sources}")
                
                response_text = resp.text

            else:
                # --- RAMO ESTERNO (GOOGLE SEARCH) ---
                mode_label = f"{Colors.BLUE}[GOOGLE SEARCH - Score: {best_score:.2f}]{Colors.ENDC}"
                
                google_search_tool = types.Tool(google_search=types.GoogleSearch())
                
                prompt = f"""
                STORICO CONVERSAZIONE:
                {history_str}
                
                NUOVA DOMANDA UTENTE: {user_query}
                
                Rispondi usando Google Search.
                """
                
                resp = rag_system.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(tools=[google_search_tool])
                )
                
                print(f"Modalità: {mode_label}")
                print(f"{Colors.BLUE}Agente:{Colors.ENDC} {resp.text}")
                
                response_text = resp.text

            # Aggiorniamo la memoria locale della CLI
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            print(f"\n{Colors.FAIL}Errore critico: {e}{Colors.ENDC}")

if __name__ == "__main__":
    # Eseguiamo il caricamento PRIMA di avviare la chat
    load_initial_data()
    chat_loop()