# 🤖 Company AI Assistant

Un Agente AI ibrido progettato per rispondere a domande aziendali utilizzando documenti interni (RAG) e a domande generiche utilizzando Google Search.

Il sistema decide autonomamente ("Agentic decision-making") quale fonte utilizzare basandosi sulla pertinenza della domanda.

## 🚀 Funzionalità Principali

* **Hybrid Knowledge:** Combina una Knowledge Base locale (documenti PDF/MD/TXT) con la ricerca web in tempo reale.
* **Semantic Router:** Un algoritmo analizza la similarità della domanda con i documenti interni. Se lo score è basso (< 0.5), l'agente attiva automaticamente la ricerca web.
* **Smart Chunking:** Divisione intelligente del testo per preservare il contesto delle frasi.
* **Persistent Memory:** Il database vettoriale viene salvato su disco (`vector_store.json`) per avvii rapidi.
* **Multi-Interface:**
    * 🌐 **Web UI:** Interfaccia moderna responsive (Flask).
    * 💻 **CLI:** Interfaccia da riga di comando per operazioni rapide.
* **Context Aware:** Mantiene la cronologia della chat per conversazioni naturali.

## 🛠️ Architettura e Scelte Tecniche

### Stack Tecnologico
* **Backend:** Python 3.10+, Flask (Web Server).
* **AI Core:** Google Gemini 1.5 Flash (via `google-genai` SDK).
* **Embedding:** Google `text-embedding-004`.
* **Vector Store:** Implementazione locale in-memory con persistenza JSON (Numpy per calcoli di similarità coseno).
* **Deployment:** Google Cloud Run (Dockerless setup).

### Design Rationale
1.  **Perché Gemini 1.5 Flash?** Scelto per la bassa latenza e il costo ridotto, ideale per un assistente real-time.
2.  **Router Semantico vs Tool Calling:** È stato implementato un router esplicito basato su soglia di similarità (Threshold 0.5) per avere un controllo deterministico e ridurre le allucinazioni su dati aziendali critici.
3.  **Persistenza JSON:** Per questo prototipo, un file JSON evita la complessità di gestire un database vettoriale esterno (come Milvus o Pinecone), mantenendo l'app portabile e semplice.

---

## 📦 Installazione e Configurazione

### Prerequisiti
* Python 3.10 o superiore
* Google Cloud CLI installata
* Account Google Cloud con API Vertex AI abilitate

### 1. Setup Locale

Clona il repository:
```bash
git clone [https://github.com/TUO_USERNAME/company-assistant.git](https://github.com/TUO_USERNAME/company-assistant.git)
cd company-assistant
```

Crea un ambiente virtuale e installa le dipendenze:

```bash
python -m venv env_agente
source env_agente/bin/activate  # Su Windows: .\env_agente\Scripts\activate
pip install -r requirements.txt
```

Autenticati con Google Cloud:

```bash
gcloud auth application-default login
```

### 1. Preparazione Dati
Inserisci i tuoi documenti aziendali (PDF, Markdown, TXT) nella cartella:

```Plaintext
Knowledge Base/
```
Il sistema indicizzerà automaticamente i nuovi file all'avvio.

### 3. Avvio
Interfaccia Web:

```bash
python main.py
# Apri http://localhost:8080
```
Interfaccia CLI:
```bash
python cli.py
```
☁️ Deployment su Google Cloud Run
L'applicazione è già stata deployata ed è accessibile online. Puoi provarla direttamente qui:
👉 [https://company-assistant-573847000470.europe-west1.run.app](https://company-assistant-573847000470.europe-west1.run.app)
