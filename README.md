# Jira Hadith Search

A semantic search application enabling users to find Hadith records by entering natural language queries. It leverages advanced embeddings from Jina AI or Hugging Face to retrieve semantically similar results from a Supabase-powered vector database.

## Live Demo

You can try the application live at:  
[https://hadithyab.com](https://hadithyab.onrender.com/)

## Technical Overview

- **Frontend:** HTML, CSS, JS (via Jinja2 templates)
- **Backend:** Python Flask
- **Embedding Model:** [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
- **Dataset:** [`IslamShia/shia-hadith`](https://github.com/IslamShia/shia-hadith)
- **Vector Store:** Supabase + `vecs` library

## Features

- Web-based search interface built with Flask and Jinja2
- Embedding generation via Jina AI or Hugging Face Inference API
- Similarity search using the `vecs` library and Supabase vector store
- Returns Arabic text, Farsi translation, source, and narrator information

## Embedding Model

This application uses the `jina-embeddings-v3` model to generate semantic vector representations of Hadith records. The embeddings are generated from the Farsi translations of Hadith records for the vector database.

You can utilize the model in two ways:

- **Jina AI Embeddings:** Access the Jina AI embedding service (default: `jina-embeddings-v3`) for high-quality, 1024-dimensional embeddings via API.
- **Hugging Face Embeddings:** Use the model through the Hugging Face Inference API with your own HF API key. If inference is disabled on the official Jina AI page, you can use the compatible model hosted at [Sajjad313/my-Jira-embedding-v3](https://huggingface.co/Sajjad313/my-Jira-embedding-v3).

## Prerequisites

- Python 3.8 or newer
- A virtual environment (recommended)
- A Supabase account with a vector store set up
- A `.env` file (optional) to store environment variables locally
- Environment variables:
  - `CONNECTION_STRING`: Connection URI for your Supabase vector store (used by search)
  - `JINA_API_KEY`: API key for Jina AI embedding service
  - `HF_API_KEY`: API key for Hugging Face Inference API
  - `COLLECTION_NAME` and `NUM_RESULTS` (optional) to override default search collection name and result count

## Installation

1. Clone the repository or download the source code:

   ```bash
   git clone <repository_url>
   cd production
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or macOS:
   source venv/bin/activate
   ```

3. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables (replace with your own values):

   ```bash
   set CONNECTION_STRING="your_connection_string"
   set JINA_API_KEY="your_jina_api_key"
   set HF_API_KEY="your_hf_api_key"
   ```

   On Unix/macOS, use `export` instead of `set`.

## Usage

1. Run the Flask server:

   uncomment app.run() at the end of the `flask_server.py` then

   then run this in terminal:

   ```bash
   python flask_server.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Enter a search query in the input box and submit to see the top similar Hadith records.

## Project Structure

```
.
├── config.py              # Application configuration and env var loader
├── flask_server.py        # Flask app entry point
├── madules.py             # Embedding and similarity search utilities
├── dev_maduels.py         # Batch embedding and upsert utilities (developer use)
├── requirements.txt       # Python dependencies
├── .env                   # Local environment variable overrides (optional)
├── templates/
│   └── index.html         # Jinja2 template for search UI
└── README.md              # Project overview and instructions
```

## Production Deployment

This application is ready for production deployment behind a WSGI server like **Waitress** or **Gunicorn**.

### Production Deployment

- On Windows, use Waitress (installed via requirements.txt):

  ```powershell
  waitress-serve --listen=0.0.0.0:5000 flask_server:app
  ```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
