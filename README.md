# Jira Hadith Search

A semantic search application enabling users to find Hadith records by entering natural language queries. It leverages advanced embeddings from Jina AI or Hugging Face to retrieve semantically similar results from a Supabase-powered vector database.

## Technical Overview

- Frontend: `HTML, CSS, JS`
- Backend: `Python Flask`
- Embedding Model: [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
- Dataset for Vector Database: [`IslamShia/shia-hadith`](https://github.com/IslamShia/shia-hadith)

## Features

- Web-based search interface built with Flask and Jinja templates
- Embedding generation via Jina AI or Hugging Face Inference API
- Similarity search using the `vecs` library and Supabase vector store
- Returns Arabic text, Farsi translation, source, and narrator information

## Embedding_model

This application used `jina-embeddings-v3` as embedding model:

you can find it both ways

- **Jina AI Embeddings**: Use Jina AI’s embedding service (default model: `jina-embeddings-v3`) via the `JINA_API_KEY` to generate high-quality, 1024-dimensional vector representations.
- **Hugging Face Embeddings**: Use the Hugging Face Inference API or with a pre-trained sentence-transformers model via our Hugging Face page `Sajjad313/my-Jira-embedding-v3` with `HF_API_KEY`; or just download the model from their own page: `jinaai/jina-embeddings-v3`

Embeddings are automatically normalized before being indexed in Supabase for efficient similarity search in our application.

## Prerequisites

- Python 3.8 or newer
- A virtual environment (recommended)
- A Supabase account with a vector store set up
- Environment variables:
  - `CONNECTION_STRING`: Connection URI for your Supabase vector store
  - `JINA_API_KEY`: API key for Jina AI embedding service
  - `HF_API_KEY`: API key for Hugging Face Inference API

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

## Usage

1. Run the Flask server:

   ```bash
   python flask_server.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Enter a search query in the input box and submit to see the top similar Hadith records.

## Project Structure

```
production/
├── flask_server.py     # Flask app entry point
├── madules.py          # Embedding and search utilities
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Jinja2 template for search UI
└── README.md           # Project overview and instructions
```

## Troubleshooting

- Ensure all environment variables are set correctly.
- Verify network access for API calls to Jina AI and Hugging Face.
- Turn off debug mode in production by setting `app.run(debug=False)`.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
