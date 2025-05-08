import os


class Config:
    DEBUG = False
    TESTING = False

    # App-specific
    COLLECTION_NAME = "jira_hadiths"
    NUM_RESULTS = 50

    # External services
    DB_URL = os.environ.get("CONNECTION_STRING", "")
    JINA_API = os.environ.get("JINA_API_KEY", "")
    HF_API = os.environ.get("HF_API_KEY", "")
    HF_REPO = "Sajjad313/my-Jira-embedding-v3"

    # Logging
    LOG_LEVEL = "INFO"
