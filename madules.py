import requests
from huggingface_hub import InferenceClient
import numpy as np
import vecs
from vecs import IndexMeasure
import logging
from config import Config


# Configuration
_DIM = 1024

DB_URL = Config.DB_URL
JINA_API = Config.JINA_API
HF_API = Config.HF_API
MY_REPO = Config.HF_REPO

logger = logging.getLogger(__name__)


def get_HF_embeddings(text, api_key=HF_API, model=MY_REPO, return_raw=False, mean_pool=True, l2_normalize=True):
    client = InferenceClient(api_key=api_key)
    try:
        result = client.feature_extraction(
            text=text,
            model=model
        )
        if return_raw:
            return result
        if mean_pool:
            mean_pooled = np.mean(result[0], axis=0)
            if l2_normalize:
                norm = np.linalg.norm(mean_pooled)
                if norm > 0:
                    mean_pooled = mean_pooled / norm
            return mean_pooled
        return result[0]
    except Exception:
        logger.error("Error fetching HuggingFace embeddings", exc_info=True)
        return None


def get_jina_embeddings(text, api_key=JINA_API, model="jina-embeddings-v3", task="retrieval.query"):
    """
    Fetches mean-pooled embeddings for input texts from Jina AI's API.
    Args:
        text (list or str): A list of input texts or a single text string to embed.
        api_key (str, optional): Jina API key for authentication. Defaults to the value of JINA_API.
        model (str, optional): The model name to use for generating embeddings. Defaults to "jina-embeddings-v3".
        task (str, optional): The embedding task type. Options include:
            - "retrieval.query": For query embeddings in asymmetric retrieval tasks.
            - "retrieval.passage": For passage embeddings in asymmetric retrieval tasks.
            - "separation": For embeddings in clustering and re-ranking applications.
            - "classification": For embeddings in classification tasks.
            - "text-matching": For embeddings in tasks quantifying similarity between two texts.
    Returns:
        np.ndarray: An array of embeddings corresponding to the input texts
        - shape: (n , 1024) list of lists 
    Raises:
        requests.HTTPError: If the API request fails.
    Example:
        >>> embeddings = get_jina_embeddings(["Hello world", "How are you?"], api_key="your_api_key")
    """

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "task": task,
        "input": text
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    # Extract embeddings from the response
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings)


# Query function with "vecs" library, using connection string
def find_similar_records(query_text: str, n: int, collection_name: str) -> list:
    """
    Finds the n most similar records to the query_text in a vecs collection,
    using the globally defined embed function and returns their metadata
    ordered by similarity.

    Args:
        query_text: The text to search for.
        n: The number of similar records to return.
        collection_name: The name of the vecs collection to query.

    Returns:
        A list of metadata dictionaries for the n most similar records,
        ordered by similarity score (closest first), or an empty list
        if an error occurs or no records are found.
    """
    logger.info("Searching for %d records similar to '%s...' in collection '%s'",
                n, query_text[:50], collection_name)

    try:
        # 1. Connect to vecs
        vx = vecs.create_client(DB_URL)

        # 2. Get the collection
        collection = vx.get_or_create_collection(
            name=collection_name, dimension=_DIM)
        logger.debug("Accessed collection '%s'", collection_name)

        # 3. Embed the query text using the global embed function
        logger.debug("Embedding query text")
        try:
            query_vector = get_jina_embeddings(query_text, api_key=JINA_API)
            if not isinstance(query_vector, (list, np.ndarray)):
                raise TypeError(
                    f"Embedding function did not return a list or numpy array, got {type(query_vector)}")
            logger.debug("Query text embedded")
        except Exception as e:
            logger.error("Error fetching embeddings: %s", e)
            raise

        # 4. Perform the similarity search, requesting metadata and value (score)
        logger.debug("Querying collection for top %d results", n)
        # Query now returns list of tuples: (id, metadata, score)
        query_results = collection.query(
            data=query_vector[0],
            limit=n,
            measure=IndexMeasure.cosine_distance,  # Or the measure your index uses
            include_metadata=True,            # <<< Get metadata directly
            include_value=True                # <<< Get the distance/similarity score
        )
        logger.debug("Query completed. Found %d results", len(query_results))

        # 5. Extract metadata directly from query results
        # The results are already ordered by similarity score (distance) by the query method.
        # Each item in query_results is expected to be (id, metadata, score)
        metadata_list = [[id, distance, metadata]
                         for id, distance, metadata in query_results]

        return metadata_list

    # Specific errors now bubble up to be handled by Flask
    except Exception:
        logger.exception("Unexpected error during similarity search")
        raise
