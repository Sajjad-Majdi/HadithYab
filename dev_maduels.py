from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from vecs import IndexMeasure, IndexMethod
import vecs
import os
import json
import time
import math
import torch


_MODEL = "jinaai/jina-embeddings-v3"
_DEVICE = 0 if torch.cuda.is_available() else -1
_DB_URL = os.getenv("CONNECTION_STRING")
_DIM = 1024

# hadiths.json structuer:
'''
{
    "_id":
        "$oid": str
    },
    "id": int,
    "hadithText": str,
    "farsiTranslation": str,
    "source": str,
    "from": str
}
'''

# embedded_json.json structure:
'''
{
    "id": int,
    "embedding_fa": List[float],
    "text_fa": str,
    "text_ar": str,
    "source": str,
    "from": str
}
'''


# Function to embed one record
def embed(text: str, embed_model=_MODEL):
    """
    Embed a single text string using the specified model.
    """

    if type(embed_model) == str:
        model = SentenceTransformer(
            embed_model, device=_DEVICE, trust_remote_code=True)
    elif type(embed_model) == SentenceTransformer:
        model = embed_model
    else:
        raise ValueError(
            "Invalid model type. Must be str or SentenceTransformer.")
    # encode all texts (float32, normalize)
    embeddings = model.encode(
        text,
        task="retrieval.query",
        convert_to_numpy=True,
        device=_DEVICE,
        normalize_embeddings=True
    ).tolist()
    return embeddings


# GPU-optimized embedding function (json -> json)
def embed_json(
    input_path: str,
    output_path: str,
    text_field: str,
    model_name: str = _MODEL,
    batch_size: int = 128,
    limit: int = None,
):
    """
    Load a JSON array of objects, embed the specified text field for each record
    up to `limit` records if provided (else entire file), on GPU with float32 precision,
    and write out a new JSON with an added 'embedding' field for each record.

    Args:
        input_path: path to input JSON file (list of dicts)
        output_path: path where to save output JSON
        text_field: key in each dict to embed
        model_name: sentence-transformers model name
        batch_size: number of texts to process per batch
        limit: max number of records to embed; None for all
    """
    # load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # apply limit
    if limit is not None:
        data = data[:limit]

    # init model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name, device=device, trust_remote_code=True)

    # extract texts
    texts = [item.get(text_field, "") for item in data]

    # encode all texts (float32, normalize)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        task="retrieval.query",
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=True
    )

    # attach embeddings
    for obj, emb in zip(data, embeddings):
        obj['embedding'] = emb.tolist()

    # write out
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return output_path


# Dynamic upsert function(one record, to any table) with "supabase" library, API and base URL
def upsert(
    table: str,
    data: dict,
    supabase_url: str = None,
    supabase_key: str = None,
    embedding_field: str = None,
    text_field: str = None,
    embed_fn=None,
):
    """
    Generic upsert function for Supabase tables.

    Args:
        table (str): Table name to upsert into.
        data (dict): Data to upsert.
        supabase_url (str, optional): Supabase URL. If None, uses env var SUPABASE_URL.
        supabase_key (str, optional): Supabase Key. If None, uses env var SUPABASE_KEY.
        embedding_field (str, optional): Name of the embedding field to fill.
        text_field (str, optional): Name of the text field to embed.
        embed_fn (callable, optional): Function to generate embedding from text.
    """
    supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
    supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase URL or Key not found in environment variables.")

    supabase = create_client(supabase_url, supabase_key)

    # Dynamically handle embedding if required
    if embedding_field and text_field and embed_fn:
        if text_field in data and embedding_field not in data:
            data[embedding_field] = embed_fn(data[text_field])

    try:
        supabase.table(table).upsert(data).execute()
        print(
            f"Upserted record {data.get('id', '[no id]')} into '{table}' successfully")
        return True
    except Exception as e:
        print(
            f"Error in upserting record {data.get('id', '[no id]')} into '{table}': {e}")
        return False


# Upsert function with "supabase" library, API and base URL
def upsert_from_embedded_json(
    json_file_path: str,
    batch_size: int = 100
) -> dict:
    """
    Second phase: Load pre-embedded records from JSON and upsert in batches of 100.
    """
    print(f"Starting batch upsert from pre-embedded JSON: {json_file_path}")
    stats = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "batches_processed": 0,
        "batches_failed": 0
    }

    try:
        # Get Supabase credentials
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL or Key not found in environment variables.")

        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Supabase client initialized.")

        # Load pre-embedded JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_records = len(data)
        print(f"Loaded {total_records} pre-embedded records from JSON file")

        # Process in batches of 100
        for i in range(0, total_records, batch_size):
            batch = data[i:i+batch_size]
            batch_number = i // batch_size + 1
            total_batches = (total_records + batch_size - 1) // batch_size

            print(
                f"Processing batch {batch_number}/{total_batches} ({len(batch)} records)")

            try:
                # Prepare batch for upsert
                batch_to_upsert = []
                for record in batch:
                    batch_to_upsert.append({
                        "id": record["id"],
                        "embedding_fa": record["embedding_fa"],
                        "text_fa": record["text_fa"],
                        "text_ar": record["text_ar"],
                        "source": record["source"],
                        "from": record["from"]
                    })

                # Upsert the batch
                response = supabase.table("Hadiths").upsert(
                    batch_to_upsert,
                    on_conflict="id"
                ).execute()

                stats["successful"] += len(batch)
                stats["batches_processed"] += 1
                print(f"Batch {batch_number} completed successfully")

            except Exception as e:
                print(f"Error in batch {batch_number}: {e}")
                stats["failed"] += len(batch)
                stats["batches_failed"] += 1

            # Update total processed count
            stats["total_processed"] += len(batch)

            # Progress update
            print(f"Progress: {stats['total_processed']}/{total_records} "
                  f"(Success: {stats['successful']}, Failed: {stats['failed']})")

        print(f"Batch upsert completed. "
              f"Total: {stats['total_processed']}, "
              f"Success: {stats['successful']}, "
              f"Failed: {stats['failed']}")

    except Exception as e:
        print(f"Error in batch upsert process: {str(e)}")

    return stats


# Upsert function with "vecs" library from json, using connection string
def upsert_vecs(
    json_file_path: str,
    collection_name: str,
    batch_size: int = 1,
    embedding_field: str = "embedding",
    metadata_fields: list[str] = None,
    limit: int = None,
):
    """
    Creates/gets a vecs collection and upserts data from a JSON file in batches.
    Throws an error if any record lacks the required embedding or metadata fields.

    Args:
        json_file_path: Path to the JSON file containing data.
        collection_name: Name of the vecs collection to create/use.
        batch_size: Number of records to upsert per batch.
        embedding_field: Key in each record whose value is the vector to upsert.
        metadata_fields: List of record keys to include as metadata (default: none).
        limit: Maximum number of records to process; None for all.

    Returns:
        dict with collection name, total_records, upserted_count, failed_count, duration.
    """
    from tqdm import tqdm
    metadata_fields = metadata_fields or []
    start_time = time.time()
    successful = failed = 0

    # Step 1: Load data
    print(f"Loading data from '{json_file_path}'...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # apply limit
    if limit is not None:
        data = data[:limit]
    total_records = len(data)
    if total_records == 0:
        raise ValueError("No records to process after applying limit.")
    print(f"✅ Loaded {total_records} records (limit={limit}).")

    # Step 2: Connect to vector DB and collection
    print("Connecting to vector database...")
    try:
        vx = vecs.create_client(_DB_URL)
        collection = vx.get_or_create_collection(
            name=collection_name, dimension=_DIM)
    except Exception as e:
        raise ConnectionError(f"Failed to connect or get collection: {e}")
    print(f"✅ Connected and ready collection '{collection_name}'.")

    # Step 3: Upsert in batches
    num_batches = math.ceil(total_records / batch_size)
    print(
        f"Starting upsert: {num_batches} batches of up to {batch_size} records each.")

    # iterate with progress bar
    for batch_idx in tqdm(range(num_batches), desc="Upserting batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_records)
        batch = data[batch_start:batch_end]
        records_to_upsert = []

        for rec in batch:
            # Validate embedding
            if embedding_field not in rec or rec[embedding_field] is None:
                raise KeyError(
                    f"Missing embedding '{embedding_field}' in record: {rec}")
            emb = rec[embedding_field]
            # Validate metadata
            meta = {}
            for field in metadata_fields:
                if field not in rec or rec[field] is None:
                    raise KeyError(
                        f"Missing metadata '{field}' in record: {rec}")
                meta[field] = rec[field]

            # prepare record
            record_id = str(batch_start + len(records_to_upsert) + 1)
            records_to_upsert.append((record_id, emb, meta))

        # execute upsert
        try:
            collection.upsert(records=records_to_upsert)
            successful += len(records_to_upsert)
        except Exception as e:
            print(f"❌ Error upserting batch {batch_idx+1}: {e}")
            failed += len(records_to_upsert)

    duration = time.time() - start_time

    # Step 4: Summary
    print("--- Upsert Complete ---")
    print(f"Collection: {collection_name}")
    print(f"Total records: {total_records}")
    print(f"Successful upserts: {successful}")
    print(f"Failed upserts: {failed}")
    print(f"Duration: {duration:.2f}s")

    return {
        "collection": collection_name,
        "total_records": total_records,
        "successful": successful,
        "failed": failed,
        "duration_seconds": duration
    }

# Function to create a vector index


def create_index(db_url: str, collection_name: str):
    """
    Creates an index on a vecs collection.

    Args:
        db_url: The connection string to the vecs database.
        collection_name: The name of the vecs collection to create the index on.
    """
    from sqlalchemy import text
    try:
        vx = vecs.create_client(db_url)
        # ses = vx.Session()
        # ses.execute(text("alter role authenticated set statement_timeout = '500s';"))
        collection = vx.get_or_create_collection(
            name=collection_name, dimension=_DIM)

        print("✅ Connected to collection.")
    except Exception as e:
        print(f"❌ Error connecting to collection: {e}")
        return False

    try:
        print("🔨 Creating index...")
        collection.create_index(
            method=IndexMethod.hnsw,
            measure=IndexMeasure.cosine_distance)
        print("✅ Index created successfully.")
        return True
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

# what types of data have shape method?


# Main execution
if __name__ == "__main__":

    '''
    create_index(
        db_url=_DB_URL,
        collection_name="jira_hadiths"
    )
    '''

    # Embedding and saving to JSON
    '''embed_json(
        input_path="data/hadiths.json",
        output_path="st_embedded_json.json",
        text_field="farsiTranslation",
        model_name=_MODEL,
        batch_size=256
    )
    '''
    # Upsert from embedded JSON

    '''
    upsert_vecs(
        json_file_path="data/jina_embedded_json.json",
        batch_size=100,
        collection_name="jira_hadiths",
        embedding_field="embedding",
        metadata_fields=["farsiTranslation", "hadithText", "source", "from"]
    )
    '''

    # Index creation

    '''
    create_index_sql(
        db_url=_DB_URL,
        collection_name="myhadiths",
        dim=_DIM
    )
    '''

    # Example query

    '''
    text = "مانع فساد جوانان"
    results = find_similar_records(
        query_text=text, n=5, collection_name="myhadiths")
    print("Results:" + str(results))
    '''
