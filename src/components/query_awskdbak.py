import json
import time
import os
from typing import List
import logging
from Strands_agent_lookup import get_department_contact  # Import the Strands function

logger = logging.getLogger(__name__)

async def query_awskd(awskd_name: str, query: str, retriever):
    # Initial metadata setup
    safe_name = get_safe_dir_name(awskd_name)
    metadata_file = f"metadata_{safe_name}/add_metadata.json"

    # Check if metadata is in cache and not expired (1 hour cache duration)
    if awskd_name in metadata_cache:
        cache_entry = metadata_cache[awskd_name]
        cache_time = cache_entry.get("timestamp", 0)
        current_time = time.time()

        # Check if cache is fresh (less than 1 hour old)
        if current_time - cache_time < 3600:  # 3600 seconds = 1 hour
            logger.info(f"Using cached metadata for awskd '{awskd_name}'")
            additional_metadata = cache_entry.get("data")
        else:
            logger.info(f"Cached metadata for awskd '{awskd_name}' is expired")

    # If not in cache or cache expired, try loading from disk
    if not additional_metadata and os.path.exists(metadata_file):
        try:
            logger.info(f"Loading additional metadata from disk: {metadata_file}")
            with open(metadata_file, 'r') as f:
                additional_metadata = json.load(f)
            logger.info("Successfully loaded metadata from disk")

            # Update the cache with the fresh data
            metadata_cache[awskd_name] = {
                "data": additional_metadata,
                "timestamp": time.time()
            }
        except Exception as disk_err:
            logger.error(f"Failed to load metadata from disk: {str(disk_err)}")

    # Get the documents from the retriever
    docs = retriever.invoke(query)

    # Get department contact using Strands agent (inference-driven lookup)
    contact_info = get_department_contact(query)

    # Format documents with a neutral title
    doc_texts = []
    for i, doc in enumerate(docs):
        doc_texts.append(f"DOCUMENT {i+1}: {doc.page_content}")
    docs_text = "\n\n".join(doc_texts)

    # Load LLM with the guardrail if available
    llm = load_llm(guardrail_id=guardrail_id)
    logger.info(f"Sending direct query to LLM {'with guardrail' if guardrail_id else ''}")

    # Prepare messages with context and query
    messages = [
        {"type": "system", "content": system_prompt},  # Define system_prompt as needed
        {"type": "human", "content": f"Here is the complete context:\n\n{docs_text}\n\nQuery: {query}"}
    ]

    response = llm.invoke(messages)

    # Combine RAG result with contact info
    result = {
        "answer": response.content,
        "context": docs,
        "metadata_included": False,  # No longer including full JSON
        "response_metadata": response.response_metadata,
        "contact_info": contact_info
    }

    return result

# Placeholder functions (assumed from context)
def get_safe_dir_name(awskd_name: str) -> str:
    # Implementation to sanitize awskd_name for directory use
    pass

def load_llm(guardrail_id: str = None):
    # Implementation to load Bedrock LLM with optional guardrail
    pass

def decimal_default(obj):
    # Custom JSON encoder for decimal types
    pass

metadata_cache = {}  # Global cache dictionary
system_prompt = "You are a helpful assistant for operations management queries. Use the provided context to answer."  # Example system prompt