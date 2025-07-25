import os
import json
import re
from uuid import uuid4
from datetime import datetime
from typing import List
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, JSONLoader,
    BSHTMLLoader, Docx2txtLoader
)
from langchain_aws import BedrockEmbeddings
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Assuming these exist globally
logger = print  # replace with actual logger if needed
awskd_stores = {}
awskd_status = {}
awskd_table = None  # This should be your boto3 DynamoDB resource or table object

async def create_awskd(
    awskd_name: str,
    source_dir: str,
    awskd_description: str = "",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    file_list: List[str] = None,
    guardrail_name: str = "DefaultGuardrail",
    has_metadata: bool = False,
    data_product_type: str = None,
    prompt_template: str = None
) -> str:
    task_id = str(uuid4())
    awskd_status[task_id] = {
        "awskd_name": awskd_name,
        "status": "processing",
        "message": "Loading documents",
        "timestamp": datetime.now().isoformat()
    }

    try:
        files = file_list or []
        if not files:
            if not os.path.exists(source_dir):
                raise ValueError(f"Directory {source_dir} does not exist")
            for root, _, filenames in os.walk(source_dir):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        logger(f"Found {len(files)} files")

        documents = []
        docs = []

        def safe_load(loader_cls, file_path, **kwargs):
            try:
                loader = loader_cls(file_path, **kwargs) if kwargs else loader_cls(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger(f"Successfully loaded: {file_path}")
            except Exception as e:
                logger(f"Error loading {file_path}: {str(e)}")

        for f in files:
            lower = f.lower()
            if lower.endswith(".pdf"):
                safe_load(PyPDFLoader, f)
            elif lower.endswith(".txt"):
                safe_load(TextLoader, f)
            elif lower.endswith(".csv"):
                safe_load(CSVLoader, f)
            elif lower.endswith(".json"):
                safe_load(JSONLoader, f, jq_schema=".", text_content=False)
            elif lower.endswith(('.html', '.htm')):
                base_name = os.path.splitext(os.path.basename(f))[0]
                markdown_chunks = [x for x in os.listdir(source_dir) if x.startswith(base_name + '_part') and x.endswith('.md')]
                if markdown_chunks:
                    for chunk in markdown_chunks:
                        safe_load(TextLoader, os.path.join(source_dir, chunk), encoding="utf-8")
                else:
                    md_path = os.path.join(source_dir, base_name + ".md")
                    if os.path.exists(md_path):
                        safe_load(TextLoader, md_path, encoding="utf-8")
                    else:
                        safe_load(BSHTMLLoader, f, bs_kwargs={'features': 'lxml'})
            elif lower.endswith(".docx"):
                safe_load(Docx2txtLoader, f)

        if not documents:
            raise ValueError(f"No compatible documents found in {source_dir}")

        logger(f"Loaded {len(documents)} documents")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(documents)
        logger(f"Split into {len(docs)} chunks")

        embeddings = BedrockEmbeddings()
        
        # Replace FAISS with OpenSearch
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', awskd_name)
        index_name = f"awskd_{safe_name}".lower()  # OpenSearch index names must be lowercase
        
        # Get AWS credentials for OpenSearch
        credentials = boto3.Session().get_credentials()
        region = boto3.Session().region_name or 'us-east-1'
        auth = AWSV4SignerAuth(credentials, region, 'aoss')
        
        # Create OpenSearch client configuration
        host = 'awskdPlatform.region.aoss.amazonaws.com'  # Replace region with actual region
        
        opensearch_client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        
        # Create OpenSearch vector store
        vector_store = OpenSearchVectorSearch.from_documents(
            docs,
            embeddings,
            opensearch_url=opensearch_client,
            index_name=index_name,
            engine="faiss",  # Or another supported engine
            bulk_size=500    # Optimize bulk indexing
        )
        
        awskd_stores[awskd_name] = vector_store
        # No need to save_local since OpenSearch persists on the service

        awskd_id = f"awskd-{uuid4()}"
        txn_id = f"txn-{uuid4()}"
        current_time = datetime.now().isoformat()

        total_size = 0
        all_files_metadata = []

        for path in files:
            try:
                name = os.path.basename(path)
                size = os.path.getsize(path)
                ext = os.path.splitext(path)[1].lower()
                total_size += size
                if ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
                    summary = f"Document: {name} ({ext.upper()} file, {size} bytes)"
                else:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(3000)
                        summary = awskd_description
                all_files_metadata.append({
                    "name": name,
                    "path": path,
                    "size": size,
                    "type": ext,
                    "upload_date": current_time,
                    "summary": summary
                })
            except Exception as e:
                all_files_metadata.append({
                    "name": os.path.basename(path),
                    "path": path,
                    "summary": "Unable to process metadata for this document"
                })

        metadata_dir = f"metadata_{safe_name}"
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "file_list_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_files_metadata, f, indent=2)

        try:
            items = awskd_table.scan(
                FilterExpression="awskd_name = :name",
                ExpressionAttributeValues={":name": awskd_name}
            ).get("Items", [])
            existing_id = items[0]["awskd_id"] if items else None
            expression_values = {
                ":txn": txn_id,
                ":status": "PENDING PUBLISH",
                ":reason": "Pending KDE Validation",
                ":description": awskd_description,
                ":timestamp": current_time,
                ":index_name": f"awskd_{safe_name}".lower(),
                ":input_size": total_size,
                ":chunk_count": len(docs),
                ":chunk_size": chunk_size,
                ":chunk_overlap": chunk_overlap,
                ":guardrail": guardrail_name,
                ":file_list": all_files_metadata,
                ":source_dir": source_dir,
                ":file_count": len(all_files_metadata),
                ":metadata_path": metadata_path
            }
            if guardrail_name and re.match(r"^[a-z0-9]{10,12}$", guardrail_name):
                expression_values[":guardrail_id"] = guardrail_name

            if existing_id:
                awskd_table.update_item(
                    Key={"awskd_id": existing_id},
                    UpdateExpression="SET txn_id=:txn, awskd_status=:status, awskd_update_reason=:reason, "
                                     "awskd_description=:description, awskd_last_update_dt=:timestamp, "
                                     "awskd_index_name=:index_name, awskd_input_size=:input_size, "
                                     "awskd_chunk_count=:chunk_count, awskd_chunk_size=:chunk_size, "
                                     "awskd_chunk_overlap=:chunk_overlap, awskd_guardrail_name=:guardrail, "
                                     "awskd_file_count=:file_count, awskd_file_metadata_path=:metadata_path, "
                                     "awskd_source_dir=:source_dir",
                    ExpressionAttributeValues=expression_values
                )
                awskd_id = existing_id
            else:
                item_dict = {
                    "awskd_id": awskd_id,
                    "txn_id": txn_id,
                    "awskd_name": awskd_name,
                    "awskd_description": awskd_description,
                    "awskd_index_name": f"awskd_{safe_name}".lower(),
                    "awskd_status": "PENDING PUBLISH",
                    "awskd_update_reason": "Pending KDE Validation",
                    "awskd_data_product_type": data_product_type or "Unstructured",
                    "awskd_approved_prompt_template": prompt_template or "",
                    "awskd_guardrail_name": guardrail_name,
                    "awskd_tool_kit": "None",
                    "awskd_creation_dt": current_time,
                    "awskd_last_update_dt": current_time,
                    "awskd_input_size": total_size,
                    "awskd_chunk_count": len(docs),
                    "awskd_chunk_size": chunk_size,
                    "awskd_chunk_overlap": chunk_overlap,
                    "awskd_created_by": "kdeadmin",
                    "awskd_file_metadata_path": metadata_path,
                    "awskd_file_count": len(all_files_metadata),
                    "awskd_source_dir": source_dir,
                    "awskd_has_metadata": has_metadata
                }
                if guardrail_name:
                    item_dict["awskd_guardrail_id"] = guardrail_name
                awskd_table.put_item(Item=item_dict)

        except Exception as ddb_error:
            logger(f"Failed to update DynamoDB: {str(ddb_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to update awskd metadata: {str(ddb_error)}")

        awskd_status[task_id] = {
            "awskd_name": awskd_name,
            "status": "completed",
            "publish_status": "PENDING PUBLISH",
            "message": f"awskd '{awskd_name}' created successfully",
            "timestamp": current_time,
            "doc_count": len(docs)
        }
        return awskd_id, txn_id, awskd_status[task_id]

    except Exception as e:
        awskd_status[task_id] = {
            "awskd_name": awskd_name,
            "status": "failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        logger(f"Error creating TKD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create TKD: {str(e)}")
