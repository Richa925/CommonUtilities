from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from datetime import datetime
import json
import boto3
import os

# AWS Bedrock setup (replace with your credentials and region)
# Note: Ensure AWS CLI is configured or set environment variables for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # Replace with your region
)
llm = Bedrock(
    client=bedrock_client,
    model_id="anthropic.claude-v2",  # Or use "amazon.titan-text-express-v1" or other Bedrock models
    model_kwargs={"max_tokens_to_sample": 512, "temperature": 0.7}  # Adjust as needed for Claude or Titan
)

# OpenSearch Serverless setup
# Replace with your OpenSearch Serverless endpoint (host), index name, and credentials
opensearch_host = "your-opensearch-serverless-endpoint.us-east-1.aoss.amazonaws.com"  # e.g., "abc123.us-east-1.aoss.amazonaws.com"
opensearch_index = "hierarchyMd_index"
opensearch_http_auth = ("username", "password")  # If using basic auth; otherwise use AWS IAM with sigv4

# Step 1: Setup metadata vector store (run once to index)
# Note: This assumes you have created a collection in OpenSearch Serverless with vector support (e.g., knn_vector field)
metadata_json = json.load(open("hierarchyMd.json"))
chunks = [
    {
        "id": subsection,
        "text": (
            f"{subsection} procedure, main_section: {details['main_section']}, "
            f"subsection: {details['subsection']}, department: {details['department']}, "
            f"contacts: {details['contacts']['person']}, email: {details['contacts']['email']}, "
            f"phone: {details['contacts']['phone']}, operations_support_line: {details['contacts']['operations_support_line']}, "
            f"address: {details['contacts'].get('address', 'N/A')}, website: {details['contacts'].get('website', 'N/A')}, "
            f"documents: {','.join(details['contacts'].get('documents', []))}"
        ),
        "metadata": {
            "main_section": details["main_section"],
            "subsection": details["subsection"],
            "department": details["department"],
            "contacts": details["contacts"]
        }
    }
    for subsection, details in metadata_json["hierarchyMd"].items()
]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and index into OpenSearch
vector_store = OpenSearchVectorSearch.from_texts(
    texts=[chunk["text"] for chunk in chunks],
    embedding=embeddings,
    metadatas=[chunk["metadata"] for chunk in chunks],
    index_name=opensearch_index,
    opensearch_url=f"https://{opensearch_host}",
    http_auth=opensearch_http_auth,  # Or use engine="aoss" for serverless with IAM
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="aoss"  # Specify for AWS OpenSearch Serverless
)

# Step 2: Generate initial response
# Primary retriever (your document vector store - assume similar OpenSearch setup)
# For this example, assume primary_vector_store is already set up similarly
primary_vector_store = OpenSearchVectorSearch(
    index_name="your_primary_index",  # Replace with your primary docs index
    embedding_function=embeddings,
    opensearch_url=f"https://{opensearch_host}",
    http_auth=opensearch_http_auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="aoss"
)
primary_retriever = primary_vector_store.as_retriever(search_kwargs={"k": 3})  # Adjust as needed

rag_prompt = PromptTemplate(
    input_variables=["context", "user_query"],  # Note: RetrievalQA uses "context" for retrieved_docs
    template=(
        "Using only the provided documents: {context}\n"
        "Answer the query: {user_query}\n"
        "Provide a concise, step-by-step explanation of the procedure.\n"
        "Do not include contact details or department information.\n"
        "If no document matches, return: 'I don’t know.'"
    )
)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=primary_retriever,
    chain_type_kwargs={"prompt": rag_prompt}
)
user_query = "What is the internal procedure for buying a bond?"
initial_response = rag_chain.run(user_query)

# Step 3: Retrieve contact details
# Reuse the same embeddings
vector_store = OpenSearchVectorSearch(
    index_name=opensearch_index,
    embedding_function=embeddings,
    opensearch_url=f"https://{opensearch_host}",
    http_auth=opensearch_http_auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="aoss"
)
search_query = "bond_purchase procedure contacts department"
retrieved_docs = vector_store.similarity_search_with_score(search_query, k=1)
if not retrieved_docs:
    extracted_details = {
        "main_section": "Unknown",
        "subsection": "Unknown",
        "department": "Unknown",
        "contacts": {
            "person": "General Support",
            "email": "support@example.com",
            "phone": "800-xxx-1217",
            "operations_support_line": "800-xxx-1217",
            "address": "N/A",
            "website": "N/A",
            "documents": []
        }
    }
else:
    # retrieved_docs is list of (Document, score)
    extracted_details = retrieved_docs[0][0].metadata

# Step 4: Format response
bullet_points = "\n".join([f"- {line.strip()}" for line in initial_response.split(". ") if line.strip()])
if initial_response == "I don’t know.":
    bullet_points = "- No relevant information found in the knowledge base."
document_links = ", ".join([
    f"[{doc}](https://exmple.com/docs/{doc})" for doc in extracted_details["contacts"].get("documents", [])
]) or "None"
response_template = f"""
Date: {datetime.now().strftime('%Y-%m-%d')}

---

**Topic**: {extracted_details['main_section']} → {extracted_details['subsection']}

**Answer**:
{bullet_points}

**Support Line & Manager Contact Details**:
- Support Line: {extracted_details['contacts']['operations_support_line']}
- Department: {extracted_details['department']}
- Team Manager: {extracted_details['contacts']['person']} ({extracted_details['contacts']['email']})
- Manager Phone: {extracted_details['contacts']['phone']}
- Address: {extracted_details['contacts']['address']}
- Website: {extracted_details['contacts']['website']}
- Documents: {document_links}

If these contacts are unavailable, please reach out to the General Support Line at 800-555-1212.

**Follow-up Questions**:
- [What are the detailed steps for this procedure?](#)
- [Who else can I contact in {extracted_details['department']}?](#)
- [Where can I find more resources on {extracted_details['subsection']}?](#)
"""
formatted_output = response_template
print(formatted_output)