# This python script parses 10-K PDFs, chunks the text, tags each chunk with company metadata,
# and outputs a vector index for semantic search and retrieval.

from llama_index.readers.file import UnstructuredReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Define directories
FILINGS_DIR = "filings"
PERSIST_DIR = "index_storage"

# Initialize UnstructuredReader
reader = UnstructuredReader()

# Load documents and loop through PDFs
documents = []
for filename in os.listdir(FILINGS_DIR):
    file_path = os.path.join(FILINGS_DIR, filename)
    if os.path.isfile(file_path):
        docs = reader.load_data(file_path)
        documents.extend(docs)

# Attach company metadata from filenames
for doc in documents:
    filename = doc.metadata.get("filename", "")
    company_prefix = filename.split("-")[0].lower()
    company_mapping = {
        "pypl": "PayPal",
        "sq": "Square",
        "tost": "Toast",
        "fi": "Fiserv"
    }
    company_name = company_mapping.get(company_prefix, company_prefix)
    
    doc.metadata["company"] = company_name

# Create text chunks and include metadata
node_parser = SentenceSplitter.from_defaults(
    chunk_size=512,
    chunk_overlap=50,
    include_metadata=True
)

# Parse documents into chunks (nodes)
nodes = node_parser.get_nodes_from_documents(documents)

# Create and save index
index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir=PERSIST_DIR)

print("Vector index with section and company metadata saved successfully.")