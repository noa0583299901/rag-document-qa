
import os
import httpx
from dotenv import load_dotenv

from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine


transport = httpx.HTTPTransport(verify=False)
client = httpx.Client(transport=transport)

load_dotenv()


DATA_DIR = "./data"
PERSIST_DIR = "./storage_ai"


Settings.llm = OpenAI(
    model="gpt-4o-mini", 
    temperature=0.1,
    http_client=client  
)

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    http_client=client  
)


Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def add_custom_metadata(file_path):
    return {"file_name": os.path.basename(file_path)}


if not os.path.exists(PERSIST_DIR):
    print("--- Creating new index from documents (this may take a moment)... ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found! Please create it and add PDFs.")
        exit()

    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        file_metadata=add_custom_metadata
    ).load_data()
    
  
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("--- Index created and saved successfully! ---")
else:
    print("--- Loading existing index from storage... ---")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3, 
    citation_chunk_size=512,
)


questions = [
    "What are the main ethical challenges of LLMs discussed in these documents?",
    "What practical applications in healthcare and education are mentioned?",
    "What techniques are used to improve LLM performance after the initial training?"
]

print("\n" + "="*50)
for q in questions:
    print(f"\nQuery: {q}")
    response = query_engine.query(q)
    print(f"\nResponse:\n{response}")
    

    print("\nSources used:")
    seen_sources = set()
    for node in response.source_nodes:
        fname = node.node.metadata.get('file_name', 'Unknown')
        page = node.node.metadata.get('page_label', 'Unknown')
        source_id = f"{fname} (Page {page})"
        if source_id not in seen_sources:
            print(f"- {source_id}")
            seen_sources.add(source_id)
    print("-" * 30)

print("\nTask completed successfully!")