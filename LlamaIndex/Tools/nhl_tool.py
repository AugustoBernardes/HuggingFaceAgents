import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool

async def build_nhl_tool(llm):
    # Initialize a persistent ChromaDB client to store and retrieve vector embeddings
    db = chromadb.PersistentClient(path="./LlamaIndex/db")

    # Get or create a collection dedicated to NHL-related data
    collection = db.get_or_create_collection("nhl")

    # Wrap the Chroma collection with LlamaIndex vector store interface
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Define the embedding model used to convert text into vector representations
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load raw documents from the local NHL data directory
    documents = SimpleDirectoryReader(input_dir="./LlamaIndex/data/nhl").load_data()

    # Create an ingestion pipeline to process and store documents into the vector database
    pipeline = IngestionPipeline(
        transformations=[
            # Split documents into smaller chunks for better retrieval performance
            SentenceSplitter(chunk_size=512, chunk_overlap=50),

            # Generate embeddings for each chunk
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Execute the ingestion pipeline asynchronously to populate the vector database
    await pipeline.arun(documents=documents)

    # Build a vector index from the stored embeddings (no reprocessing needed)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # Create a query engine that uses the provided LLM to answer questions using retrieved context
    query_engine = index.as_query_engine(llm=llm)

    # Wrap the query engine as a tool so it can be used by an agent
    return QueryEngineTool.from_defaults(
        query_engine,
        name="nhl_expert",
        description="Expert in NHL hockey, teams, standings, matches and playoffs"
    )