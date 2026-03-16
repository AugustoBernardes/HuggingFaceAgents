import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


async def main():

    # Load documents
    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()

    # Prepare storage
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Ingestion pipeline (Split chuncks)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )

    # RUN PIPELINE
    await pipeline.arun(documents=documents)

    # Embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # LLM
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct"
    )

    # Query engine
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    response = query_engine.query("Which teams fought in a war in New York?")
    print(response)


asyncio.run(main())