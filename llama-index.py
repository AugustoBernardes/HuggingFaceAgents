import asyncio


from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool


import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from llama_index.tools.google import GmailToolSpec

# External community tool
tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

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
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
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
    query_engine = index.as_query_engine(llm=llm)
    # By a tool i can ask an agent and he decide if will use or not
    hockey_games_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="nhl_games_search",
        description="Use this tool to answer questions about NHL hockey games, teams, matches, and results"
    )
    
    # Testing NHl tool
    tool_response = hockey_games_tool.call("What are the current teams on playoffs ?")


    # Using query i call directly the engine RAG to check my data
    #response = query_engine.query("Which teams fought in a war in New York?")
    print(tool_response)


asyncio.run(main())