import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool

async def build_company_documentation_tool(llm):
    db = chromadb.PersistentClient(path="./LlamaIndex/db")
    collection = db.get_or_create_collection("documentation")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    documents = SimpleDirectoryReader(input_dir="./LlamaIndex/data/documentation").load_data()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            embed_model,
        ],
        vector_store=vector_store,
    )

    await pipeline.arun(documents=documents)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    query_engine = index.as_query_engine(llm=llm)

    return QueryEngineTool.from_defaults(
        query_engine,
        name="company_documentation_expert",
        description="Expert in find internal company documentation and answer any doubt about it"
    )