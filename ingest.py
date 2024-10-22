from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.node_parser import SimpleNodeParser

# Dataset Directory Path
DATASET = "dataset/"

# Faiss Index Path
FAISS_INDEX = "vectorstore/"

# Create Vector Store and Index
def embed_all():
    """
    Embed all files in the dataset directory
    """
    # Create the document loader using llamaindex
    loader = SimpleDirectoryReader(DATASET, recursive=True)
    documents = loader.load_data()

    # Parse documents into nodes using llamaindex's node parser
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)


    index = GPTVectorStoreIndex(nodes)


    index.storage_context.persist(persist_dir=FAISS_INDEX)

if __name__ == "__main__":
    embed_all()
