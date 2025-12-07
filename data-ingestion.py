import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Constants
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


def main() -> None:
    """
    Main function to build or check the Vector Database.
    It loads PDF documents, splits them into chunks,
    embeds them using a HuggingFace model, and stores them in a FAISS vector database.
    """
    # Setting up directories
    current_dir_path: Path = Path(__file__).parent
    data_path: Path = current_dir_path / "data"
    persistent_directory: Path = current_dir_path / "data-ingestion-local"

    # Checking if the directory already exists
    if not persistent_directory.exists():
        print("[INFO] Initiating the build of Vector Database .. 📌📌\n")

        # Checking if the folder that contains the required PDFs exists
        if not data_path.exists():
            raise FileNotFoundError(f"[ALERT] {data_path} doesn't exist. ⚠️⚠️")

        # List of all the PDFs
        pdfs: list[Path] = [p for p in data_path.iterdir() if p.suffix == ".pdf"]

        doc_container: list[Document] = []  # Container for chunked documents

        # Taking each item from `pdfs` and loading it using PyPDFLoader
        for pdf_path in pdfs:
            loader = PyPDFLoader(
                file_path=str(pdf_path),  # PyPDFLoader expects a string path
                extract_images=False,
            )
            # Returns a list of `Document` objects. Each such object has - 1. Page Content // 2. Metadata
            docs_raw: list[Document] = loader.load()
            # Appending each `Document` object to the previously declared container (list)
            doc_container.extend(docs_raw)

        # Splitting the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs_split: list[Document] = splitter.split_documents(documents=doc_container)

        # Displaying information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs_split)}\n")

        # Embedding and vector store
        embed_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[INFO] Started embedding")
        start_time = time.time()  # Noting the starting time

        """
        Creating the embeddings for the documents and
        then storing them in a FAISS vector database
        """
        vector_db = FAISS.from_documents(documents=docs_split, embedding=embed_func)

        # Save the FAISS index locally
        vector_db.save_local(str(persistent_directory))

        end_time = time.time()  # Noting the end time
        print("[INFO] Finished embedding")
        print(f"[ADD. INFO] Time taken: {end_time - start_time:.2f} seconds")

    else:
        print("[ALERT] Vector Database already exists. ⚠️⚠️")


if __name__ == "__main__":
    main()
