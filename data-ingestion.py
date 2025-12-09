import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Constants
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


def main() -> None:
    """
    Main function to build or check the Vector Database.
    It loads PDF and DOCX documents, splits them into chunks,
    embeds them using a HuggingFace model, and stores them in a FAISS vector database.
    """
    # Setting up directories
    current_dir_path: Path = Path(__file__).parent
    data_path: Path = current_dir_path / "data"
    persistent_directory: Path = current_dir_path / "data-ingestion-local"

    # Checking if the directory already exists
    if not persistent_directory.exists():
        print("[INFO] Initiating the build of Vector Database .. \U0001f50d\U0001f50d\n")

        # Checking if the folder that contains the required PDFs/DOCX files exists
        if not data_path.exists():
            raise FileNotFoundError(f"[ALERT] {data_path} doesn't exist. \u26a0\ufe0f\u26a0\ufe0f")

        # List of all the PDFs and DOCX files
        documents_to_process: list[Path] = [
            p for p in data_path.iterdir() if p.suffix == ".pdf" or p.suffix == ".docx"
        ]

        if not documents_to_process:
            print(f"[ALERT] No .pdf or .docx files found in {data_path}. \u26a0\ufe0f\u26a0\ufe0f")
            return

        doc_container: list[Document] = []  # Container for chunked documents

        # Taking each item from `documents_to_process` and loading it using the appropriate loader
        for doc_path in documents_to_process:
            loader = None
            if doc_path.suffix == ".pdf":
                loader = PyPDFLoader(
                    file_path=str(doc_path),  # PyPDFLoader expects a string path
                    extract_images=False,
                )
            elif doc_path.suffix == ".docx":
                loader = Docx2txtLoader(
                    file_path=str(doc_path)  # Docx2txtLoader expects a string path
                )
            else:
                print(f"[WARNING] Skipping unsupported file type: {doc_path.name} \u26a0\ufe0f")
                continue

            # Returns a list of `Document` objects. Each such object has - 1. Page Content // 2. Metadata
            docs_raw: list[Document] = loader.load()
            # Appending each `Document` object to the previously declared container (list)
            doc_container.extend(docs_raw)

        if not doc_container:
            print("[ALERT] No content was loaded from the documents. Vector DB not built. \u26a0\ufe0f")
            return

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
        print("[ALERT] Vector Database already exists. \u26a0\ufe0f\u26a0\ufe0f")


if __name__ == "__main__":
    main()
