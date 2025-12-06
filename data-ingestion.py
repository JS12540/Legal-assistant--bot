import time
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

## langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document


def main():
    ## setting up directories
    # Using pathlib for modern path handling
    current_dir_path: Path = Path(__file__).parent
    data_path: Path = current_dir_path / "data"
    persistent_directory: Path = current_dir_path / "data-ingestion-local"

    ## checking if the directory already exists
    if not persistent_directory.exists():
        print("[INFO] Initiating the build of Vector Database .. 📌📌", end="\n\n")

        ## checking if the folder that contains the required PDFs exists
        if not data_path.exists():
            raise FileNotFoundError(
                f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
            )

        ## list of all the PDFs
        # Using pathlib's iterdir and suffix for listing PDFs
        pdfs: list[Path] = [p for p in data_path.iterdir() if p.suffix == ".pdf"]

        doc_container: list[Document] = [] ## <- list of chunked documents aka container
        
        ## taking each item from `pdfs` and loading it using PyPDFLoader
        for pdf_path in pdfs: # Renamed 'pdf' to 'pdf_path' for clarity
            loader = PyPDFLoader(file_path=str(pdf_path), # PyPDFLoader expects a string path
                                 extract_images=False)
            docsRaw: list[Document] = loader.load() ## <- returns a list of `Document` objects. Each such object has - 1. Page Content // 2. Metadata
            for doc in docsRaw:
                doc_container.append(doc) ## <- appending each `Document` object to the previously declared container (list)

        ## splitting the document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs_split: list[Document] = splitter.split_documents(documents=doc_container)

        ## displaying information about the split documents
        print("\n--- Document Chunks Information ---", end="\n")
        print(f"Number of document chunks: {len(docs_split)}", end="\n\n")

        ## embedding and vector store
        embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") ## <- open-source embedding model from HuggingFace - taking the default model only
        print("[INFO] Started embedding", end="\n")
        start = time.time() ## <- noting the starting time

        """
        Creating the embeddings for the documents and
        then storing them in a FAISS vector database
        """
        vectorDB = FAISS.from_documents(documents=docs_split,
                                       embedding=embedF)
        
        # Save the FAISS index locally
        vectorDB.save_local(str(persistent_directory)) # save_local expects a string path
        
        end = time.time() ## <- noting the end time
        print("[INFO] Finished embedding", end="\n")
        print(f"[ADD. INFO] Time taken: {end - start} seconds")

    else:
        print("[ALERT] Vector Database already exists. ⚠️⚠️")


if __name__ == "__main__":
    main()
