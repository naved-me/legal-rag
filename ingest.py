import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = "motor_laws_sample.pdf"
CHROMA_PATH = "chroma_db"

def ingest_data():
    #Check if file exists
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}. Please add it to the folder.")
        
    print("Loading PDF...")
    
    #Extract Text manually using pypdf
    reader = PdfReader(PDF_PATH)
    documents = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Wrap text in a LangChain Document and attach the page number
            doc = Document(page_content=text, metadata={"page": page_num + 1})
            documents.append(doc)
            
    print(f"Extracted {len(documents)} pages.")
    
    #Chunk the Text
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    #Create Embeddings and Save to Vector Database
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Saving to Chroma Database...")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    ingest_data()
