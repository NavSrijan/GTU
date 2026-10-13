from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.chains import RetrievalQA 

from llm import create_llm

from dotenv import load_dotenv
import os

load_dotenv()

def process_pdf(file_path: str = "./resume.pdf"): 
    loader = PyPDFLoader(file_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    ) 

    chunks = splitter.split_documents(document)

    return chunks

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2" 
)

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="resume_data_collection",
    persist_directory="./vector_db"
)

def ingest_data():
    chunks = process_pdf()
    vector_store.add_documents(chunks)

    return "Data ingestion completed." # Check if vector_db folder has been created or not. 

def rag_chain(query: str):
    llm = create_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return chain.invoke(query)



if __name__ == "__main__":
    # ----------Run this Immediately After creating process_pdf--------------------------
    # chunks = process_pdf()

    # for chunk in chunks:
    #     print(chunk)
    #     print("\n\n\n")

    # ----------Run this Immediately After creating ingest_data--------------------------
    # ingest_data() # First run will take time to download the model.

    # ----------Run this Immediately After creating rag_chain--------------------------
    query = input("Enter your query: ")
    response = rag_chain(query)
    print(response) # Ask the students to figure out how to get the clean response (the final response only).
