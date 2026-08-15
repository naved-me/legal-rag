import os
from dotenv import load_dotenv

# Load API keys from the .env file automatically
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Using Groq here for fast, free cloud inference
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

CHROMA_PATH = "chroma_db"

def query_rag(question: str, history: list = None):
    if history is None:
        history = []
        
    print("Loading database...")
    # 1. Load the exact same embedding model we used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to our existing Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 3. Create a Retriever
    # This turns our database into a search engine
    # search_kwargs={"k": 3} means "fetch the top 3 most relevant chunks"
    retriever = db.as_retriever(search_kwargs={"k": 3}) 
    
    # 4. Set up the Language Model (LLM)
    # WARNING: Groq API Key needed
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # 5. Create the Prompt Template
    system_prompt = (
        "You are a helpful legal assistant specializing in Motor Laws. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, just say that you don't know. "
        "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # 6. Build the RAG Chain using modern LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # First, a mini-chain that formats the docs and gets the answer
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Then, a main chain that runs the retriever and saves the sources
    rag_chain_with_source = RunnableParallel(
        {
            "context": itemgetter("input") | retriever,
            "input": itemgetter("input"),
            "history": itemgetter("history")
        }
    ).assign(answer=rag_chain_from_docs)
    
    # 7. Ask the Question!
    print(f"\nThinking about: '{question}'...")
    response = rag_chain_with_source.invoke({"input": question, "history": history})
    
    print("\n--- Answer ---")
    print(response["answer"])
    
    print("\n--- Sources Used ---")
    for doc in response["context"]:
        print(f"- From Page: {doc.metadata.get('page')}")
        
    history.extend([
        HumanMessage(content=question),
        AIMessage(content=response["answer"])
    ])
        
    return response

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Motor Legal Chatbot!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if user_input.strip() == "":
            continue
            
        query_rag(user_input, chat_history)
        print("\n" + "="*50 + "\n")
