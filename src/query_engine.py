import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_query_engine():
    # 1. Load vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="car-rag-agent",
        embedding_function=embedding_model,
        persist_directory="./chroma_langchain_db",
    )
    
    # 2. Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Return top 5 most relevant results
    )
    
    # 3. Set up LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # 4. Create a custom prompt template
    template = """You are a helpful automotive assistant. Answer the user's question based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    # 5. Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = create_query_engine()
    
    # Example query
    query = "What are some good cars with high MPG?"
    result = qa_chain({"query": query})
    
    print(f"Query: {query}")
    print(f"Answer: {result['result']}")
    print("\nSource documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.page_content}") 