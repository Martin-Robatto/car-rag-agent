import pandas as pd
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from download_dataset import get_dataset
from uuid import uuid4

load_dotenv()
db_location = "./chroma_langchain_db"
openai_api_key = os.getenv("OPENAI_API_KEY")
add_documents = not os.path.exists(db_location)

if add_documents:
    # 1. Load dataset
    df = get_dataset()

    # 2. Convert each row into a Document
    def row_to_doc(row):
        text = (
            f"{row['name']} - "
            f"{row['cylinders']} cylinders, {row['displacement']}cc, "
            f"{row['horsepower']} HP, {row['weight']} lbs, "
            f"{row['mpg']} MPG, acceleration: {row['acceleration']} sec, "
            f"model year: {row['model_year']}, origin: {row['origin']}"
        )
        metadata = row.to_dict()
        return Document(page_content=text, metadata=metadata)

    docs = [row_to_doc(row) for _, row in df.iterrows()]

    # 3. Set up OpenAI embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # # 4. Create Chroma vector store
    vector_store = Chroma(
        collection_name="car-rag-agent",
        embedding_function=embedding_model,
        persist_directory=db_location,  # Where to save data locally, remove if not necessary
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    print("✅ Embeddings stored in Chroma DB.")
else:
    print("✅ You already have the documents loaded to the Chroma DB.")
