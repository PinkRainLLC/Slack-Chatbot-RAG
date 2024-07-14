# ~~~!!! Library Setup !!!~~~
import os
import re
import glob


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


from configparser import ConfigParser

import argparse

# ~~~!!! Config Setup !!!~~~

config = ConfigParser()
config.read("config.ini")

mindset_data_glob = config["Data"]["pdf_glob"]
pc_index_name = config["Pinecone"]["pc_index"]
model_name = config["Model"]["model_name"]


config_key = ConfigParser()
config_key.read("key_config.ini")

PINECONE_API_TOKEN = config_key["Pinecone"]["pinecone_api_key"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add document(s) to Pinecone DB")

    parser.add_argument("-d", "--document_glob", type=str, help="Path to the document/Glob to the documents")
    parser.add_argument("-i", "--create_index", type=bool, default=False, help="Only use this if this is your first time running this or you need to recreate the index")

    args = parser.parse_args()
    

    # ~~~!!! Environment Variable Setup !!!~~~

    os.environ["PINECONE_API_KEY"] = PINECONE_API_TOKEN

    if args.document_glob:
        mindset_data_glob = args.document_glob
    
    all_docs = glob.glob(mindset_data_glob)

    # ~~~!!! Doc Splitting ~~~!!!
    doc_chunks = []

    for doc in all_docs:
        loader = PyPDFLoader(doc)
        pages = loader.load()

        #For some reason the doc had every space as a new line?.... So doing this to get rid of them..
        for doc in pages:
            doc.page_content = re.sub("\n", " ", doc.page_content)


        # Splits docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        doc_chunks += text_splitter.split_documents(pages)


    lc_embeddings = HuggingFaceEmbeddings(model_name=model_name)


    # Will need this for the Pinecone setup
    hf_model = SentenceTransformer(model_name)
    embedding_size = hf_model.get_sentence_embedding_dimension()

    # ~~~!!! Pinecone DB Setup ~~~!!!

    # Really only used when initalizing a new table/DB for Pinecone. Shouldn't need to do again
    if args.create_index:
        pc = Pinecone(api_key=PINECONE_API_TOKEN)

        pc.create_index(
            name=pc_index_name,
            dimension=embedding_size, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )


    # ~~~!!! Pinecone Store Vectors !!!~~~

    vector_store = PineconeVectorStore(index_name=pc_index_name, embedding=lc_embeddings)

    vectorstore_from_docs = PineconeVectorStore.from_documents(
            doc_chunks,
            index_name=pc_index_name,
            embedding=lc_embeddings
        )