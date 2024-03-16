from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    
    return texts

def generate_embeddings(texts):
    contents = [text.page_content for text in texts]
    embeddings_batch = embeddings.embed_documents(contents)
    return embeddings_batch

