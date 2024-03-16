from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from icecream import ic
from hashlib import md5
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("langchain-anki")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def generate_vector_id(text):
    return md5(text.encode('utf-8')).hexdigest()

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

documents_store = {}
def upload_embeddings(texts, embedding_vectors):
    upserts = []
    for text, embedding in zip(texts, embedding_vectors):
        doc_id = generate_vector_id(text.page_content[:50])
        upserts.append((doc_id, embedding))
        
        documents_store[doc_id] = text.page_content
    
    index.upsert(vectors=upserts)
    
def retrieve_document_content(doc_id):
    return documents_store.get(doc_id, "Document not found")

def get_answer_from_docs(query, top_k=5):
    query_embedding = embeddings.embed_documents([query])[0]
    results = index.query(vector=[query_embedding], top_k=top_k)
    matches = results['matches']
    
    document_contents = [retrieve_document_content(match['id']) for match in matches]
    documents = [Document(content) for content in document_contents]
    
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    input_dict = {
        'input_documents': documents,  
        'question': query,   
    }
    ic(input_dict)
    answer = chain.invoke(input_dict)
    return answer['output_text']

