from process_pdf import process_pdf, generate_embeddings
from pinecone_handler import upload_embeddings, get_answer_from_docs

if __name__ == "__main__":
    pdf_path = "plants.pdf"
    texts = process_pdf(pdf_path)
    embeddings = generate_embeddings(texts)
    upload_embeddings(texts, embeddings)

    query = "What is the largest plant in the world?"
    answer = get_answer_from_docs(query)
    print("Answer:", answer)