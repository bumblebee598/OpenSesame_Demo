from pinecone import Pinecone
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
from langchain_community.embeddings import SentenceTransformerEmbeddings

pinecone_api_key = os.environ["PINECONE_API_KEY"]

# Loading medical paper pdf files
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text
         
# Generate embeddings for Medical papers
def generate_embeddings(documents):
    embedding_model = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    embeddings = embedding_model.embed_documents(documents)
    return embeddings

# uploading papers to pinecone collection
def upload_papers_to_pinecone(directory_path, index) :
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            title = filename
            text = extract_text_from_pdf(file_path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 700,
                chunk_overlap = 80
            )
            documents = splitter.split_text(text)
            embeddings = generate_embeddings(documents)
            for i in range(len(documents)):
                text = documents[i]
                vector = embeddings[i]
                metadata = {"title" : title, "text" : text}
                index.upsert(
                    vectors = [{
                        "id":title+str(i),
                        "values":vector,
                        "metadata":metadata
                    }],
                    namespace="sample"
                )

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("demo-index")
upload_papers_to_pinecone("./data", index)
            









