import os
import io
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import requests
import PyPDF2
from database import vector_db
from bs4 import BeautifulSoup


# Load .env file
load_dotenv()
vector_db.clean()

# Init the OpenAI key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Init - embeddings, text splitter, and summarizer
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
summarizer = pipeline("summarization")

# Extract
def extract_and_store_text(url, is_pdf=False):
    try:
      
        if is_pdf:
            text = extract_text_from_pdf(url)
        else:
            text = extract_text_from_web(url)

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk)
            vector_db.add(embedding, {"content": chunk, "source": url})
        
        vector_db.save()
        return chunks
    except Exception as e:
        raise Exception(f"Error extracting and storing text: {str(e)}")

def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        return text
    except requests.RequestException as e:
        raise Exception(f"Error downloading PDF: {str(e)}")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
def extract_text_from_web(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()

        extracted_texts = [doc.page_content for doc in data]
        combined_text = ' '.join(extracted_texts)

        soup = BeautifulSoup(combined_text, 'html.parser')
        cleaned_text = soup.get_text(separator=' ')

        return cleaned_text
    except Exception as e:
        raise Exception(f"Error extracting text from web: {str(e)}")

# def get_local_file_path(pdf_path):
#     try:
#         if os.path.exists(pdf_path):
#             return os.path.abspath(pdf_path)
#         else:
#             raise FileNotFoundError(f"The file at {pdf_path} does not exist.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

def summarize_text(text, max_length=100):
    chunks = text_splitter.split_text(text)
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) > max_length:
            chunk = ' '.join(chunk.split()[:max_length])
        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

# response to question from Web UI
def question_answering(query):
    try:
        query_embedding = embeddings.embed_query(query)
        relevant_docs = vector_db.search(query_embedding)
        
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        # Summarize Split to save tokens 
        if len(context.split()) > 16000:
            context = summarize_text(context, max_length=100)
        
        if not context.strip():
            return "I don't have enough information to answer this question."
        
        prompt = f"""Answer the following question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question:
{query}

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error in question answering: {str(e)}")
    

#Query
def process_query_with_openai(query):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Process this search query for data extraction: {query}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message["content"]
    except Exception as e:
        raise Exception(f"Error processing query with OpenAI: {str(e)}")



