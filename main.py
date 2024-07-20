import os
import openai
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Set OpenAI API key
openai.api_key = 'sk-proj'

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        images_text = ""
        page_texts = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                page_texts.append((page_num + 1, page_text))
        # Extract text from images in the PDF
        images_text += extract_text_from_images_in_pdf(pdf_path, page_texts)
    return text, images_text, page_texts

def extract_text_from_images_in_pdf(pdf_path, page_texts):
    images = convert_from_path(pdf_path)
    images_text = []
    for i, image in enumerate(images):
        # Save image temporarily
        image_path = f"/tmp/page_{i}.png"
        image.save(image_path, "PNG")
        
        # Use Tesseract OCR to extract text from the image
        image_text = pytesseract.image_to_string(image_path)
        images_text.append(image_text)
        page_texts.append((i + 1, image_text))
        
        # Remove temporary image file
        os.remove(image_path)

    return " ".join(images_text)

def process_documents(documents_path):
    documents = []
    for file_name in os.listdir(documents_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(documents_path, file_name)
            text, images_text, page_texts = extract_text_from_pdf(pdf_path)
            documents.append({'file_name': file_name, 'text': text, 'images_text': images_text, 'page_texts': page_texts})
    return documents

def create_vector_store(documents):
    vectorizer = TfidfVectorizer()
    texts = []
    for doc in documents:
        combined_text = doc['text'] + " " + doc['images_text']
        if combined_text.strip():  # Check if the combined text is not empty
            texts.append(combined_text)
        else:
            print(f"Warning: Document '{doc['file_name']}' is empty or contains only stop words. Skipping.")
    
    if not texts:
        raise ValueError("No valid documents found with content.")

    vectors = vectorizer.fit_transform(texts).toarray()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    return index, vectorizer, documents

def query_vector_store(query, index, vectorizer, documents):
    query_vector = vectorizer.transform([query]).toarray()
    D, I = index.search(np.array(query_vector, dtype=np.float32), 1)
    return documents[I[0][0]]

def split_text_to_chunks(text, max_tokens=2048):
    words = text.split()
    chunks = []
    chunk = []
    total_tokens = 0

    for word in words:
        total_tokens += len(word.split())
        if total_tokens > max_tokens:
            chunks.append(' '.join(chunk))
            chunk = [word]
            total_tokens = len(word.split())
        else:
            chunk.append(word)
    
    if chunk:
        chunks.append(' '.join(chunk))
    
    return chunks

def get_answer_from_document(query, document):
    combined_text = f"{document['text']} {document['images_text']}"
    chunks = split_text_to_chunks(combined_text, max_tokens=2048)

    answers = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {query}\nDocument: {chunk}\nAnswer:"}
            ],
            max_tokens=150
        )
        answers.append(response.choices[0].message['content'].strip())

    return ' '.join(answers)

def find_page_number(query, page_texts):
    vectorizer = TfidfVectorizer()
    pages = [page_text for page_num, page_text in page_texts]
    vectors = vectorizer.fit_transform(pages).toarray()
    query_vector = vectorizer.transform([query]).toarray()
    
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    
    D, I = index.search(np.array(query_vector, dtype=np.float32), 1)
    return page_texts[I[0][0]][0]  # Return the page number

def save_to_excel(data, file_path):
    df = pd.DataFrame(data, columns=['Serial Number', 'Question', 'Answer', 'Source', 'Page Number'])
    df.to_excel(file_path, index=False)

if _name_ == "_main_":
    # Ensure documents directory exists
    documents_path = 'data/documents'
    if not os.path.exists(documents_path):
        os.makedirs(documents_path)

    # Process documents
    documents = process_documents(documents_path)

    # Create vector store
    try:
        index, vectorizer, documents = create_vector_store(documents)
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # List to store the Q&A data
    qa_data = []
    
    queries = []
    print("Enter your queries (type 'done' to finish):")
    while True:
        query = input("Query: ")
        if query.lower() == 'done':
            break
        queries.append(query)

    # Process each query
    for i, query in enumerate(queries):
        best_document = query_vector_store(query, index, vectorizer, documents)
        answer = get_answer_from_document(query, best_document)
        page_number = find_page_number(query, best_document['page_texts'])

        # Store the result
        qa_data.append([i + 1, query, answer, best_document['file_name'], page_number])

    # Save the Q&A data to an Excel file
    save_to_excel(qa_data, os.path.join(documents_path, 'query_answers.xlsx'))

