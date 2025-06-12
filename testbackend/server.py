import os
import re
import json
import openai
import psycopg2
import time
import requests
import asyncio
import logging
from datetime import datetime
from flask import jsonify
from flask import Flask, request, Response, stream_with_context
from dotenv import load_dotenv
from flask_cors import CORS

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pyppeteer import launch

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import pinecone
from tqdm import tqdm
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
PORT = os.getenv("PORT")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "csv")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
CORS(app, supports_credentials=True, origins="*")

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
)
cursor = conn.cursor()
print("Connected to PostgreSQL database successfully!")

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def get_openai_response(prompt, message):
    """Helper function to get a response from OpenAI."""
    response = openai.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": f'{prompt}, Message {message}'}],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def parse_product_data(page_content):
    """Parse the product data from the page content."""
    product_data = {}
    current_key = None
    current_value = []

    for line in page_content.split('\n'):
        if ': ' in line:
            if current_key:
                # Join accumulated lines for the previous key as a single string
                product_data[current_key] = '\n'.join(current_value).strip()
            # Start new key-value pair
            current_key, value = line.split(': ', 1)
            current_value = [value.strip()]
        elif current_key:
            # Continue accumulating lines for the current key
            current_value.append(line.strip())

    # Add the last key-value pair if it exists
    if current_key:
        product_data[current_key] = '\n'.join(current_value).strip()

    return product_data

def get_response(index: str, message: str):
    print(f'## CLIENT ----------> {message}')
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    pc = Pinecone(api_key=PINECONE_KEY)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = PineconeVectorStore(
        pinecone_api_key=PINECONE_KEY, 
        index_name=index, 
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={ "k": 20}
    )

    docs = vectorstore.similarity_search_with_score(message, k=20)
    print("result:",docs)

    SYSTEM_TEMPLATE = """
        You are assistant from random website and this context data is scraped data from website: 
        <context>
        {context}
        </context>
        Using this scraping data, make relevant response for user's request. 
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(  
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """
                    Given the above conversation, generate a search query to look up in order to get information relevant 
                    to the conversation. Only respond with the query, nothing else.
                """
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    all_content = ""
    keyword_chunks = {}
        
    stream = conversational_retrieval_chain.stream(
        {
            "messages": [
                HumanMessage(content=message),
            ]
        },
    )

    print("Streaming Response: ", stream)

    for chunk in stream:
        for key in chunk:
            if key == "answer":
                all_content += chunk[key]
    yield f'data: {all_content}\n\n'

def extract_chunks(driver, min_length=100, max_length=500):
    """
    Extracts moderate-sized text chunks from top-level content tags.
    Returns a list of {'tag', 'summary', 'content'} dicts.
    """
    chunk_tags = ['article', 'main', 'section', 'div']
    chunks = []
    found = False

    for tag in chunk_tags:
        elements = driver.find_elements(By.TAG_NAME, tag)
        for el in elements:
            text = el.text.strip()
            if len(text) > min_length:
                found = True
                summary = text[:80].replace('\n', ' ') + '...' if len(text) > 80 else text
                if len(text) > max_length:
                    # If too long, split into paragraphs
                    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
                    for para in paragraphs:
                        if len(para) > min_length:
                            chunks.append({'tag': tag, 'summary': para[:80] + '...', 'content': para})
                else:
                    chunks.append({'tag': tag, 'summary': summary, 'content': text})

    if not found:
        # Fallback to body direct children
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
            children = body.find_elements(By.XPATH, './*')
            for child in children:
                text = child.text.strip()
                if len(text) > min_length:
                    tag = child.tag_name
                    summary = text[:80].replace('\n', ' ') + '...' if len(text) > 80 else text
                    if len(text) > max_length:
                        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
                        for para in paragraphs:
                            if len(para) > min_length:
                                chunks.append({'tag': tag, 'summary': para[:80] + '...', 'content': para})
                    else:
                        chunks.append({'tag': tag, 'summary': summary, 'content': text})
        except Exception as e:
            logging.warning(f'Could not extract from <body>: {e}')
    return chunks

def get_all_internal_links(driver, base_url):
    """Collect all unique internal links from the current page."""
    base_domain = urlparse(base_url).netloc
    links = set()
    anchor_tags = driver.find_elements(By.TAG_NAME, "a")
    for link in anchor_tags:
        href = link.get_attribute("href")
        if not href:
            continue
        parsed = urlparse(href)
        if not parsed.scheme or not parsed.netloc:
            href = urljoin(base_url, href)
            parsed = urlparse(href)
        if parsed.netloc == base_domain and href not in links:
            links.add(href)
    return list(links)

# def scrape_site_content(url, max_pages=30, delay=1):
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(options=chrome_options)

#     visited = set()
#     results = []
#     page_count = 0

#     logging.info(f"Starting scrape from: {url}")

#     # Step 1: Collect all possible links from the landing page
#     driver.get(url)
#     time.sleep(delay)
#     all_links = get_all_internal_links(driver, url)
#     all_links = [url] + [l for l in all_links if l != url]
#     unique_links = []
#     for l in all_links:
#         if l not in unique_links:
#             unique_links.append(l)
#         if len(unique_links) >= max_pages:
#             break

#     logging.info(f"Found {len(unique_links)} unique links to scrape.")

#     # Step 2: Scrape each link
#     for current_url in unique_links:
#         if current_url in visited:
#             logging.info(f"Already visited: {current_url} -- skipping.")
#             continue

#         try:
#             logging.info(f"Scraping page {page_count+1}/{max_pages}: {current_url}")
#             start_time = time.time()
#             driver.get(current_url)
#             time.sleep(delay)  # Wait for page load

#             # Extract moderate-sized chunks
#             chunks = extract_chunks(driver)

#             logging.info(f"Extracted {len(chunks)} chunks from {current_url}")

#             results.append({
#                 "url": current_url,
#                 "chunks": chunks
#             })
#             visited.add(current_url)
#             page_count += 1

#             elapsed = time.time() - start_time
#             logging.info(f"Scraped {current_url} in {elapsed:.2f} seconds. Total pages scraped: {page_count}")

#             if page_count >= max_pages:
#                 logging.info(f"Reached max pages limit: {max_pages}. Stopping.")
#                 break

#         except Exception as e:
#             logging.error(f"Error scraping {current_url}: {e}")

#     driver.quit()
#     logging.info(f"Scraping finished. Total pages scraped: {len(visited)}")
#     return results

# def chunk_text(text, chunk_size=1500, overlap=200):
#     """Splits text into chunks for embedding."""
#     tokens = text.split()
#     chunks = []
#     i = 0
#     while i < len(tokens):
#         chunk = ' '.join(tokens[i:i+chunk_size])
#         chunks.append(chunk)
#         i += chunk_size - overlap
#     return chunks

# def embed_texts_openai(texts, openai_api_key, model="text-embedding-3-small"):
#     openai.api_key = openai_api_key
#     embeddings = []
#     for i in tqdm(range(0, len(texts), 10)):
#         batch = texts[i:i+10]
#         resp = openai.embeddings.create(input=batch, model=model)
#         for emb in resp.data:
#             embeddings.append(emb.embedding)
#     return embeddings

def create_pinecone_index(api_key, env, index_name, dimension):
    pc = Pinecone(api_key=api_key)
    # List index names
    index_names = pc.list_indexes().names()
    if index_name not in index_names:
        # You may need to adjust cloud/region to match your Pinecone project
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',  # or 'gcp', etc.
                region=env    # e.g., 'us-east-1', 'us-west-2'
            )
        )
    return pc.Index(index_name)

def get_internal_urls(seed_url, max_pages=30):
    """Return a list of unique internal links from a page."""
    import requests
    from urllib.parse import urlparse, urljoin
    from bs4 import BeautifulSoup

    visited = set()
    to_visit = [seed_url]
    base_domain = urlparse(seed_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a['href'])
                if urlparse(href).netloc == base_domain and href not in visited and href not in to_visit:
                    to_visit.append(href)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        visited.add(url)
        if len(visited) >= max_pages:
            break
    return list(visited)

def ingest_website_to_pinecone(
    website_url,
    index_name,
    chatbot_name,
    embedding_model='text-embedding-3-small',
    chunk_size=1500,
    overlap=200,
    max_pages=30,
    max_chunks=2000,  # NEW: Hard cap, adjust as needed
    embedding_batch_size=100,
):
    print('Discovering URLs...')
    urls = get_internal_urls(website_url, max_pages=max_pages)
    print(f"Found {len(urls)} URLs to load.")

    print('Loading & parsing pages...')
    docs = []
    for url in tqdm(urls):
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={},
                bs_get_text_kwargs={"separator": " ", "strip": True},
            )
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {url}: {e}")

    print(f"Loaded {len(docs)} documents.")

    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")

    # Limit the number of chunks for embedding and upserting
    if len(split_docs) > max_chunks:
        print(f"Too many chunks ({len(split_docs)}). Truncating to {max_chunks}.")
        split_docs = split_docs[:max_chunks]

    print("Creating Pinecone index (if needed)...")
    dimension = 1536  # For OpenAI text-embedding-3-small/large
    create_pinecone_index(
        api_key=PINECONE_KEY,
        env=PINECONE_ENV,
        index_name=index_name,
        dimension=dimension
    )

    print("Embedding and upserting to Pinecone...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embedding_model)

    # Batching manually for memory and API safety
    from langchain.docstore.document import Document
    for i in tqdm(range(0, len(split_docs), embedding_batch_size)):
        batch_docs = split_docs[i:i+embedding_batch_size]
        try:
            # This will embed and upsert this batch
            PineconeVectorStore.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                index_name=index_name,
                namespace=PINECONE_NAMESPACE,
                pinecone_api_key=PINECONE_KEY,
            )
        except Exception as e:
            print(f"Error embedding/upserting batch {i//embedding_batch_size}: {e}")

    print(f'Ingestion complete. Pinecone index: {index_name}')

    created_at = datetime.now() 
    if chatbot_name is not None:
        cursor.execute(
            """
            INSERT INTO customchatbot (name, link, pinecone_index, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (chatbot_name, website_url, index_name, created_at)
        )
        conn.commit()
        print("Inserted chatbot record into customchatbot table.")

    return index_name

@app.route("/chat")
def sse_request():
    """Handle chat requests."""
    message = request.args.get('message', '')
    index = request.args.get('index', '')
    return Response(get_response(index, message))

@app.route("/chatbot_list", methods=["GET"])
def get_chatbot_list():
    try:
        cursor.execute("SELECT * FROM customchatbot")
        rows = cursor.fetchall()
        # Get column names
        colnames = [desc[0] for desc in cursor.description]
        # Convert rows to list of dicts
        chatbot_list = [dict(zip(colnames, row)) for row in rows]
        return jsonify({"success": True, "chatbots": chatbot_list}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# @app.route("/get_db_response")
# def db_request():
#     category = request.args.get('category', '')
#     type = request.args.get('type', '')
#     subcategory = request.args.get('subcategory', '')
    
#     result_data = get_db_response(category=category, type=type, subcategory=subcategory)

#     return jsonify(result_data)

@app.route("/create_chatbot", methods=["POST"])
def create_chatbot_request():
    data = request.get_json()
    website = data.get('website', '')
    index_name = data.get('index', '')
    chatbot_name = data.get('chatbot_name', '')

    if not website or not index_name or not chatbot_name:
        return jsonify({"error": "Missing 'website' or 'index' parameter"}), 400

    try:
        created_index = ingest_website_to_pinecone(
            website_url=website,
            index_name=index_name,
            chatbot_name=chatbot_name
            # You can add/override other params here if needed
        )
        return jsonify({"success": True, "index_name": created_index}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)