import boto3
import pandas as pd
import mwclient
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import json
import logging
import warnings
from botocore.config import Config
from tqdm import tqdm

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore').setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

# Configure boto3 client with timeout
config = Config(
    read_timeout=60,
    connect_timeout=60,
    retries={'max_attempts': 3}
)

# Initialize AWS Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    config=config,
    region_name='us-west-2',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

# Function to get Wikipedia article titles
def get_wikipedia_titles(category, max_articles=10):
    site = mwclient.Site('en.wikipedia.org')
    category_page = site.pages[category]
    titles = [cm.name for cm in category_page.members() if isinstance(cm, mwclient.page.Page)]
    return titles[:max_articles]

# Function to get Wikipedia article text and construct the full URL
def get_wikipedia_text(title):
    site = mwclient.Site('en.wikipedia.org')
    page = site.pages[title]
    base_url = "https://en.wikipedia.org/wiki/"
    fullurl = base_url + title.replace(" ", "_")
    return page.text(), fullurl

# Function to split text into chunks by sections and paragraphs
def split_text_into_chunks(text, max_tokens=500):
    sections = re.split(r'==\s.*\s==', text)
    chunks = []

    for section in sections:
        paragraphs = section.split('\n')
        chunk = ""
        for paragraph in paragraphs:
            if len(chunk.split()) + len(paragraph.split()) > max_tokens:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = paragraph
            else:
                chunk += "\n" + paragraph
        if chunk:
            chunks.append(chunk.strip())
    return chunks

# Function to clean text
def clean_text(text):
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = re.sub(r"{{.*?}}", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = text.strip()
    return text

# Function to generate embeddings using Amazon Titan
def generate_embeddings(text):
    model_id = "amazon.titan-embed-text-v1"
    body = json.dumps({"inputText": text})
    logger.info("Generating embeddings for text: %s", text[:100])  # Log the first 100 characters

    try:
        response = bedrock_client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        embedding = np.array(response_body['embedding'])
        logger.info("Successfully generated embeddings.")
        return embedding
    except Exception as e:
        logger.error("Error generating embeddings: %s", e)
        return None

# Collect documents
titles = get_wikipedia_titles("Category:2014 FIFA World Cup")
articles = [(title, *get_wikipedia_text(title)) for title in titles]

# Chunk and clean documents
chunks = []
for title, text, url in articles:
    cleaned_text = clean_text(text)
    chunked_texts = split_text_into_chunks(cleaned_text)
    for chunk in chunked_texts:
        chunks.append((title, url, chunk))

# Embed document chunks with progress indicator
df = pd.DataFrame(chunks, columns=['title', 'url', 'text'])
df['embedding'] = None

for i in tqdm(range(len(df)), desc="Processing chunks"):
    df.at[i, 'embedding'] = generate_embeddings(df.at[i, 'text'])

# Check for any rows with None embeddings and remove them
df = df[df['embedding'].notnull()]

# Save embeddings to CSV
df.to_csv("document_embeddings.csv", index=False)

# Function to search for relevant documents
def search(query, df):
    query_embedding = generate_embeddings(query)
    if query_embedding is None:
        return pd.DataFrame(columns=['title', 'url', 'text', 'similarity'])
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    results = df.sort_values(by='similarity', ascending=False)
    return results[['title', 'url', 'text', 'similarity']]

# Query
query = "2014 FIFA World Cup final match"
results = search(query, df)
print(results.head())
