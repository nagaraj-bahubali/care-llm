import os
import spacy
import requests
import time
import nltk
import torch
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from xml.etree import ElementTree as ET
from difflib import SequenceMatcher
nltk.download('wordnet')
from nltk.corpus import wordnet

# Load models once and reuse them
_spacy_model = None
_sentence_transformer = None

def load_spacy_model_once(model_name="en_ner_bc5cdr_md"):
    global _spacy_model
    try:
        if _spacy_model is None:
            _spacy_model = spacy.load(model_name)
        return _spacy_model
    except Exception as e:
        raise RuntimeError(f"Error loading the Spacy model '{model_name}': {e}")


def load_sentence_transformer(model_name='paraphrase-MiniLM-L6-v2'):
    global _sentence_transformer
    try:
        if _sentence_transformer is None:
            _sentence_transformer = SentenceTransformer(model_name)
        return _sentence_transformer
    except Exception as e:
        raise RuntimeError(f"Error loading the SentenceTransformer model '{model_name}': {e}")



def set_environment():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load Spacy model for entity extraction
def load_spacy_model(model_name="en_ner_bc5cdr_md"):
    try:
        nlp = spacy.load(model_name)  # Model for chemical/disease extraction
        return nlp
    except Exception as e:
        raise RuntimeError(f"Error loading the Spacy model: {e}")

# A helper function for text similarity comparison using Sentence-Transformers
def vectorize_texts_and_compute_similarity(text1, text2, model_name='paraphrase-MiniLM-L6-v2'):
    try:
        # Load a pre-trained model from Sentence-Transformers
        model = load_sentence_transformer(model_name)
        model.tokenizer.clean_up_tokenization_spaces = False
        
        # Encode both texts to get the embeddings (vectors)
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        # Convert to numpy and detach from GPU
        embedding1 = embedding1.cpu().detach().numpy()
        embedding2 = embedding2.cpu().detach().numpy()
        
        # Compute cosine similarity between the two embeddings
        similarity = cosine_similarity(embedding1.reshape(1, -1), 
                                        embedding2.reshape(1, -1))
        
        return similarity[0][0]
    
    except Exception as e:
        raise RuntimeError(f"Error computing similarity: {e}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Function to query PubMed and fetch articles using entities with retry logic
def search_pubmed(query, retries=3, backoff=2):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": "10"  # Adjust the number of results as needed
        }
        with requests.Session() as session:
            for attempt in range(retries):
                response = session.get(base_url, params=params)
                if response.status_code == 200:
                    return response.json().get('esearchresult', {}).get('idlist', [])
                else:
                    time.sleep(backoff ** attempt)  # Exponential backoff

        return []  # Return empty list if all retries fail
    except Exception as e:
        raise RuntimeError(f"Error querying PubMed: {e}")

# Function to fetch abstracts from PubMed using article IDs with retry logic
def fetch_abstracts(pubmed_ids, num_articles=5, retries=3, backoff=2):
    try:
        if not pubmed_ids or any(id is None for id in pubmed_ids):
            raise ValueError("The list of PubMed IDs contains invalid entries (None or empty).")

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(pubmed_ids), "retmode": "xml", "rettype": "abstract"}

        for attempt in range(retries):
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                try:
                    # Parse the XML response
                    root = ET.fromstring(response.content)
                    articles = []
                    for article in root.findall('.//PubmedArticle'):
                        # Extract title and abstract
                        title = article.find('.//ArticleTitle')
                        abstract_text = article.find('.//Abstract/AbstractText')

                        # Safely extract text, provide default if None
                        article_title = title.text.strip() if title is not None else "No title available"
                        article_abstract = abstract_text.text.strip() if abstract_text is not None else "No abstract available"

                        # Add article to the list
                        articles.append({
                            "title": article_title,
                            "abstract": article_abstract
                        })

                    # Return the first few articles
                    return articles[:num_articles]

                except Exception as parse_error:
                    print(f"Error parsing response: {parse_error}")
            else:
                print(f"Received non-200 status code: {response.status_code}")

            # Exponential backoff
            time.sleep(backoff ** attempt)

        # Return an empty list if all retries fail
        return []

    except Exception as e:
        raise RuntimeError(f"Error fetching PubMed abstracts: {e}")
    finally:
        gc.collect()
    
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def relaxed_entity_match(entity1, entity2):
    # Use partial string matching to account for synonyms and similar words
    sequence_match = SequenceMatcher(None, entity1, entity2).ratio()
    if sequence_match > 0.7:  
        return True
    synonyms = get_synonyms(entity1)
    if entity2 in synonyms:
        return True
    
    return False

# Contextual precision calculation with relaxed matching
def calculate_contextual_precision(original_text, simplified_text):
    nlp = load_spacy_model()

    # Extract entities from original and simplified texts
    original_doc = nlp(original_text)
    simplified_doc = nlp(simplified_text)
    
    original_entities = set((ent.text.lower(), ent.label_) for ent in original_doc.ents)
    simplified_entities = set((ent.text.lower(), ent.label_) for ent in simplified_doc.ents)

    print("original_entities : ", original_entities)
    print("simplified_entities : ", simplified_entities)

    matched_entities = 0
    total_entities = len(original_entities)

    for orig_entity, orig_label in original_entities:
        for simp_entity, simp_label in simplified_entities:
            if orig_label == simp_label and relaxed_entity_match(orig_entity, simp_entity):
                matched_entities += 1
                break  # Move to the next original entity after a match is found

    precision = matched_entities / total_entities if total_entities > 0 else 0.0
    return precision

# Main validation function
async def validate(simplified_text, original_text):
    try:
        nlp, doc, entities, query_terms, pubmed_ids, articles, merged_abstract = None, None, None, None, None, None, None
        # Load the Spacy model to extract entities
        nlp = load_spacy_model()

        # Extract entities from the original text
        doc = nlp(original_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Use extracted entities to create a PubMed query
        query_terms = [entity for entity, label in entities if label in ['CHEMICAL', 'DISEASE']]

        if not query_terms:
            return {
                "status_code": 200,
                "similarity": 0.0,
                "error_response": "No relevant entities found in the original text."
            }
        combined_query = " OR ".join(query_terms)

        pubmed_ids = search_pubmed(combined_query)

        if not pubmed_ids:
            return {
                "status_code": 200,
                "similarity": 0.0,
                "error_response": "No relevant articles found on PubMed for the query terms."
            }
        
        # Fetch top abstracts from PubMed
        articles = fetch_abstracts(pubmed_ids)
        top_5_articles = articles[:5]  # Get the top 5 articles
        merged_abstract = " ".join(article['abstract'] for article in top_5_articles).strip()

        # Compute similarity between the simplified text and the merged abstracts
        similarity = vectorize_texts_and_compute_similarity(simplified_text, merged_abstract)

        # Calculate contextual precision
        contextual_precision = calculate_contextual_precision(original_text, simplified_text)
        
        # Combine the two scores
        final_score = (similarity + contextual_precision) / 2
        return {
            "status_code": 200,
            "similarity": final_score,
            "error_response": ""
        }


    except Exception as e:
        return {
            "status_code": 500,
            "similarity": 0.0,
            "error_response": str(e)
        }
    
    finally:
        # Explicitly delete large objects and clear caches
        del nlp, doc, entities, query_terms, pubmed_ids, articles, merged_abstract
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

