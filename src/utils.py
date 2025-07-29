import string
import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords,wordnet

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Embedding raw text by a pretrained model
def get_embedding_model(model_name):
    '''
    Load a pretrained model from Hugging Face so that it can tokenize and 
    vectorize raw text automatically.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = embed_model.to(device)
    embed_model.eval()
    return tokenizer, embed_model

def get_embeddings(texts, model, tokenizer, device,batch_size=32):
    '''
    Vectorzing raw text using a pretrained model.
    '''
    embeddings = []
    for i in tqdm(range(0,len(texts),batch_size), desc = "Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(batch_texts_with_prefix, max_length = 512, padding = True, truncation = True)
        batch_dict = {k: torch.tensor(v).to(device) for k,v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def average_pool(last_hidden_states, attention_mask):
    '''
    Convert a matrix of an embedded sentence into a vector.
    '''
    last_hidden = last_hidden_states.masked_fill(~attention_mask[...,None].bool(),0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[...,None]


# Manual preprocessing
def preprocess_text(text):
    '''
    Preprocess the raw text for the next step of manual vectorization. 
    Another option is retaining the raw text for the pretrained embedding 
    models on Hugging Face.
    '''
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    stem = stemming(tokens)
    return stem

def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    #translator = {k: '' for k in list(string.punctuation)}
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = stopwords.words('english')
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def create_dictionary(messages):
    '''
    Manually vectorizing preprocessed text.
    '''
    dictionary = []
    for tokens in messages:
        if tokens not in dictionary:
            dictionary.append(tokens)
            
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

def create_features(tokens, dictionary):
    
# X = np.array([create_features(tokens,dictionary) for tokens in messages])
