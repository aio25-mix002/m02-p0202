import re
import string
import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from nltk.corpus import stopwords, wordnet
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd

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

def process_dataframe(path):
    df= pd.read_csv(path)
    df= df.drop_duplicates()
    df = df.dropna()
    return df

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    final_text = []
    for word in text.split():
        if word not in stop_words and len(word) > 2:
            pos = pos_tag([word])
            lema = lemmatizer.lemmatize(word,get_simple_pos(pos[0][1]))
            final_text.append(lema)
    return " ".join(final_text)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    lema = lemmatize_words(text)
    return lema

    
def create_model(name):
    if name == 'Logistic Regression':
        model = LogisticRegression()
    elif name == 'Support Vector Machine':
        model = SVC(kernel='linear', C=1, probability=True)
    else:
        model = RandomForestClassifier(n_estimators=400, random_state=11)

    return model

def create_vector(name):
    if name == 'TFIDF':
        return TfidfVectorizer(max_df=0.9, min_df=2)
    else:
        return CountVectorizer(max_df=0.9, min_df=2)

def create_train_test_data(X,Y,augment):
    xtrain, xtest, ytrain, ytest = train_test_split(X,Y,random_state=42, test_size = 0.3, stratify = Y)
    if augment == 'SMOTE':
        sm = SMOTE(random_state = 42)
        xtrain, ytrain = sm.fit_resample(xtrain, ytrain)
    elif augment == 'ADASYN':
        ada = ADASYN(random_state = 42)
        xtrain, ytrain = ada.fit_resample(xtrain, ytrain)
    return xtrain, xtest, ytrain, ytest

def train_model(model_name,features_vector,labels_vector):
    model = create_model(model_name)
    return model.fit(features_vector,labels_vector)

# def create_features(tokens, dictionary):
    
# X = np.array([create_features(tokens,dictionary) for tokens in messages])
