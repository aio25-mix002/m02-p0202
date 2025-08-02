import string
import re
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# punkt is replaced by punkt_tab in NLTK 3.8.2
# https://github.com/nltk/nltk/issues/3293
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")


class VectorizerOptions:
    TFIDF = "TF-IDF"
    BAG_OF_WORDS = "Bag of Words"


# Embedding raw text by a pretrained model
def get_embedding_model(model_name):
    """
    Load a pretrained model from Hugging Face so that it can tokenize and
    vectorize raw text automatically.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = embed_model.to(device)
    embed_model.eval()
    return tokenizer, embed_model


def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    """
    Vectorzing raw text using a pretrained model.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(
            batch_texts_with_prefix, max_length=512, padding=True, truncation=True
        )
        batch_dict = {k: torch.tensor(v).to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def average_pool(last_hidden_states, attention_mask):
    """
    Convert a matrix of an embedded sentence into a vector.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Manual preprocessing
def preprocess_text(text, semantic_method="stemming", stemming_method="porter"):
    """
    Preprocess the raw text for the next step of manual vectorization.
    Another option is retaining the raw text for the pretrained embedding
    models on Hugging Face.
    
    Args:
        text: Input text string
        semantic_method: "stemming" or "lemmatization" 
        stemming_method: "porter" or "snowball" (only used if semantic_method="stemming")
    
    Returns:
        List of processed tokens
    """
    # Text Normalizing
    text = lowercase(text)

    # Cleaning
    text = punctuation_removal(text)

    # Tokenization
    tokens = tokenize(text)

    # Removing noise & Dimensionality reduction.
    tokens = remove_stopwords(tokens)

    # Semantic Normalization
    if semantic_method == "lemmatization":
        return lemmatization(tokens)
    else:  # default to stemming
        return stemming(tokens, method=stemming_method)


def lowercase(text):
    return text.lower()


def punctuation_removal(text):
    # translator = {k: '' for k in list(string.punctuation)}
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = stopwords.words("english")
    return [token for token in tokens if token not in stop_words]


def stemming(tokens, method="porter"):
    """
    Improved stemming with multiple algorithm options.
    
    Args:
        tokens: List of tokens to stem
        method: Stemming algorithm - "porter" (default) or "snowball"
    
    Returns:
        List of stemmed tokens
    """
    if method == "snowball":
        stemmer = SnowballStemmer("english")
    else:  # default to porter
        stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatization(tokens):
    """
    Lemmatize tokens with POS tagging for better accuracy.
    
    Args:
        tokens: List of tokens to lemmatize
    
    Returns:
        List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, get_simple_pos(pos)) for token, pos in pos_tags]


def extract_text_features(text):
    """
    Extract metadata features from text that can indicate spam.
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Basic text statistics
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Capital letter ratio (spam indicator)
    total_letters = sum(1 for c in text if c.isalpha())
    capital_letters = sum(1 for c in text if c.isupper())
    features['capital_ratio'] = capital_letters / total_letters if total_letters > 0 else 0
    
    # URL detection
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features['has_url'] = 1 if re.search(url_pattern, text, re.IGNORECASE) else 0
    features['url_count'] = len(re.findall(url_pattern, text, re.IGNORECASE))
    
    # Phone number detection (basic patterns)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    features['has_phone'] = 1 if re.search(phone_pattern, text) else 0
    features['phone_count'] = len(re.findall(phone_pattern, text))
    
    # Email detection
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    features['has_email'] = 1 if re.search(email_pattern, text) else 0
    features['email_count'] = len(re.findall(email_pattern, text))
    
    # Exclamation and question marks (spam indicators)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Currency symbols (spam indicator)
    currency_pattern = r'[$£€¥₹]'
    features['has_currency'] = 1 if re.search(currency_pattern, text) else 0
    features['currency_count'] = len(re.findall(currency_pattern, text))
    
    return features


def has_url(text):
    """Check if text contains URLs."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return bool(re.search(url_pattern, text, re.IGNORECASE))


def has_phone(text):
    """Check if text contains phone numbers."""
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return bool(re.search(phone_pattern, text))


def has_email(text):
    """Check if text contains email addresses."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return bool(re.search(email_pattern, text))


def create_dictionary(messages):
    """
    Manually vectorizing preprocessed text.
    """
    dictionary = []
    for tokens in messages:
        if tokens not in dictionary:
            dictionary.append(tokens)

    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features
    """
    Manually vectorizing preprocessed text.
    """
    dictionary = []
    for tokens in messages:
        if tokens not in dictionary:
            dictionary.append(tokens)

    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features


def load_data(filepath, columns, drop_duplicates=True, dropna=True):
    """
    Load data from a CSV file, clean it, and encode labels.
    """
    try:
        # Đọc dữ liệu, chỉ lấy 2 cột đầu và đặt lại tên
        # df = pd.read_csv(filepath, encoding='latin1', usecols=columns)
        df = pd.read_csv(filepath, usecols=columns)
        df.columns = columns

        if drop_duplicates:
            df = df.drop_duplicates()
        if dropna:
            df = df.dropna()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def encode_labels(df, label_column="Category"):
    """
    Encode labels in the DataFrame.
    """
    encoder = LabelEncoder()
    result = encoder.fit_transform(df[label_column])
    return encoder, result


def process_dataframe(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def get_simple_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    final_text = []
    for word in text.split():
        if word not in stop_words and len(word) > 2:
            pos = pos_tag([word])
            lema = lemmatizer.lemmatize(word, get_simple_pos(pos[0][1]))
            final_text.append(lema)
    return " ".join(final_text)


# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", '', text)
#     lema = lemmatize_words(text)
#     return lema


def create_vector(name):
    if name == "TFIDF":
        return TfidfVectorizer(max_df=0.9, min_df=2)
    else:
        return CountVectorizer(max_df=0.9, min_df=2)


def vectorize_tokenized_text(corpus, vector_name):
    """
    Vectorizing the preprocessed text.
    """
    if vector_name == VectorizerOptions.TFIDF:
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)

    elif vector_name == VectorizerOptions.BAG_OF_WORDS:
        vectorizer = CountVectorizer(max_df=0.9, min_df=2)
    else:
        raise ValueError(f"Unknown vectorization method: {vector_name}")
    # if corpus is a list of lists, convert it to a list of strings
    if isinstance(corpus[0], list):
        messages = [
            " ".join(msg) if isinstance(msg, list) else str(msg) for msg in corpus
        ]
    else:
        messages = [str(msg) for msg in corpus]
    # Fit and transform the messages
    vectorized_messages = vectorizer.fit_transform(messages)
    return vectorizer, vectorized_messages


def create_train_test_data(X, Y, augment):
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, random_state=42, test_size=0.3, stratify=Y
    )
    if augment == "SMOTE":
        sm = SMOTE(random_state=42)
        xtrain, ytrain = sm.fit_resample(xtrain, ytrain)
    elif augment == "ADASYN":
        ada = ADASYN(random_state=42)
        xtrain, ytrain = ada.fit_resample(xtrain, ytrain)
    return xtrain, xtest, ytrain, ytest



def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot a confusion matrix using Seaborn and Matplotlib.
    This function returns a figure object so that app.py can display it using st.pyplot().
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))  # Tăng chiều cao một chút cho đẹp hơn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12})  # Tăng kích thước font số
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    return fig
