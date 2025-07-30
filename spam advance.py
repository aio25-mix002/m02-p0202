import re
import nltk
import string 
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from string import punctuation
import time
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
from deep_translator import GoogleTranslator
import streamlit as st

st.set_page_config(
    page_title="Data Analysis & ML Models",
    page_icon="📊",
    layout="wide"
)

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

st.markdown("""
    <style>
        .stButton>button{
            width:100%;
            background-color:#ff4b4b;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 1rem;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            background-color: #f0f2f6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
        }
        h1, h2, h3 { color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_spam_data():
    df= pd.read_csv('spam.csv')
    df= df.drop_duplicates()
    df = df.dropna()
    return df

def punctuation_removal(text):
    #translator = {k: '' for k in list(string.punctuation)}
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)
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

df = load_spam_data()
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()
messages = [preprocess_text(message) for message in messages]
le = LabelEncoder()
y = le.fit_transform(labels)

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
    
tabs = st.tabs(["📋 Data Overview", "Compare Model", "Compare Augmentation", "Compare BAGs - TFIDF", "Predict"])
models = ['Logistic Regression', 'Support Vector Machine', 'Random Forest']
augments = ['No Augmentation','SMOTE','ADASYN']
vectors = ['Bag of Words','TFIDF']
with tabs[0]:
    st.header("📋 Data Overview")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("Total Records",len(df))
    with col2:
        st.metric("Ham",len(np.where(y == 0)[0]))
    with col3:
        st.metric("Spam",len(np.where(y == 1)[0]))

    st.dataframe(df.head(10))
    st.markdown("### 📊 WordCloud")
    col4,col5 = st.columns(2)
    ham_words = ' '.join([messages[i] for i in np.where(y == 0)[0]])
    spam_words = ' '.join([messages[i] for i in np.where(y == 1)[0]])
    with col4:
        st.subheader("Ham")
        fig = plt.figure(figsize=(10, 6))
        word_cloud_ham = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(ham_words)
        plt.imshow(word_cloud_ham, interpolation='bilinear')
        st.pyplot(fig)
    with col5:
        st.subheader("Spam")
        fig = plt.figure(figsize=(10, 6))
        word_cloud_spam = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(spam_words)
        plt.imshow(word_cloud_spam, interpolation='bilinear')
        st.pyplot(fig)
with tabs[1]:
    st.header("🎯 Compare Model")
    aug_name = st.selectbox("Data Augmentation",augments,key="1")
    vector_name = st.selectbox("Vectorize",vectors,key="2")
    if st.button("🚀 Train Model",key=3):
        cols = st.columns(len(models))
        for index, model_name in enumerate(models):
            with cols[index]:
                vectorize = create_vector(vector_name)
                features = vectorize.fit_transform(messages)
                xtrain, xtest, ytrain, ytest= create_train_test_data(features,y,aug_name)
                model_train = train_model(model_name,xtrain,ytrain)
                pred = model_train.predict(xtest)
                accuracy = accuracy_score(ytest, pred)
                f1_scores = f1_score(ytest, pred)
                cm = confusion_matrix(ytest, pred)
                st.subheader(model_name)
                fig = plt.figure(figsize=(4,4))
                sns.heatmap(cm, linewidths=1, fmt='d', cmap='Greens', annot=True)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                title = f'Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}'
                plt.title(title)
                st.pyplot(fig)
with tabs[2]:
    st.header("🎯 Compare Augmentation")
    model_name = st.selectbox("Model Name",models,key="4")
    vector_name = st.selectbox("Vectorize",vectors,key="5")
    if st.button("🚀 Train Model",key=6):
        cols = st.columns(len(augments))
        for index, aug_name in enumerate(augments):
            with cols[index]:
                vectorize = create_vector(vector_name)
                features = vectorize.fit_transform(messages)
                xtrain, xtest, ytrain, ytest= create_train_test_data(features,y,aug_name)         
                model_train = train_model(model_name,xtrain,ytrain)
                pred = model_train.predict(xtest)
                accuracy = accuracy_score(ytest, pred)
                f1_scores = f1_score(ytest, pred)
                cm = confusion_matrix(ytest, pred)
                st.subheader(aug_name)
                fig = plt.figure(figsize=(4,4))
                sns.heatmap(cm, linewidths=1, fmt='d', cmap='Greens', annot=True)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                title = f'Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}'
                plt.title(title)
                st.pyplot(fig)

with tabs[3]:
    st.header("🎯 Compare BAGs - TFIDF")
    model_name = st.selectbox("Model Name",models,key="7")
    aug_name = st.selectbox("Augmentation",augments,key="8")
    if st.button("🚀 Train Model",key=9):
        cols = st.columns(len(vectors))
        for index, vector_name in enumerate(vectors):
            with cols[index]:
                vectorize = create_vector(vector_name)
                features = vectorize.fit_transform(messages)
                xtrain, xtest, ytrain, ytest= create_train_test_data(features,y,aug_name)
                model_train = train_model(model_name,xtrain,ytrain)
                pred = model_train.predict(xtest)
                accuracy = accuracy_score(ytest, pred)
                f1_scores = f1_score(ytest, pred)
                cm = confusion_matrix(ytest, pred)
                st.subheader(vector_name)
                fig = plt.figure(figsize=(4,4))
                sns.heatmap(cm, linewidths=1, fmt='d', cmap='Greens', annot=True)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                title = f'Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}'
                plt.title(title)
                st.pyplot(fig)

with tabs[4]:
    st.header("🎯 Predict")
    message_input = st.text_input("Message", "")
    translated = GoogleTranslator(source='auto', target='en').translate(message_input)
    message_pred = [preprocess_text(translated)]
    vector_name = st.selectbox("Vectorize",vectors,key="10")
    aug_name = st.selectbox("Augmentation",augments,key="11")
    model_name = st.selectbox("Model Name",models,key="12")
    if st.button("🚀 Train Model",key=13):
        st.markdown(f'<h3 style="color:blue;">Message: {translated}</h3>', unsafe_allow_html=True)
        vectorize = create_vector(vector_name)
        features = vectorize.fit_transform(messages)
        xtrain, xtest, ytrain, ytest= create_train_test_data(features,y,aug_name)
        model_train = train_model(model_name,xtrain,ytrain)
        messages_vector = vectorize.transform(message_pred)
        pred = model_train.predict(messages_vector)
        st.markdown(f'<h3 style="color:green;">Class Predicted: {le.inverse_transform(pred)[0]}</h3>', unsafe_allow_html=True)
