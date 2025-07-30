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
from models import spam_classifier_pipeline
from utils import *

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

st.set_page_config(
    page_title="Data Analysis & ML Models",
    page_icon="📊",
    layout="wide"
)



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
def create_feature_label(path):
    df = process_dataframe(path)
    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()
    messages = [preprocess_text(message) for message in messages]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return df ,messages, y, le

df,messages,y, le = create_feature_label('../data/spam.csv')


# UI 
def main():    
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
            cols_tab1 = st.columns(len(models))
            for index, model_name in enumerate(models):
                with cols_tab1[index]:
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
            cols_tab2 = st.columns(len(augments))
            for index, aug_name in enumerate(augments):
                with cols_tab2[index]:
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
            cols_tab3 = st.columns(len(vectors))
            for index, vector_name in enumerate(vectors):
                with cols_tab3[index]:
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

if __name__ == "__main__":
    main()