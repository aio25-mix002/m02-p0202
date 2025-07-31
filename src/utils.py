# utils.py

import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, ADASYN


@st.cache_data
def load_and_prep_data(path='data/spam.csv'):
    """
    Tải dữ liệu từ CSV, dọn dẹp và encode nhãn.
    Cache lại kết quả để tăng tốc độ tải lại ứng dụng.
    """
    try:
        # Đọc dữ liệu, chỉ lấy 2 cột đầu và đặt lại tên
        df = pd.read_csv(path, encoding='latin1', usecols=[0, 1])
        df.columns = ['Category', 'Message']
        df = df.drop_duplicates().dropna()

        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['Category'])
        st.session_state.le = le  # Lưu encoder để dùng lại ở nơi khác
        return df, le
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file tại '{path}'.")
        return None, None


def get_wordnet_pos(tag):
    """Chuyển đổi POS tag của NLTK sang định dạng WordNet."""
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


def lemmatize_text(text):
    """Thực hiện lemmatization trên văn bản đã được tách từ."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = []

    pos_tags = pos_tag(text.split())
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            lemmatized_tokens.append(lemma)
    return " ".join(lemmatized_tokens)


def preprocess_text(text):
    """Pipeline tiền xử lý văn bản hoàn chỉnh."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)  # Xóa ký tự đặc biệt
    text = lemmatize_text(text)
    return text

def create_model(name):
    """Tạo một instance của mô hình ML cổ điển dựa trên tên."""
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    return models.get(name)

def create_vector(name):
    """Tạo TfidfVectorizer hoặc CountVectorizer."""
    if name == 'TF-IDF':
        return TfidfVectorizer(stop_words='english', max_df=0.9, min_df=3, ngram_range=(1, 2))
    else:  # Bag of Words
        return CountVectorizer(stop_words='english', max_df=0.9, min_df=3, ngram_range=(1, 2))





def create_train_test_data(X, y, augment, test_size=0.1, random_state=42):
    """Chia dữ liệu train/test và tùy chọn tăng cường dữ liệu cho tập train."""
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if augment == 'SMOTE':
        sampler = SMOTE(random_state=random_state)
        xtrain, ytrain = sampler.fit_resample(xtrain, ytrain)
    elif augment == 'ADASYN':
        sampler = ADASYN(random_state=random_state)
        xtrain, ytrain = sampler.fit_resample(xtrain, ytrain)

    return xtrain, xtest, ytrain, ytest
def train_model(model_name, features_vector, labels_vector):
    """Huấn luyện và trả về model (giống main)."""
    model = create_model(model_name)
    return model.fit(features_vector, labels_vector)

def plot_confusion_matrix_seaborn(cm, labels):
    """
    Vẽ confusion matrix bằng Seaborn và Matplotlib.
    Hàm này trả về đối tượng figure để app.py có thể hiển thị bằng st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))  # Tăng chiều cao một chút cho đẹp hơn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12})  # Tăng kích thước font số
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    return fig