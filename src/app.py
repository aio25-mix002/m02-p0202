# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils import (
    load_and_prep_data,
    preprocess_text,
    create_vector,                # NEW NAME
    create_train_test_data,       # NEW NAME
    train_model,                  # NEW
    plot_confusion_matrix_seaborn
)
from models import retrain_and_predict_single

# Tùy chọn import thư viện dịch
try:
    from deep_translator import GoogleTranslator

    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


def main():
    """Hàm chính điều khiển toàn bộ ứng dụng Streamlit."""
    st.set_page_config(page_title="Spam Classifier", page_icon="📊", layout="wide")
    st.title("Spam Messages Classifier")

    # --- Tải và xử lý dữ liệu một lần duy nhất ---
    df, le = load_and_prep_data('data/spam.csv')
    if df is None:
        st.stop()

    @st.cache_data
    def get_processed_messages(messages_series):
        return [preprocess_text(msg) for msg in messages_series]

    messages = get_processed_messages(df['Message'])
    y = df['label_encoded'].values

    # --- Giao diện Tabs ---
    tabs = st.tabs(["📋 Data Overview", "⚖️ Model Comparison", "💡 Live Prediction"])

    # Định nghĩa các tùy chọn chung
    model_options = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'Naive Bayes', 'XGBoost']
    aug_options = ['No Augmentation', 'SMOTE', 'ADASYN']
    vector_options = ['TF-IDF', 'Bag of Words']

    # --- TAB 1: DATA OVERVIEW ---
    with tabs[0]:
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("HAM Messages", (df['Category'] == 'ham').sum())
        col3.metric("SPAM Messages", (df['Category'] == 'spam').sum())

        st.dataframe(df.head(10))

        # --- THÊM MỚI: Biểu đồ phân tích độ dài tin nhắn ---
        st.markdown("### Message Length Analysis")
        # Tính toán độ dài tin nhắn
        df['Message_Length'] = df['Message'].apply(len)

        # Tạo biểu đồ histogram
        fig_len, ax_len = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Message_Length', hue='Category', kde=True, bins=60, ax=ax_len,
                     palette={'ham': 'skyblue', 'spam': 'salmon'})
        ax_len.set_title('Distribution of Message Length by Category', fontsize=16)
        ax_len.set_xlabel('Message Length')
        ax_len.set_ylabel('Frequency')
        st.pyplot(fig_len)
        st.markdown("""
        * Biểu đồ trên cho thấy phân phối độ dài của các tin nhắn.
        * Tin nhắn **HAM** (thông thường) có xu hướng ngắn hơn.
        * Tin nhắn **SPAM** thường có độ dài lớn hơn, có thể do chứa nhiều thông tin quảng cáo, mời chào.
        """)
        # --- KẾT THÚC PHẦN THÊM MỚI ---

        st.markdown("### Word Clouds")
        col4, col5 = st.columns(2)
        ham_words = ' '.join(df[df['Category'] == 'ham']['Message'])
        spam_words = ' '.join(df[df['Category'] == 'spam']['Message'])

        with col4:
            st.subheader("HAM Word Cloud")
            wc_ham = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(ham_words)
            st.image(wc_ham.to_array(), use_container_width=True)

        with col5:
            st.subheader("SPAM Word Cloud")
            wc_spam = WordCloud(width=800, height=400, background_color='black', colormap='plasma').generate(spam_words)
            st.image(wc_spam.to_array(), use_container_width=True)

    # --- TAB 2: MODEL COMPARISON ---
    with tabs[1]:
        st.header("Model Performance Comparison")
        st.sidebar.header("Comparison Settings")

        selected_aug = st.sidebar.selectbox("Data Augmentation", aug_options, key="comp_aug")
        selected_vector = st.sidebar.selectbox("Vectorizer", vector_options, key="comp_vec")

        if st.sidebar.button("🚀 Train & Compare Models", use_container_width=True):
            vectorizer = create_vector(selected_vector)
            X = vectorizer.fit_transform(messages)
            X_train, X_test, y_train, y_test = create_train_test_data(X, y, selected_aug)

            cols = st.columns(len(model_options))
            for i, model_name in enumerate(model_options):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{model_name}**")
                        model = train_model(model_name, X_train, y_train)  # <-- dùng chuẩn main
                        y_pred = model.predict(X_test)

                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                        st.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

                        cm = confusion_matrix(y_test, y_pred)
                        st.pyplot(plot_confusion_matrix_seaborn(cm, le.classes_))

    # --- TAB 3: LIVE PREDICTION ---
    with tabs[2]:
        st.header("Predict New Message")
        st.sidebar.header("Prediction Settings")

        pred_model = st.sidebar.selectbox("Model", model_options, key="pred_model")
        pred_vector = st.sidebar.selectbox("Vectorizer", vector_options, key="pred_vec")
        pred_aug = st.sidebar.selectbox("Training Augmentation", aug_options, key="pred_aug")

        message_input = st.text_area("Enter a message to classify:", height=150,
                                     placeholder="e.g., Congratulations! You've won...")

        if TRANSLATOR_AVAILABLE:
            translate_input = st.checkbox("Translate VI -> EN before predicting")

        if st.button("Classify Message", type="primary", use_container_width=True):
            if not message_input.strip():
                st.warning("Please enter a message.")
            else:
                with st.spinner("Processing..."):
                    final_input = message_input
                    if TRANSLATOR_AVAILABLE and translate_input:
                        final_input = GoogleTranslator(source='auto', target='en').translate(message_input)
                        st.info(f"Translated to: '{final_input}'")

                    processed_input = preprocess_text(final_input)
                    vectorizer = create_vector(pred_vector)

                    pred_id, conf = retrain_and_predict_single(
                        pred_model, vectorizer, messages, y, pred_aug, processed_input
                    )
                    prediction = le.inverse_transform([pred_id])[0]

                    if prediction == 'spam':
                        st.error(f"### Predicted: SPAM (Confidence: {conf:.2%})", icon="🚫")
                    else:
                        st.success(f"### Predicted: HAM (Confidence: {conf:.2%})", icon="✔️")


if __name__ == "__main__":
    main()