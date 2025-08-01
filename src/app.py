import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, f1_score

import streamlit as st
import utils as utils
import models as models
from config import AppConfig, seed_everything

# If translator is available, use it for translation
try:
    from deep_translator import GoogleTranslator

    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

seed_everything()
if "init_done" not in st.session_state:
    st.session_state.init_done = False


@st.cache_resource
def create_model_faiss():
    MODEL_NAME = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device, tokenizer


@st.cache_data
def setup_data():
    df = utils.load_data(
        AppConfig.Path.RAW_DATA_FILE,
        columns=["Category", "Message"],
        drop_duplicates=True,
        dropna=True,
    )
    messages_raw = df["Message"].values.tolist()
    messages = [utils.preprocess_text(message) for message in messages_raw]
    labels = df["Category"].values.tolist()
    encoder, y = utils.encode_labels(df, label_column="Category")
    return df, messages_raw, messages, labels, y, encoder


def create_embedding_metadata(messages, labels, model, tokenizer, device, y):
    x_embeddings = models.get_embeddings(messages, model, tokenizer, device)
    metadata = [
        {"index": i, "message": message, "label": label, "label_encoded": y[i]}
        for i, (message, label) in enumerate(zip(messages, labels))
    ]
    return x_embeddings, metadata


# UI
def main():
    """Hàm chính điều khiển toàn bộ ứng dụng Streamlit."""
    st.set_page_config(page_title="Spam Classifier", page_icon="📊", layout="wide")
    st.title("Spam Messages Classifier")

    # --- Tải và xử lý dữ liệu một lần duy nhất ---
    df, messages_raw, messages, labels, y, le = setup_data()

    if not st.session_state.init_done:
        st.warning("⏳ Đang tải models...")
        create_model_faiss()
        st.session_state.init_done = True
        st.success("✅ Models đã sẵn sàng!")
        st.rerun()

    # Định nghĩa các tùy chọn chung
    model_options = models.StatisticalModelOptions.get_all_models()
    aug_options = ["No Augmentation", "SMOTE", "ADASYN"]
    vector_options = [
        utils.VectorizerOptions.TFIDF,
        utils.VectorizerOptions.BAG_OF_WORDS,
    ]

    # --- Giao diện Tabs ---
    tabs = st.tabs(["📋 Data Overview", "⚖️ Model Comparison", "🎯 FAISS", "💡 Live Prediction"])

    # --- TAB 1: DATA OVERVIEW ---
    with tabs[0]:
        st.header("📋 Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))

        col2.metric("HAM Messages", (df["Category"] == "ham").sum(), f"{(df["Category"] == "ham").sum() / len(df) * 100:.2f}%")
        col3.metric("SPAM Messages", (df["Category"] == "spam").sum(), f"{(df["Category"] == "spam").sum() / len(df) * 100:.2f}%")

        st.dataframe(df.head(10))

        # --- THÊM MỚI: Biểu đồ phân tích độ dài tin nhắn ---
        st.markdown("### Message Length Analysis")
        # Tính toán độ dài tin nhắn
        df["Message_Length"] = df["Message"].apply(len)

        # Tạo biểu đồ histogram
        fig_len, ax_len = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=df,
            x="Message_Length",
            hue="Category",
            kde=True,
            bins=60,
            ax=ax_len,
            palette={"ham": "skyblue", "spam": "salmon"},
        )
        ax_len.set_title("Distribution of Message Length by Category", fontsize=16)
        ax_len.set_xlabel("Message Length")
        ax_len.set_ylabel("Frequency")
        st.pyplot(fig_len, use_container_width=False)
        st.markdown(
            """
        * Biểu đồ trên cho thấy phân phối độ dài của các tin nhắn.
        * Tin nhắn **HAM** (thông thường) có xu hướng ngắn hơn.
        * Tin nhắn **SPAM** thường có độ dài lớn hơn, có thể do chứa nhiều thông tin quảng cáo, mời chào.
        """
        )

        st.markdown("### 📊 WordCloud")
        st.markdown(
            """
        (Đếm các từ đã được tiền xử lý)
        """
        )
        col4, col5 = st.columns(2)
        ham_words = " ".join([" ".join(messages[i]) for i in np.where(y == 0)[0]])
        spam_words = " ".join([" ".join(messages[i]) for i in np.where(y == 1)[0]])
        with col4:
            st.subheader("HAM Word Cloud")
            wc_ham = WordCloud(
                width=800, height=400, background_color="white", colormap="viridis"
            ).generate(ham_words)
            st.image(wc_ham.to_array(), use_container_width=True)

        with col5:
            st.subheader("SPAM Word Cloud")
            wc_spam = WordCloud(
                width=800, height=400, background_color="black", colormap="plasma"
            ).generate(spam_words)
            st.image(wc_spam.to_array(), use_container_width=True)

    # --- TAB 2: MODEL COMPARISON ---
    with tabs[1]:
        st.header("🎯 Compare Model")

        # Settings section for model comparison
        st.markdown("### Data Settings")
        selected_aug = st.selectbox("Data Augmentation", aug_options, key="comp_aug")
        selected_vector = st.selectbox(
            "Label Vectorizer", vector_options, key="comp_vec"
        )

        st.markdown("### Model Settings")
        st.markdown("""
        <style>
            .stMultiSelect [data-baseweb="select"] span{
                max-width: none !important;
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
        </style>
        """, unsafe_allow_html=True)
        # List all model options in checkbox for each options, default to all
        selected_models = st.multiselect(
            "### Select Models", model_options, default=model_options
        )
        if not selected_models:
            st.warning("Please select at least one model to compare.")
            return

        if st.button("🚀 Train & Compare Models", use_container_width=True):
            _, x = utils.vectorize_tokenized_text(messages, selected_vector)
            x_train, x_test, y_train, y_test = utils.create_train_test_data(
                x, y, selected_aug
            )
            cols = st.columns(len(selected_models))
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{model_name}**")
                        model = models.train_model(model_name, x_train, y_train)
                        y_pred = model.predict(x_test)

                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                        st.metric(
                            "F1-Score",
                            f"{f1_score(y_test, y_pred, average='weighted'):.3f}",
                        )

                        st.pyplot(
                            utils.plot_confusion_matrix(y_test, y_pred, le.classes_),
                            use_container_width=False,
                        )

    # --- TAB: FAISS ---
    with tabs[2]:
        st.header("🎯 FAISS")
        st.markdown("### KNN Setting")
        selected_k_values = st.multiselect(
            "### Select K Values", [1, 3, 5], default=[1, 3, 5]
        )


        if st.button("🚀 RUN", type="primary", use_container_width=True, key ="faiss_run"):
            # Setup embedding model
            model, device, tokenizer = create_model_faiss()

            # Create embeddings and metadata
            x_embeddings, metadata = create_embedding_metadata(
                messages_raw, labels, model, tokenizer, device, y
            )

            # Create FAISS index
            index, train_metadata, test_metadata, x_test_emb, y_test = (
                models.create_train_test_metadata(
                    messages_raw,
                    y,
                    x_embeddings,
                    metadata,
                    test_size=AppConfig.TEST_SIZE,
                    seed=AppConfig.SEED,
                )
            )

            #k_values = [1, 3, 5]
            y_pred = models.evaluate_knn_accuracy(
                x_test_emb, index, train_metadata, selected_k_values
            )
            cols_tab5 = st.columns(len(selected_k_values))
            for index, k in enumerate(selected_k_values):
                with cols_tab5[index]:
                    st.subheader(f"K Nearest Neighbors: {k}")

                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred[k]):.3f}")
                    st.metric(
                        "F1-Score",
                        f"{f1_score(y_test, y_pred[k], average='weighted'):.3f}",
                    )

                    st.pyplot(
                        utils.plot_confusion_matrix(y_test, y_pred[k], le.classes_),
                        use_container_width=False,
                    )

    # --- TAB: LIVE PREDICTION ---
    with tabs[3]:
        st.header("💡 Live Prediction")
        # Settings section for model comparison
        st.markdown("### Data Settings")
        pred_aug = st.selectbox("Data Augmentation", aug_options, key="pred_aug")
        pred_vector = st.selectbox("Label Vectorizer", vector_options, key="pred_vec")

        st.markdown("### Model Settings")
        pred_model = st.selectbox("Model", model_options, key="pred_model")

        st.markdown("### Your Message")
        message_input = st.text_area(
            "Enter a message to classify:",
            height=150,
            placeholder="e.g., Congratulations! You've won...",
        )

        if TRANSLATOR_AVAILABLE:
            translate_input = st.checkbox("Translate to EN before predicting")

        if st.button("🚀 RUN", type="primary", use_container_width=True, key="live_run"):
            if not message_input.strip():
                st.warning("Please enter a message.")
            else:
                with st.spinner("Processing..."):
                    final_input = message_input
                    if TRANSLATOR_AVAILABLE and translate_input:
                        final_input = GoogleTranslator(
                            source="auto", target="en"
                        ).translate(message_input)
                        st.info(f"Translated to: '{final_input}'")

                    processed_input = utils.preprocess_text(final_input)

                    # Train
                    vectorizer, x = utils.vectorize_tokenized_text(
                        messages, pred_vector
                    )
                    x_train, x_test, y_train, y_test = utils.create_train_test_data(
                        x, y, pred_aug
                    )
                    model_train = models.train_model(pred_model, x_train, y_train)

                    # Predict
                    processed_input_vector = vectorizer.transform(
                        [" ".join(processed_input)]
                    )
                    pred = model_train.predict(processed_input_vector)
                    pred_id = pred[0]
                    pred_proba = model_train.predict_proba(processed_input_vector)[0]
                    pred_confidence = pred_proba[pred_id]

                    prediction = le.inverse_transform([pred_id])[0]
                    if prediction == "spam":
                        st.error(
                            f"### Predicted: SPAM (Confidence: {pred_confidence:.2%})",
                            icon="🚫",
                        )
                    else:
                        st.success(
                            f"### Predicted: HAM (Confidence: {pred_confidence:.2%})",
                            icon="✔️",
                        )

    # with tabs[2]:
    #     st.header("🎯 Compare Augmentation")
    #     model_name = st.selectbox("Model Name", vector_options, key="4")
    #     vector_name = st.selectbox("Vectorize", vector_options, key="5")
    #     if st.button("🚀 Train Model", key=6):
    #         cols_tab2 = st.columns(len(augments))
    #         for index, aug_name in enumerate(augments):
    #             with cols_tab2[index]:
    #                 # vectorize = utils.create_vector(vector_name)
    #                 # features = vectorize.fit_transform(messages_raw)
    #                 _, features = utils.vectorize_tokenized_text(messages, vector_name)
    #                 xtrain, xtest, ytrain, ytest = utils.create_train_test_data(
    #                     features, y, aug_name
    #                 )
    #                 model_train = models.train_model(model_name, xtrain, ytrain)
    #                 pred = model_train.predict(xtest)
    #                 accuracy = accuracy_score(ytest, pred)
    #                 f1_scores = f1_score(ytest, pred)
    #                 cm = confusion_matrix(ytest, pred)
    #                 st.subheader(aug_name)
    #                 fig = plt.figure(figsize=(4, 4))
    #                 sns.heatmap(cm, linewidths=1, fmt="d", cmap="Greens", annot=True)
    #                 plt.ylabel("Actual label")
    #                 plt.xlabel("Predicted label")
    #                 title = f"Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}"
    #                 plt.title(title)
    #                 st.pyplot(fig)

    # with tabs[3]:
    #     st.header("🎯 Compare BAGs - TFIDF")
    #     model_name = st.selectbox("Model Name", vector_options, key="7")
    #     aug_name = st.selectbox("Augmentation", augments, key="8")
    #     if st.button("🚀 Train Model", key=9):
    #         cols_tab3 = st.columns(len(vector_options))
    #         for index, vector_name in enumerate(vector_options):
    #             with cols_tab3[index]:
    #                 # vectorize = utils.create_vector(vector_name)
    #                 # features = vectorize.fit_transform(messages)
    #                 _, features = utils.vectorize_tokenized_text(messages, vector_name)
    #                 xtrain, xtest, ytrain, ytest = utils.create_train_test_data(
    #                     features, y, aug_name
    #                 )
    #                 model_train = models.train_model(model_name, xtrain, ytrain)
    #                 pred = model_train.predict(xtest)
    #                 accuracy = accuracy_score(ytest, pred)
    #                 f1_scores = f1_score(ytest, pred)
    #                 cm = confusion_matrix(ytest, pred)
    #                 st.subheader(vector_name)
    #                 fig = plt.figure(figsize=(4, 4))
    #                 sns.heatmap(cm, linewidths=1, fmt="d", cmap="Greens", annot=True)
    #                 plt.ylabel("Actual label")
    #                 plt.xlabel("Predicted label")
    #                 title = f"Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}"
    #                 plt.title(title)
    #                 st.pyplot(fig)

    # with tabs[4]:
    #     st.header("🎯 Predict")
    #     testmsg = st.text_input("Message", "")
    #     vector_name = st.selectbox("Vectorize", vector_options, key="10")
    #     aug_name = st.selectbox("Augmentation", augments, key="11")
    #     model_name = st.selectbox("Model Name", vector_options, key="12")

    #     if st.button("🚀 Train Model", key=13):
    #         # Translate message if needed
    #         testmsg_translated = GoogleTranslator(source="auto", target="en").translate(
    #             testmsg
    #         )
    #         st.markdown(
    #             f'<h3 style="color:blue;">Message: {testmsg_translated}</h3>',
    #             unsafe_allow_html=True,
    #         )
    #         # Preprocess message
    #         testmsg_preprocessed = utils.preprocess_text(testmsg_translated)

    #         # Train
    #         vectorizer, features = utils.vectorize_tokenized_text(messages, vector_name)
    #         xtrain, xtest, ytrain, ytest = utils.create_train_test_data(
    #             features, y, aug_name
    #         )
    #         model_train = models.train_model(model_name, xtrain, ytrain)

    #         # Predict
    #         testmsg_vector = vectorizer.transform([" ".join(testmsg_preprocessed)])
    #         pred = model_train.predict(testmsg_vector)
    #         st.markdown(
    #             f'<h3 style="color:green;">Class Predicted: {le.inverse_transform(pred)[0]}</h3>',
    #             unsafe_allow_html=True,
    #         )

    # with tabs[5]:
    #     st.header("🎯 FAISS")

    #     if st.button("🚀 Run"):
    #         # Setup embedding model
    #         model, device, tokenizer = create_model_faiss()

    #         # Create embeddings and metadata
    #         X_embeddings, metadata = create_embedding_metadata(
    #             messages_raw, labels, model, tokenizer, device
    #         )

    #         # Create FAISS index
    #         index, train_metadata, test_metadata, X_test_emb, y_test = (
    #             models.create_train_test_metadata(
    #                 messages_raw,
    #                 y,
    #                 X_embeddings,
    #                 metadata,
    #                 test_size=AppConfig.TEST_SIZE,
    #                 seed=AppConfig.SEED,
    #             )
    #         )

    #         k_values = [1, 3, 5]
    #         predict_results = models.evaluate_knn_accuracy(
    #             X_test_emb, index, train_metadata, k_values
    #         )
    #         cols_tab5 = st.columns(len(k_values))
    #         for index, k in enumerate(k_values):
    #             with cols_tab5[index]:
    #                 st.subheader(f"K Nearest Neighbors: {k}")
    #                 accuracy = accuracy_score(y_test, predict_results[k])
    #                 f1_scores = f1_score(y_test, predict_results[k])
    #                 cm = confusion_matrix(y_test, predict_results[k])
    #                 fig = plt.figure(figsize=(4, 4))
    #                 sns.heatmap(cm, linewidths=1, fmt="d", cmap="Greens", annot=True)
    #                 plt.ylabel("Actual label")
    #                 plt.xlabel("Predicted label")
    #                 title = f"Accuracy Score: {accuracy:.3f}, F1 Score: {f1_scores:.3f}"
    #                 plt.title(title)
    #                 st.pyplot(fig)


if __name__ == "__main__":
    main()
