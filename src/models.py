# If you would like to run this file (models.py), specifically use the model "intfloat/multilingual-e5-base", please run this code from terminal: "huggingface-cli login"

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
import torch
import torch.nn.functional as F
import faiss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import config as config


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Training pipeline
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
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


def create_train_test_metadata(messages, y, x_embeddings, metadata, test_size, seed):
    train_indices, test_indices = train_test_split(
        range(len(messages)), test_size=test_size, stratify=y, random_state=seed
    )
    x_train_emb = x_embeddings[train_indices]
    x_test_emb = x_embeddings[test_indices]
    # y_train = y[train_indices]
    y_test = y[test_indices]
    train_metadata = [metadata[i] for i in train_indices]
    test_metadata = [metadata[i] for i in test_indices]
    index = faiss.IndexFlatIP(x_train_emb.shape[1])
    index.add(x_train_emb.astype("float32"))
    return index, train_metadata, test_metadata, x_test_emb, y_test


def classify_with_knn(query_text, model, tokenizer, device, index, train_metadata, k=1):
    query_with_prefix = f"query: {query_text}"
    batch_dict = tokenizer(
        [query_with_prefix], max_length=512, padding=True, truncation=True
    )
    batch_dict = {k: torch.tensor(v).to(device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(query_embedding, k)
    predictions = []
    neighbor_info = []
    for i in range(k):
        neighbor_idx = indices[0][i]
        neighbor_score = scores[0][i]
        neighbor_label = train_metadata[neighbor_idx]["label"]
        neighbor_message = train_metadata[neighbor_idx]["message"]
        predictions.append(neighbor_label)
        neighbor_info.append(
            {
                "score": float(neighbor_score),
                "label": neighbor_label,
                "message": (
                    neighbor_message[:100] + "..."
                    if len(neighbor_message) > 100
                    else neighbor_message
                ),
            }
        )
    unique_labels, counts = np.unique(predictions, return_counts=True)
    final_prediction = unique_labels[np.argmax(counts)]
    return final_prediction, neighbor_info


def evaluate_knn_accuracy(test_embeddings, index, train_metadata, k_values=[1, 3, 5]):
    predict_k = {}
    for k in k_values:
        total = len(test_embeddings)
        predict = []
        for i in range(total):
            query_embedding = test_embeddings[i : i + 1].astype("float32")
            scores, indices = index.search(query_embedding, k)
            predictions = []
            neighbor_details = []
            for j in range(k):
                neighbor_idx = indices[0][j]
                neighbor_label = train_metadata[neighbor_idx]["label_encoded"]
                neighbor_message = train_metadata[neighbor_idx]["message"]
                neighbor_score = float(scores[0][j])
                predictions.append(neighbor_label)
                neighbor_details.append(
                    {
                        "label": neighbor_label,
                        "message": neighbor_message,
                        "score": neighbor_score,
                    }
                )
            unique_labels, counts = np.unique(predictions, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            predict.append(predicted_label)
        predict_k[k] = predict
    return predict_k


def spam_classifier_pipeline(
    user_input, model, tokenizer, device, index, train_metadata, k=3
):

    prediction, neighbors = classify_with_knn(
        user_input, model, tokenizer, device, index, train_metadata, k=k
    )
    return {
        "prediction": prediction,
        "neighbors": [neighbor["label"] for neighbor in neighbors],
    }


def train_model(model_name, features_vector, labels_vector):
    model = _create_model(model_name)
    return model.fit(features_vector, labels_vector)


def _create_model(name):
    """Tạo một instance của mô hình ML cổ điển dựa trên tên."""
    random_state = config.AppConfig.SEED
    models = {
        StatisticalModelOptions.LOGISTIC_REGRESSION: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        StatisticalModelOptions.SUPPORT_VECTOR_MACHINE: SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        StatisticalModelOptions.GAUSSIAN_NAIVE_BAYES: GaussianNB(),
        StatisticalModelOptions.CATEGORICAL_NAIVE_BAYES: CategoricalNB(),
        StatisticalModelOptions.MULTINOMIAL_NAIVE_BAYES: MultinomialNB(),
        StatisticalModelOptions.RANDOM_FOREST: RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced"
        ),
        StatisticalModelOptions.XGBOOST: XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=random_state
        ),
    }
    return models.get(name)


class StatisticalModelOptions:
    LOGISTIC_REGRESSION = "Logistic Regression"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"
    GAUSSIAN_NAIVE_BAYES = "Gaussian Naive Bayes"
    CATEGORICAL_NAIVE_BAYES = "Categorical Naive Bayes"
    MULTINOMIAL_NAIVE_BAYES = "Multinomial Naive Bayes"
    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"

    @staticmethod
    def get_all_models():
        return [
            StatisticalModelOptions.LOGISTIC_REGRESSION,
            StatisticalModelOptions.SUPPORT_VECTOR_MACHINE,
            # StatisticalModelOptions.GAUSSIAN_NAIVE_BAYES,
            # StatisticalModelOptions.CATEGORICAL_NAIVE_BAYES,
            StatisticalModelOptions.MULTINOMIAL_NAIVE_BAYES,
            StatisticalModelOptions.RANDOM_FOREST,
            StatisticalModelOptions.XGBOOST,
        ]
