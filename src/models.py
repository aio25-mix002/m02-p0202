# models.py

from utils import create_model

from imblearn.over_sampling import SMOTE, ADASYN


def train_and_predict(model_name, X_train, y_train, X_test):
    """
    Huấn luyện một mô hình và trả về dự đoán cùng mô hình đã huấn luyện.
    """
    model = create_model(model_name)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction, model


def retrain_and_predict_single(model_name, vectorizer,
                               message_corpus, label_corpus,
                               augment_method, new_message):
    """
    Huấn luyện lại trên toàn bộ dữ liệu (tùy augment) rồi dự đoán tin nhắn mới.
    """
    # Vector hóa & augment
    X_full = vectorizer.fit_transform(message_corpus)
    y_full = label_corpus
    if augment_method == 'SMOTE':
        X_full, y_full = SMOTE(random_state=42).fit_resample(X_full, y_full)
    elif augment_method == 'ADASYN':
        X_full, y_full = ADASYN(random_state=42).fit_resample(X_full, y_full)

    # Huấn luyện
    model = create_model(model_name)
    model.fit(X_full, y_full)

    # Dự đoán
    X_input = vectorizer.transform([new_message])
    pred_id = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    confidence = pred_proba[pred_id]
    return pred_id, confidence