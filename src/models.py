# If you would like to run this file (models.py), specifically use the model "intfloat/multilingual-e5-base", please run this code from terminal: "huggingface-cli login"

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch.nn.functional as F
import faiss
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import get_embedding_model, get_embeddings, average_pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "intfloat/multilingual-e5-base"

# Training pipeline
def training(messages, labels, model_name=MODEL_NAME, val_size=0.2, test_size=0.125, seed=42, k=1, device=DEVICE):
    '''
    Training pipeline that includes these steps:
    1. Getting embedded features and encoded label.
    2. Splitting train, validation, test sets.
    3. Training a classification model (KNN by default).
    4. Saving a checkpoint if applicable.
    5. Evaluating the model on test set.
    '''
    # Getting tokenizer and vectorizer
    tokenizer, embed_model = get_embedding_model(model_name=model_name)
    
    # Getting processed features and label. 
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_embeddings = get_embeddings(messages, model=embed_model, tokenizer=tokenizer, device=device)
    metadata = [{"index": i,"message": message, "label": label, "label_encoded":y[i]} 
                for i,(message, label) in enumerate(zip(messages,labels))]

    # Train-val-test split
    train_indices, val_indices = train_test_split(range(len(messages)), 
                                                  test_size = val_size, 
                                                  stratify = y, 
                                                  random_state = seed)
    train_indices, test_indices = train_test_split(train_indices, 
                                                   test_size = test_size, 
                                                   stratify = [y[i] for i in train_indices], #y
                                                   random_state = seed)
    X_train_emb = X_embeddings[train_indices]
    X_test_emb = X_embeddings[test_indices]
    X_val_emb = X_embeddings[val_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    y_val = y[val_indices]
    train_metadata = [metadata[i] for i in train_indices]
    test_metadata = [metadata[i] for i in test_indices]
    val_metadata = [metadata[i] for i in val_indices]
    
    embedding_dim = X_train_emb.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(X_train_emb.astype("float32"))
    
    # Training
    k_values = [1, 3, 5]
    print("Hyperparameter tuning on validation set...")
    accuracy_results, error_results, final_k = hyperparameter_tuning(X_val_emb,val_metadata, index, train_metadata, k_values)
    print("\n" + "="*50)
    print("ACCURACY RESULTS")
    print("="*50)
    for k, accuracy in accuracy_results.items():
        print(f"Top-{k} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*50)

    error_analysis = {
        "timestamp" : datetime.now().isoformat(),
        "model" : model_name,
        "val_size": len(X_val_emb),
        "accuracy_results": accuracy_results,
        "errors_by_k" : {}
    }
    for k, errors in error_results.items():
        error_analysis["errors_by_k"][f"k_{k}"] = {
            "total_errors": len(errors),
            "error_rate" : len(errors) / len(X_val_emb),
            "errors" : errors
        }
    output_file = "error_analysis.json"
    with open(output_file,"w", encoding="utf-8") as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    print(f"\n***Error analysis saved to: {output_file}***")
    print()
    print(f"***Summary:")
    for k, errors in error_results.items():
        print(f"k = {k}: {len(errors)} errors out of {len(X_val_emb)} samples")
       
    # Evaluation
    k_values = [final_k]
    print("\n#####")
    print("Evaluation on test set...")
    accuracy_results, error_results, _ = hyperparameter_tuning(X_test_emb, test_metadata, index, train_metadata, k_values)
    print("\n" + "="*50)
    print("ACCURACY RESULTS")
    print("="*50)
    for k, accuracy in accuracy_results.items():
        print(f"Top-{k} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*50)

    error_analysis = {
        "timestamp" : datetime.now().isoformat(),
        "model" : model_name,
        "val_size": len(X_val_emb),
        "accuracy_results": accuracy_results,
        "errors_by_k" : {}
    }
    for k, errors in error_results.items():
        error_analysis["errors_by_k"][f"k_{k}"] = {
            "total_errors": len(errors),
            "error_rate" : len(errors) / len(X_val_emb),
            "errors" : errors
        }
    output_file = "error_analysis.json"
    with open(output_file,"w", encoding="utf-8") as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    print(f"\n***Error analysis saved to: {output_file}***")
    print()
    print(f"***Summary:")
    for k, errors in error_results.items():
        print(f"k = {k}: {len(errors)} errors out of {len(X_val_emb)} samples")

 
def hyperparameter_tuning(val_embeddings, val_metadata, index, train_metadata, k_values):
    '''
    Tuning hyperparameters such as k. 
    Logging and visualizing accuracies if possible.
    Getting the hypeparameters that give the most accurate model.
    '''
    results = {}
    all_errors = {}
    final_k = k_values[0]
    for idx, k in enumerate(k_values):
        correct = 0
        total = len(val_embeddings)
        errors = []
        for i in tqdm(range(total),desc=f"Evaluating k={k}"):
            query_embedding = val_embeddings[i:i+1].astype("float32")
            true_label = val_metadata[i]["label"]
            true_message = val_metadata[i]["message"]
            scores, indices = index.search(query_embedding,k)
            predictions = []
            neighbor_details = []
            for j in range(k):
                neighbor_idx = indices[0][j]
                neighbor_label = train_metadata[neighbor_idx]["label"]
                neighbor_message = train_metadata[neighbor_idx]["message"]
                neighbor_score = float(scores[0][j])
                predictions.append(neighbor_label)
                neighbor_details.append({
                    "label": neighbor_label,
                    "message": neighbor_message,
                    "score":neighbor_score
                })
        unique_labels, counts = np.unique(predictions, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]

        if predicted_label == true_label:
            correct += 1
        else:
            error_info = {
                "index" : i,
                "original_index" : val_metadata[i]["index"],
                "message" : true_message,
                "true_label" : true_label,
                "predicted_label" : predicted_label,
                "neighbors" : neighbor_details,
                "label_distribution" : {label: int(count) for label, count in zip(unique_labels,counts)}
            }
            errors.append(error_info)
        accuracy = correct / total
        error_count = total - correct
        results[k] = accuracy
        all_errors[k] = errors
        print(f"Accuracy with k = {k}: {accuracy:.4f}")
        print(f"Number of errors with k = {k}: {error_count}/{total} ({(error_count/total)*100:.2f}%)")
        if accuracy > results[final_k]:
            final_k = k
    # print('results of hyper tune', results)
    return results, all_errors, final_k
    
def classify_with_knn(query_text,model,tokenizer,device,index,train_metadata,k=1):
    query_with_prefix = f"query: {query_text}"
    batch_dict = tokenizer([query_with_prefix],max_length=512,padding=True,truncation=True)
    batch_dict = {k:torch.tensor(v).to(device) for k,v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding= average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        query_embedding = F.normalize(query_embedding,p=2,dim=1)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(query_embedding,k)
    predictions= []
    neighbor_info = []
    for i in range(k):
        neighbor_idx = indices[0][i]
        neighbor_score = scores[0][i]
        neighbor_label = train_metadata[neighbor_idx]['label']
        neighbor_message = train_metadata[neighbor_idx]['message']
        predictions.append(neighbor_label)
        neighbor_info.append({
            "score": float(neighbor_score),
            "label": neighbor_label,
            "message":neighbor_message[:100] + "..." if len(neighbor_message) > 100 else neighbor_message
        })
    unique_labels, counts = np.unique(predictions, return_counts = True)
    final_prediction = unique_labels[np.argmax(counts)]
    return final_prediction, neighbor_info         
        
def spam_classifier_pipeline(user_input, model, tokenizer, index, train_metadata, device=DEVICE, k=3):
    print()
    print(f"***Classifying: '{user_input}'")
    print()
    print(f"***Using top-{k} nearest neighbors")
    print()

    prediction,neighbors = classify_with_knn(user_input, model, tokenizer, device, index, train_metadata, k=k)
    print(f"*** Prediction: {prediction.upper()}")
    print()

    print("***Top neighbors:")
    for i, neighbor in enumerate(neighbors,1):
        print(f"{i}. Label {neighbor['label']} | Score: {neighbor['score']:.4f}")
        print(f"Message: {neighbor['message']}")
        print()
    labels = [n["label"] for n in neighbors]
    label_counts = {label: labels.count(label) for label in set(labels)}
    return {
        "prediction": prediction,
        "neighbors" : neighbors,
        "label_distribution": label_counts
    }

if __name__ == '__main__':
    # training(messages, labels)
    DATASET_PATH = '../data/spam.csv'
    df = pd.read_csv(DATASET_PATH) ##[:500]
    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()
    training(messages, labels) 