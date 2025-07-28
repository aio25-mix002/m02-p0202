import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Train-test split
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = VAL_SIZE, shuffle = True, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = TEST_SIZE, shuffle = True, random_state= SEED)

# Training
model = GaussianNB()
print("Start training...")
model = model.fit(X_train,y_train)
print("Training completed!")

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
val_f1_scores = f1_score(y_val, y_val_pred)
test_f1_scores = f1_score(y_test, y_test_pred)
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
fig.suptitle('Naive Bayes')

sns.heatmap(val_cm,ax=axes[0], annot = True,linewidths=1, fmt='d', cmap='Blues')
val_title = f'Val Accuracy Score: {val_accuracy:.3f}, F1 Score: {val_f1_scores:.3f}'
axes[0].set_title(val_title)

sns.heatmap(test_cm,ax=axes[1], annot = True,linewidths=1, fmt='d', cmap='Greens')
test_title = f'Test Accuracy Score: {test_accuracy:.3f}, F1 Score: {test_f1_scores:.3f}'
axes[1].set_title(test_title)

plt.show()

# Predict and evaluation
def predict(text, model, dictionary, label_encoder):
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features= np.array(features).reshape(1,-1)
    prediction = model.predict(features)
    prediction_cls = label_encoder.inverse_transform(prediction)[0]
    return prediction_cls

