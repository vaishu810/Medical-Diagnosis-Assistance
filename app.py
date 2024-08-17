from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the training dataset
df = pd.read_csv('Training.csv')

# Features and target variable
X = df.iloc[:, :-1]  # Features (symptoms)
y = df['prognosis']   # Target variable

# Encode features
X_encoded = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Predict on testing data
test_data = pd.read_csv('Testing.csv')
test_data_encoded = pd.get_dummies(test_data)
test_predictions = model.predict(test_data_encoded)

# Calculate accuracy on testing data
true_labels = test_data['prognosis']
accuracy_test = accuracy_score(true_labels, test_predictions)
classification_rep_test = classification_report(true_labels, test_predictions)

# Exclude 'prognosis' from the symptom list
symptoms = df.columns[:-1].tolist()

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    # Exclude 'prognosis' from the symptoms
    selected_symptoms = {symptom: int(request.form[symptom] or 0) for symptom in symptoms if symptom != 'prognosis'}
    input_data = pd.get_dummies(pd.DataFrame(selected_symptoms, index=[0]))

    # Ensure that the input data columns match the feature names used during training
    input_data = input_data.reindex(columns=X_encoded.columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
