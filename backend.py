from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import csv
import os
from waitress import serve

# Create the Flask app
app = Flask(__name__)

# Load data and train model
training = pd.read_csv(r'C:\Users\spide\OneDrive\Documents\AI MD\AI-Based-medical-diagnosis-main 2\Data\Training.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
clf = DecisionTreeClassifier().fit(x, y)

# Load symptom metadata
severity = {}
with open(r'C:\Users\spide\OneDrive\Documents\AI MD\AI-Based-medical-diagnosis-main 2\MasterData\symptom_severity.csv') as f:
    reader = csv.reader(f)
    severity = {rows[0]: int(rows[1]) for rows in reader}

description = {}
with open(r'C:\Users\spide\OneDrive\Documents\AI MD\AI-Based-medical-diagnosis-main 2\MasterData\symptom_Description.csv') as f:
    reader = csv.reader(f)
    description = {rows[0]: rows[1] for rows in reader}

precautions = {}
with open(r'C:\Users\spide\OneDrive\Documents\AI MD\AI-Based-medical-diagnosis-main 2\MasterData\symptom_precaution.csv') as f:
    reader = csv.reader(f)
    precautions = {rows[0]: rows[1:] for rows in reader}

@app.route("/")
def home():
    return render_template("index.html", symptoms=list(cols))

@app.route("/diagnose", methods=["POST"])
def diagnose():
    selected = request.form.getlist("symptoms")
    input_data = [0] * len(cols)
    for s in selected:
        if s in cols:
            input_data[cols.get_loc(s)] = 1
    prediction = clf.predict([input_data])[0]
    disease = le.inverse_transform([prediction])[0]
    return render_template("result.html", 
                           disease=disease, 
                           description=description.get(disease, "No description available."), 
                           precautions=precautions.get(disease, []))

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)