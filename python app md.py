import os
import re
import sys
import csv
import warnings
from flask import Flask, request, render_template
from waitress import serve
import pandas as pd
import numpy as np
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Base directory for data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MASTER_DIR = os.path.join(BASE_DIR, 'MasterData')

# File paths
TRAIN_PATH = os.path.join(DATA_DIR, 'Training.csv')
TEST_PATH = os.path.join(DATA_DIR, 'Testing.csv')
SYMPTOM_SEV_PATH = os.path.join(MASTER_DIR, 'symptom_severity.csv')
SYMPTOM_DESC_PATH = os.path.join(MASTER_DIR, 'symptom_Description.csv')
SYMPTOM_PREC_PATH = os.path.join(MASTER_DIR, 'symptom_precaution.csv')

# Load and preprocess data
training = pd.read_csv(TRAIN_PATH)
testing = pd.read_csv(TEST_PATH)
cols = training.columns[:-1]
x_full = training[cols]
y_full = training['prognosis']

# Label encoding for diseases
le = preprocessing.LabelEncoder()
le.fit(y_full)
y_encoded = le.transform(y_full)

# Train a Decision Tree on full data for web diagnostics
clf_full = DecisionTreeClassifier().fit(x_full, y_encoded)

# Prepare models and evaluation for CLI mode
x = training[cols]
y = y_encoded
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf_cli = DecisionTreeClassifier().fit(x_train, y_train)
svm_cli = SVC().fit(x_train, y_train)
cv_scores = cross_val_score(clf_cli, x_test, y_test, cv=3)
svm_score = svm_cli.score(x_test, y_test)

# Symptom metadata containers
severity_dict = {}
description_dict = {}
precaution_dict = {}

# Load metadata

def get_severity_dict():
    with open(SYMPTOM_SEV_PATH) as f:
        reader = csv.reader(f)
        for row in reader:
            severity_dict[row[0]] = int(row[1])


def get_description_dict():
    with open(SYMPTOM_DESC_PATH) as f:
        reader = csv.reader(f)
        for row in reader:
            description_dict[row[0]] = row[1]


def get_precaution_dict():
    with open(SYMPTOM_PREC_PATH) as f:
        reader = csv.reader(f)
        for row in reader:
            precaution_dict[row[0]] = row[1:]

# CLI helper functions

def read_text(text: str):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def calc_condition(symptoms, days: int):
    total = sum(severity_dict.get(s, 0) for s in symptoms)
    if (total * days) / (len(symptoms) + 1) > 13:
        print("You should consult a doctor.")
    else:
        print("It might not be severe but take precautions.")


def sec_predict(symptoms_list):
    df = pd.read_csv(TEST_PATH)
    X = df.iloc[:, :-1]
    y = le.transform(df['prognosis'])
    model = DecisionTreeClassifier().fit(X, y)
    symptom_index = {symp: idx for idx, symp in enumerate(X.columns)}
    vec = np.zeros(len(symptom_index))
    for symp in symptoms_list:
        vec[symptom_index[symp]] = 1
    return model.predict([vec])


def print_disease(node_array):
    vals = node_array.nonzero()
    diseases = le.inverse_transform(vals[0])
    return [d.strip() for d in diseases]


def check_pattern(symptom_list, user_input: str):
    term = user_input.replace(' ', '_')
    pattern = re.compile(term)
    matches = [s for s in symptom_list if pattern.search(s)]
    return (bool(matches), matches)


def get_user_info():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    name = input("Your Name? -> ")
    print(f"Hello, {name}!")


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]
    symptom_list = list(feature_names)

    # User selects initial symptom
    while True:
        inp = input("Enter the symptom you are experiencing -> ")
        found, options = check_pattern(symptom_list, inp)
        if found:
            print("Matches found:")
            for i, opt in enumerate(options):
                print(f"{i}: {opt}")
            idx = int(input(f"Select (0-{len(options)-1}): "))
            chosen = options[idx]
            break
        else:
            print("No match, try again.")

    # Number of days
    days = 0
    while True:
        try:
            days = int(input("From how many days? -> "))
            break
        except ValueError:
            print("Enter a valid number.")

    # Traverse tree
    present = []
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            thresh = tree_.threshold[node]
            val = 1 if name == chosen else 0
            if val <= thresh:
                recurse(tree_.children_left[node], depth+1)
            else:
                present.append(name)
                recurse(tree_.children_right[node], depth+1)
        else:
            preds = print_disease(tree_.value[node])
            # Secondary prediction
            get_precaution_dict()
            sec_pred = sec_predict(present)
            calc_condition(present, days)
            if preds[0] == sec_pred[0]:
                print(f"You may have {preds[0]}")
                print(description_dict[preds[0]])
            else:
                print(f"You may have {preds[0]} or {sec_pred[0]}")
                print(description_dict[preds[0]])
                print(description_dict[sec_pred[0]])
            print("Take these precautions:")
            for i, prec in enumerate(precaution_dict[preds[0]]):
                print(f"{i+1}. {prec}")

    recurse(0, 1)
    print("""----------------------------------------END----------------------------------------""")

# Initialize metadata
get_severity_dict()
get_description_dict()
get_precaution_dict()

# Flask web app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", symptoms=list(cols))

@app.route("/diagnose", methods=["POST"])
def diagnose():
    selected = request.form.getlist("symptoms")
    input_vec = [0]*len(cols)
    for s in selected:
        if s in cols:
            input_vec[list(cols).index(s)] = 1
    pred = clf_full.predict([input_vec])[0]
    disease = le.inverse_transform([pred])[0]
    return render_template(
        "result.html",
        disease=disease,
        description=description_dict.get(disease, "No description available."),
        precautions=precaution_dict.get(disease, [])
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'cli':
        print(f"Decision Tree CV Score: {cv_scores.mean():.4f}")
        print(f"SVM Score: {svm_score:.4f}")
        get_user_info()
        tree_to_code(clf_cli, cols)
    else:
        serve(app, host="0.0.0.0", port=5000)
