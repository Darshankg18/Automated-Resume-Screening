import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['skill_match_score', 'experience_match_score', 'education_match_score', 'semantic_similarity']]
    y = df['shortlisted']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)
#     precision = precision_score(y_test, predictions)
#     recall = recall_score(y_test, predictions)
#     f1 = f1_score(y_test, predictions)

#     print("Accuracy:", round(accuracy, 3))
#     print("Precision:", round(precision, 3))
#     print("Recall:", round(recall, 3))
#     print("F1-Score:", round(f1, 3))

#    return accuracy, precision, recall, f1

