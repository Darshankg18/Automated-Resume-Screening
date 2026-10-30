import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

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

def train_logistic_regression(X_train, y_train,C=1.0):
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train,y_train):
    model=SVC(kernel='rbf',probability=True,random_state=42)
    model.fit(X_train,y_train)
    return model

def train_xgboost(X_train,Y_train):
    model=XGBClassifier(n_estimators=100,random_state=42,eval_metric='logloss')
    model.fit(X_train,Y_train)
    return model

def train_lightgbm(X_train,Y_train):
    model=LGBMClassifier(n_estimators=100,random_state=42,verbose=-1)
    model.fit(X_train,Y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-Score:", round(f1, 3))

    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    print("ROC-AUC:", round(roc_auc, 3))

    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    X, y = load_data("data/candidate_features/sample_candidate_features.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=== Logistic Regression ===")
    log_model = train_logistic_regression(X_train, y_train)
    evaluate_model(log_model, X_test, y_test)

    print()
    print("=== Random Forest ===")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    print()
    print("=== SVM ===")
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test)

    print()
    print("=== XGBoost ===")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)

    print()
    print("=== LightGBM ===")
    lgbm_model = train_lightgbm(X_train, y_train)
    evaluate_model(lgbm_model, X_test, y_test)
