import pandas as pd
from train_model import load_data,split_data,train_logistic_regression

def get_match_scores(model,X):
    probabilities=model.predict_proba(X)
    match_scores=probabilities[:,1]*100
    return match_scores

if __name__ == "__main__":
    X, y = load_data("data/candidate_features/sample_candidate_features.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_logistic_regression(X_train, y_train)
    scores = get_match_scores(model, X_test)

    results = pd.DataFrame({
        "actual_shortlisted": y_test.values,
        "match_score": scores.round(1)
    })
    print(results)