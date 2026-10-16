
import pandas as pd
from train_model import load_data,split_data,train_logistic_regression,evaluate_model

def get_match_scores(model,X):
    probabilities=model.predict_proba(X)
    match_scores=probabilities[:,1]*100 
    return match_scores

def add_ranks(results):
    results["rank"] = results["match_score"].rank(ascending=False, method="min").astype(int)
    return results.sort_values("rank")

def get_recommendation(score):
    if score >= 75:
        return "Strongly Recommended"
    elif score >= 50:
        return "Recommended"
    else:
        return "Not Recommended"

if __name__ == "__main__":
    X, y = load_data("data/candidate_features/sample_candidate_features.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_logistic_regression(X_train, y_train, C=10)
    evaluate_model(model, X_test, y_test)
    print()

    scores = get_match_scores(model, X_test)

    results = pd.DataFrame({
        "actual_shortlisted": y_test.values,
        "match_score": scores.round(1)
    })
    results = add_ranks(results)
    results["recommendation"] = results["match_score"].apply(get_recommendation)
    print(results)

