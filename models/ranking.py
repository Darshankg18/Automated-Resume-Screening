import pandas as pd

def get_match_scores(model,X):
    probabilities=model.predict_proba(X)
    match_scores=probabilities[:,1]*100
    return match_scores