import joblib
from train_model import load_data, split_data, train_logistic_regression

if __name__ == "__main__":
    X, y = load_data("data/candidate_features/sample_candidate_features.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_logistic_regression(X_train, y_train, C=10)
    joblib.dump(model, "models/best_model.joblib")
    print("Model saved to models/best_model.joblib")