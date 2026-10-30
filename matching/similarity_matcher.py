import numpy as np
import pandas as pd
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity

def euclidean_distance(vector_a, vector_b):
    distance = np.linalg.norm(vector_a - vector_b)
    return distance

def load_candidate_features(csv_path):
    df = pd.read_csv(csv_path)
    return df

def get_feature_vector(row):
    return np.array([
        row['skill_match_score'],
        row['experience_match_score'],
        row['education_match_score'],
        row['semantic_similarity']
    ])

