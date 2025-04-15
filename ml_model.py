import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import joblib

# Create folers for saving files
def setup_folders(csv_dir="csv", model_dir="model"):
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return csv_dir, model_dir

# Load image embeddings and return as a dictionary
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    embedding_columns = [str(i) for i in range(512)]
    embedding_dict = {
        row["filename"]: row[embedding_columns].values.astype(np.float32)
        for _, row in df.iterrows()
    }
    return df, embedding_dict

# Split outfit anchor images into train, validation, and test sets
def split_anchors(df_outfits, csv_dir):
    all_anchors = df_outfits["input_image"].apply(lambda x: x.split("\\")[-1]).unique()
    anchors_trainval, anchors_test = train_test_split(all_anchors, test_size=0.2, random_state=42)
    anchors_train, anchors_val = train_test_split(anchors_trainval, test_size=0.2, random_state=42)

    pd.DataFrame({"test_anchors": anchors_test}).to_csv(
        os.path.join(csv_dir, "test_anchor_images_1.csv"), index=False
    )

    return set(anchors_train), set(anchors_val), set(anchors_test)

# Build feature vector from an outfit row using embeddings
def build_vector_from_row(row, score, embedding_dict):
    anchor_img = row["input_image"].split("\\")[-1]
    if anchor_img not in embedding_dict:
        return None
    anchor_vec = embedding_dict[anchor_img]
    components = []
    for col in ["cluster_0", "cluster_1", "cluster_2"]:
        fname = row[col].split("\\")[-1]
        if fname == anchor_img:
            continue
        if fname in embedding_dict:
            components.append(embedding_dict[fname])
        else:
            return None
    if len(components) == 2:
        return np.concatenate([anchor_vec] + components), score
    return None

# Construct training dataset from outfit rows
def build_dataset(df_outfits, embedding_dict, usable_anchors):
    X, y = [], []
    for _, row in df_outfits.iterrows():
        anchor_img = row["input_image"].split("\\")[-1]
        if anchor_img not in usable_anchors:
            continue
        score = row["llm_score"] / 10.0
        result = build_vector_from_row(row, score, embedding_dict)
        if result:
            vec, target = result
            X.append(vec)
            y.append(target)
    return np.array(X), np.array(y)

# Train Random Forest model with cross-validation and save it
def train_and_evaluate(X, y, model_path):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    mse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)

    model.fit(X, y)
    joblib.dump(model, model_path)

    return np.mean(mse_scores)


def main():
    csv_dir, model_dir = setup_folders()
    df_embed, embedding_dict = load_embeddings(os.path.join(csv_dir, "image_clusters.csv"))
    df_outfits = pd.read_csv(os.path.join(csv_dir, "llava_outfit_scores.csv"))

    anchors_train, anchors_val, anchors_test = split_anchors(df_outfits, csv_dir)
    print(f"Anchors → Train: {len(anchors_train)}, Val: {len(anchors_val)}, Test: {len(anchors_test)}")

    usable_anchors = anchors_train | anchors_val
    X, y = build_dataset(df_outfits, embedding_dict, usable_anchors)
    print(f"Created {len(X)} regression samples.")

    mse = train_and_evaluate(X, y, os.path.join(model_dir, "rf_regressor_model.pkl"))
    print(f"Cross-Validation MSE: {mse:.4f}")
    print("Model saved as model/rf_regressor_model.pkl")

if __name__ == "__main__":
    main()