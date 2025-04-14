import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import joblib

# Setup folders 
csv_dir = "csv"
model_dir = "model"
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load embeddings 
df_embed = pd.read_csv(os.path.join(csv_dir, "image_clusters.csv"))
embedding_columns = [str(i) for i in range(512)]
embedding_dict = {
    row["filename"]: row[embedding_columns].values.astype(np.float32)
    for _, row in df_embed.iterrows()
}

# Load scored outfits 
df_outfits = pd.read_csv(os.path.join(csv_dir, "llava_outfit_scores.csv"))

# Split anchor filenames 
all_anchors = df_outfits["input_image"].apply(lambda x: x.split("\\")[-1]).unique()
anchors_trainval, anchors_test = train_test_split(all_anchors, test_size=0.2, random_state=42)
anchors_train, anchors_val = train_test_split(anchors_trainval, test_size=0.2, random_state=42)

print(f"Anchors → Train: {len(anchors_train)}, Val: {len(anchors_val)}, Test: {len(anchors_test)}")
pd.DataFrame({"test_anchors": anchors_test}).to_csv(os.path.join(csv_dir, "test_anchor_images.csv"), index=False)

# Helper to build feature vector 
def build_vector_from_row(row, score):
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

# Build regression dataset 
X, y = [], []
usable_anchors = set(anchors_train) | set(anchors_val)

for _, row in df_outfits.iterrows():
    anchor_img = row["input_image"].split("\\")[-1]
    if anchor_img not in usable_anchors:
        continue
    score = row["llm_score"] / 10.0
    result = build_vector_from_row(row, score)
    if result:
        vec, target = result
        X.append(vec)
        y.append(target)

X = np.array(X)
y = np.array(y)
print(f"Created {len(X)} regression samples.")

# Cross-validation training 
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

# Final model training 
model.fit(X, y)
joblib.dump(model, os.path.join(model_dir, "rf_regressor_model.pkl"))

# === Report result ===
print(f"Cross-Validation MSE: {np.mean(mse_scores):.4f}")
print("Model saved as model/rf_regressor_model.pkl")
