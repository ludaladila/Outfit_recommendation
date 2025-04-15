import pandas as pd
import os
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from collections import Counter

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def encode_image(img_path, model, preprocess, device):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return F.normalize(features, dim=1)

def load_dataset_and_features(excel_path, model, preprocess, device, image_base_dir="static"):
    df = pd.read_excel(excel_path)
    df.fillna("", inplace=True)

    features = []
    for path in df["main_image_url"]:
        full_path = os.path.join(image_base_dir, path)
        if os.path.exists(full_path):
            features.append(encode_image(full_path, model, preprocess, device))
        else:
            features.append(None)
    return df, features

# === Find best match and corresponding category ===
def find_best_match_category(user_img_path, model, preprocess, device, df, features):
    user_feat = encode_image(user_img_path, model, preprocess, device)
    similarities = [(user_feat @ feat.T).item() if feat is not None else -1 for feat in features]
    best_idx = similarities.index(max(similarities))
    return df.iloc[best_idx]["category"]

# === Get top 2 recommended items for a category ===
def get_top2_recommendations_by_category(df, category):
    filtered = df[df["category"] == category]
    all_suggestions = []

    for _, row in filtered.iterrows():
        all_suggestions.append(row["suggestion_1_name"].strip())
        all_suggestions.append(row["suggestion_2_name"].strip())

    top_two = Counter(all_suggestions).most_common(2)
    results = []
    for name, _ in top_two:
        matched_row = df[
            (df["suggestion_1_name"] == name) | (df["suggestion_2_name"] == name)
        ].iloc[0]
        if matched_row["suggestion_1_name"] == name:
            results.append((name, matched_row["suggestion_1_image"], matched_row["suggestion_1_price"]))
        else:
            results.append((name, matched_row["suggestion_2_image"], matched_row["suggestion_2_price"]))
    return results

# === Main recommend function ===
def recommend(user_img_path, excel_path="/content/drive/MyDrive/outfit_dataset/data.xlsx", image_base_dir="static"):
    model, preprocess, device = load_model()
    df, features = load_dataset_and_features(excel_path, model, preprocess, device, image_base_dir)
    category = find_best_match_category(user_img_path, model, preprocess, device, df, features)
    top2 = get_top2_recommendations_by_category(df, category)

    print(f"Upload the image into category: {category}")
    print("recomendation:")
    for name in top2:
        print(f"  - {name}")
    return category, top2

