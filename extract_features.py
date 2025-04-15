'''This code is for embedding all images in folder data, output as image_features.csv'''
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms

def get_image_paths(folder: str) -> list:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_resnet18():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC layer
    model.eval()
    return model

def extract_feature(img_path: str, model, transform) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        feature = model(img_tensor).squeeze().numpy()  # shape: (512,)
    return feature

def extract_features_for_folder(folder: str, model, transform):
    image_paths = get_image_paths(folder)
    features = []
    valid_paths = []

    for path in image_paths:
        try:
            feat = extract_feature(path, model, transform)
            features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    
    return np.array(features), valid_paths

def save_features_to_csv(features: np.ndarray, image_paths: list, output_path: str):
    df = pd.DataFrame(features)
    df["image_path"] = image_paths
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")

if __name__ == "__main__":
    image_folder = "data"
    output_dir = "csv"
    os.makedirs(output_dir, exist_ok=True)  # Ensure 'csv' folder exists
    output_csv = os.path.join(output_dir, "image_features.csv")

    transform = build_transform()
    resnet = load_resnet18()

    print("Extracting features...")
    features, paths = extract_features_for_folder(image_folder, resnet, transform)

    print(f"Extracted {len(features)} feature vectors.")
    save_features_to_csv(features, paths, output_csv)

