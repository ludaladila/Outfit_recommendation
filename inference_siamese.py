import torch
import json
import zipfile
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from train_siamese_resnet50 import SiameseWithProjection
from tqdm import tqdm
import faiss
import numpy as np
from collections import defaultdict



def load_model(path_to_model, device="cuda"):
    # Reconstruct architecture
    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Your model class must match how it was trained
    model = SiameseWithProjection(clip_model=clip)

    # Load weights
    state_dict = torch.load(path_to_model, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict)
    return model, processor

def build_faiss_indices(model, processor, metadata_path, zip_path, index_out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    category_to_embeddings = defaultdict(list)
    category_to_ids = defaultdict(list)

    with zipfile.ZipFile(zip_path, 'r') as archive:
        for product_name, info in tqdm(metadata.items(), desc="Encoding products"):
            category = info['category']
            image_path = info['image']

            with archive.open(image_path) as f:
                image = Image.open(f).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.clip(pixel_values=inputs['pixel_values']).last_hidden_state[:, 0, :]
                emb = model.projector(emb).squeeze(0).cpu().numpy()

            category_to_embeddings[category].append(emb)
            category_to_ids[category].append(product_name)

    # Create one index per category
    index_db = {}
    for category, vectors in category_to_embeddings.items():
        array = np.stack(vectors).astype("float32")
        index = faiss.IndexFlatL2(array.shape[1])
        index.add(array)
        index_db[category] = {
            "faiss_index": index,
            "id_map": category_to_ids[category]
        }

    # Save index db to disk using faiss + JSON pointer map
    faiss.write_index(faiss.IndexFlatL2(1), index_out_path + "/dummy")  # Just to ensure folder exists
    for cat, obj in index_db.items():
        faiss.write_index(obj["faiss_index"], f"{index_out_path}/{cat}.index")
        with open(f"{index_out_path}/{cat}_ids.json", "w") as f:
            json.dump(obj["id_map"], f)

    print("FAISS indices saved.")


def recommend_with_faiss(user_image_path, model, processor, faiss_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Load user image
    image = Image.open(user_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        user_emb = model.clip(pixel_values=inputs['pixel_values']).last_hidden_state[:, 0, :]
        user_emb = model.projector(user_emb).squeeze(0).cpu().numpy().astype("float32")

    # Iterate over category indexes
    recommendations = {}
    import os
    for file in os.listdir(faiss_dir):
        if file.endswith(".index"):
            category = file.replace(".index", "")
            index = faiss.read_index(f"{faiss_dir}/{file}")
            with open(f"{faiss_dir}/{category}_ids.json") as f:
                id_map = json.load(f)

            D, I = index.search(user_emb[None], k=1)  # top-1
            top_idx = int(I[0][0])
            recommendations[category] = id_map[top_idx]

    return recommendations


if __name__ == '__main__':
    model, processor = load_model("best_model.pt")
    # Step 1: Build FAISS from product catalog
    build_faiss_indices(
        model=model,
        processor=processor,
        metadata_path="product_metadata.json",
        zip_path="all_product_images.zip",
        index_out_path="faiss_indices"
    )

    # Step 2: Recommend items given user image
    recs = recommend_with_faiss(
        user_image_path="./data/image_test.jpg",
        model=model,
        processor=processor,
        faiss_dir="faiss_indices"
    )

    print("Recommendations:")
    for cat, prod in recs.items():
        print(f"{cat}: {prod}")
