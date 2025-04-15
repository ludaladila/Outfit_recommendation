from transformers import CLIPImageProcessor, CLIPVisionModel, get_cosine_schedule_with_warmup
import zipfile
from tqdm import tqdm
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


# === Paths ===
CSV_PATH = "/content/product_pairs.csv"
META_PATH = "/content/product_details.json"
ZIP_PATH = "/content/data.zip"

# === Load metadata ===
with open(META_PATH, 'r') as f:
    product_meta = json.load(f)


pairs_df = pd.read_csv('./dataset/product_pairs.csv')

# --------- Dataset ---------
class ProductPairDataset(Dataset):
    def __init__(self, dataframe, product_metadata, processor, zip_file):
        self.df = dataframe
        self.metadata = product_metadata
        self.processor = processor
        self.zip_file = zip_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product1 = row['product_1']
        product2 = row['product_2']
        label = row['label']

        path1 = self.metadata[product1]['image']
        path2 = self.metadata[product2]['image']

        # Load images directly from ZIP without extracting
        with self.zip_file.open(path1) as file1:
            image1 = Image.open(file1).convert('RGB')
        with self.zip_file.open(path2) as file2:
            image2 = Image.open(file2).convert('RGB')

        anchor = self.processor(images=image1, return_tensors="pt")['pixel_values'].squeeze(0)
        candidate = self.processor(images=image2, return_tensors="pt")['pixel_values'].squeeze(0)

        return {
            'anchor_pixel_values': anchor,
            'candidate_pixel_values': candidate,
            'label': torch.tensor(label, dtype=torch.float)
        }


# --------- Model ---------
class SiameseWithProjection(nn.Module):
    def __init__(self, clip_model, projection_dim=128):
        super().__init__()
        self.clip = clip_model
        self.projector = nn.Sequential(
            nn.Linear(clip_model.config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, anchor, candidate):
        anchor_feat = self.clip(pixel_values=anchor).last_hidden_state[:, 0, :]
        candidate_feat = self.clip(pixel_values=candidate).last_hidden_state[:, 0, :]

        anchor_proj = self.projector(anchor_feat)
        candidate_proj = self.projector(candidate_feat)

        return anchor_proj, candidate_proj

# --------- Loss ---------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, candidate, label):
        distance = F.pairwise_distance(anchor, candidate)
        loss = (label * distance.pow(2) +
                (1 - label) * F.relu(self.margin - distance).pow(2))
        return loss.mean()

# --------- Load Data ---------
def load_data(pair_csv_path, metadata_json_path, processor, zip_file):
    print("Loading data...")
    with open(metadata_json_path, 'r') as f:
        product_data = json.load(f)

    pairs_df = pd.read_csv(pair_csv_path)
    dataset = ProductPairDataset(pairs_df, product_data, processor, zip_file)
    print(f"Loaded {len(dataset)} product pairs.")
    return dataset

# --------- Validate ---------
def validate(model, dataloader, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0.0
    distances_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            anchor = batch['anchor_pixel_values'].to(device)
            candidate = batch['candidate_pixel_values'].to(device)
            label = batch['label'].to(device)

            anchor_proj, candidate_proj = model(anchor, candidate)
            loss = loss_fn(anchor_proj, candidate_proj, label)
            total_loss += loss.item()

            distances = torch.norm(anchor_proj - candidate_proj, dim=1)
            distances_all.extend(distances.cpu().numpy())
            labels_all.extend(label.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Convert distances to similarity scores (lower distance = more similar)
    distances_all = np.array(distances_all)
    labels_all = np.array(labels_all)
    similarity_scores = -distances_all  # Higher = more similar

    try:
        roc_auc = roc_auc_score(labels_all, similarity_scores)
    except ValueError:
        roc_auc = float('nan')  # If only one class present

    try:
        pr_auc = average_precision_score(labels_all, similarity_scores)
    except ValueError:
        pr_auc = float('nan')

    return avg_loss, roc_auc, pr_auc


# --------- Training ---------
def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, model_save_path="best_model.pt", num_epochs=5, device="cuda"):
    print(f"Training on {device} for {num_epochs} epochs...")
    model.to(device)
    model.train()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in loop:
            anchor = batch['anchor_pixel_values'].to(device)
            candidate = batch['candidate_pixel_values'].to(device)
            label = batch['label'].to(device)

            anchor_proj, candidate_proj = model(anchor, candidate)
            loss = loss_fn(anchor_proj, candidate_proj, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_loss, roc_auc, pr_auc = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {roc_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model to {model_save_path}")

# --------- Main ---------
def main():
    print("Initializing processor and CLIP model...")
    # Hyperparameters
    pair_csv_path = "/content/product_pairs.csv"
    metadata_json_path = "/content/product_details.json"
    batch_size = 32
    num_epochs = 3
    model_save_path = "best_siamese_model.pt"

    lr = 2e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    zip_path = "data.zip"  # adjust as needed
    zip_file = zipfile.ZipFile(zip_path, 'r')

    # Load model & processor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    print("Freezing CLIP layers except the last transformer block...")
    # Freeze all layers
    for param in clip_model.parameters():
        param.requires_grad = False
  
    # Unfreeze the last transformer block
    for param in clip_model.vision_model.encoder.layers[-1].parameters():
        param.requires_grad = True

    zip_path = "data.zip"
    print(f"Opening ZIP archive from: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
      dataset = load_data(pair_csv_path, metadata_json_path, processor, zip_file)
  
      # Example: 80% train, 20% validation split
      train_size = int(0.8 * len(dataset))
      val_size = len(dataset) - train_size
      train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

      # Build model
      model = SiameseWithProjection(clip_model).to(device)
      loss_fn = ContrastiveLoss(margin=1.0)
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

      num_training_steps = len(train_loader) * num_epochs
      num_warmup_steps = int(0.1 * num_training_steps)
      scheduler = get_cosine_schedule_with_warmup(
          optimizer,
          num_warmup_steps=num_warmup_steps,
          num_training_steps=num_training_steps
      )

      # Train
      train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs=num_epochs, device=device)

# --- Run ---
if __name__ == "__main__":
    main()
