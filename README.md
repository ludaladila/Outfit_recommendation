# Outfit_recommendation

# 0. Project Overview

Imagine walking into a outlet to buy some clothes for the Spring. Or shopping online on websites like Zara or H&M. There are always way too many options, which leads to two issues for customers: 1. decision paralysis 2. uncertainty in preference. To elaborate on the latter, assume you found a t-shirt that looks nice. But how are you going to match it with other clothing items you have? What will this go well with? Right now, there's no easy way to get immediate feedback to these questions.

So we decided to build a web app where users can upload images of clothing items, and we recommend complementary items to "complete the look". In essence, we are an AI-based personalized style coordinator, helping users brainstorm outfits.

## Key Features

- **Model Approaches:**
  - **Naive Mean Model:** A simple CLIP-based baseline that recommends the most frequently co-occurring items within the predicted category of the input image.
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques. We use a combination of feature extraction using ResNet and a Random Forest model to train our model.
  - **Deep Learning Models:** A Siamese Neural Network with a CLIP backbone, fine-tuned using a contrastive loss objective on the last layer of the CLIP architecture.

## Evaluation Metric

Since the approaches we used on three models are really different and might not be fair to use a same method to evaluate across all of them, so we will provide the detail of evaluation of each model separately.

# 1. Running Instruction

### Run the demo

[Outfit_recommendation](https://huggingface.co/spaces/yiqing111/Outfit_recommendation) .

### Run the app locally

#### **Environment Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Outfit_recommendation.git](https://github.com/ludaladila/Outfit_recommendation.git
   cd Outfit_recommendation
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the following files/folders are present:
   - `best_model.pt` — pretrained Siamese model checkpoint
   - `faiss_indices/` — FAISS index files for each category
   - `product_metadata.json` — metadata for product names, categories, prices, image paths
   - `all_product_images.zip` — zipped archive of all product images（download form [image_dataset](https://drive.google.com/drive/folders/1hAGdJju8bOw-_z2MK0L9RHW1VW9qDCNZ))

#### **Run the App**

```bash
streamlit run app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

# 2. Data

image_dataset:[image_dataset](https://drive.google.com/drive/folders/1hAGdJju8bOw-_z2MK0L9RHW1VW9qDCNZ)

It implemented a two-stage Selenium-based scraping pipeline to collect structured outfit data from the H&M website. The first stage collects product links across categories and genders, while the second stage visits each product page to extract detailed metadata and the top 2 recommended matching items. The final dataset (data.xlsx) includes product names, prices, descriptions, image URLs, and suggested pairings, which supports training and evaluation of our outfit recommendation models.

# 3. Approaches

## Naive Mean Model

### Description

The naive_model implements a CLIP-based category-matching recommendation baseline. It encodes a user-uploaded image and finds the most similar product from the dataset using cosine similarity in the CLIP embedding space. Based on the matched product’s category, the script then recommends the top-2 most frequently suggested items from that category across the dataset.

### Evaluation

We evaluate recommendation quality using LLaVA by generating a natural-language prompt for each outfit. The model returns a 0–10 score with reasoning, simulating human judgment.

### Result

The average result is 6.51 and for each category
| Category | Top 1 | Top 2 |
| --- | --- | --- |
| Cardigans & Sweaters | KNIT CARDIGAN | BALLET FLATS |
| Hoodies & Sweatshirts | SNEAKERS | BAGGY JEANS |
| Jackets & Coats | SNEAKERS | CHUNKY DERBY SHOES |
| Jeans | LEATHER BELT | 5-PACK SOCKS |
| Pants | SNEAKERS | CHUNKY DERBY SHOES |
| Polos | SNEAKERS | LEATHER BELT |
| Shirts & Blouses | MICROFIBER CAMISOLE TOP | WIDE HIGH JEANS |
| Shorts | 5-PACK SPORTS SOCKS WITH DRYMOVE™ | 5-PACK SOCKS |
| Suits & Blazers | LEATHER BELT | REGULAR FIT FINE-KNIT T-SHIRT |
| Sweaters & Cardigans | SNEAKERS | SLIM FIT RIBBED TANK TOP |
| Tops & T-Shirts | DENIM JACKET | SANDALS |

## Non-Deep Learning Models

### Data Preparation

For the non-deep learning model approach, We define a complete outfit as one top + one bottom + one other items(e.g. sunglasses, shoes, perfume).

1. Embedded all images through a pretrained ResNet-18 model.
2. Used KMeans to group images into three clusters based on the embeddings.
3. Based on the cluster, ran a script that randomly picked one item in each cluster to form 700 complete outfits, and let LlaVa rate these outfits from 1-10.

### RandomForestRegressor

1. Took the 700 complete outfits and splitted them into training, validation, and testing set. Saved the testing set and kept them untouched.
2. Performed 5-fold cross-validation on the training data(X: outfits, Y: LlaVa score) to evaluate model performance and average the validation MSE across folds.
3. Saved the model.

### Evaluation

1. Ran a script to generate the complete outfit based on images in testing set.
2. Computed coisine similarity between items. Kept the top 3 outfit that have nearest coisine similarity, let the saved RandomForestRegressor rate them and keep the one with highest rating.
3. Let LlaVa rate these outfits again and averaged the score of these complete items.

### Results

- On the testing set, the average LLaVA rating for outfits increased from **7.74 to 8.03** after implementing the RandomForestRegressor.
- Comparing the ratings from the trained RandomForestRegressor and LLaVA, the **MSE is 4.83**.

## Deep Learning Models

### Data Preparation

We wrote a script `scripts/generate_pairs.py` which generate a dataframe of (product_1, product_2, label) tuples, where `product_1` and `product_2` are product pairs and `label` indicates whether they are complementary or not. We use product name as the unique identifier for a given product, and also create a dictionary of product name and product attributes, where we can easily find product attributes such as path to image, price, description, etc. using the product name as a key. This dictionary is stored in `dataset/product_metadata.json`

### Model Architecture, Training, Results

We use CLIP as the backbone to a Siamese Neural Network. In our architecture, each image is embedded with our CLIP model, and a projection layer downsamples the dimensionality to 128. Then, we compute the similarity between the two embeddings and use contrastive loss to update the weights to the final layer of our CLIP model + projection layer (all other layers in the CLIP model are kept frozen to prevent catastrophic forgetting). A contrastive loss objective allows us to embed complementary products closer to each other and embed non-complementary products further away from one another.

We trained our model for 3 epochs with learning rate scheduling for robust results.

Using ROC-AUC as a validation metric, we obtain a score of 0.895. The weights to the fine-tuned model are available on Google Drive [at this link](https://drive.google.com/drive/folders/1hAGdJju8bOw-_z2MK0L9RHW1VW9qDCNZ?usp=sharing)

### Inference Optimization

To optimize inference, we construct a FAISS index which stores pre-computed embeddings of the products in our H&M database. This means that at inference time, we only need to compute the embedding of user-uploaded outfit, and conduct a O(N) similarity search over the index.

# 4. Previous approaches

Previous studies on outfit recommendation typically fall into two categories: retrieval-based methods and generation-based methods. Retrieval-based systems rely on visual similarity using image embeddings (e.g., from CNNs or CLIP) to find matching items, often leveraging co-occurrence patterns or compatibility graphs. Generation-based methods, on the other hand, use models such as GANs or Transformers to synthesize compatible outfit combinations or directly predict user preferences. Recent works have also explored graph neural networks, style embeddings, and multi-modal fusion to improve contextual matching. However, these methods often require large-scale labeled fashion datasets and careful training to avoid bias or overfitting.

# 5. Application

This is our demo [Outfit_recommendation](https://huggingface.co/spaces/yiqing111/Outfit_recommendation) .
Users can upload a photo and receive personalized outfit suggestions, filtered by clothing categories, with product images and prices displayed.

# 6. Ethics Statement

This project is intended for educational and research purposes only. The outfit recommendations are generated based on image similarity and learned style patterns, and may not always reflect diverse cultural, gender, or body-style preferences.

No personal data is stored or used beyond temporary image processing, and all recommendations are non-commercial and anonymized.

We encourage responsible and inclusive use of AI models in fashion applications, and welcome further work to address fairness and representation in personalized recommendation systems.
