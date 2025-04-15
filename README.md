# Outfit_recommendation

# 0. Project Overview
(jason)
## Key Features
- **Model Approaches:** 
  - **Naive Mean Model:** A simple CLIP-based baseline that recommends the most frequently co-occurring items within the predicted category of the input image.
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques. We use a combination of feature extraction using ResNet and a Random Forest model to train our model.
  - **Deep Learning Models:** (jason)
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
For the non-deep learning model approach, I define a complete outfit as one top + one bottom + one other items(e.g. sunglasses, shoes, perfume). 
1. Embedded all images through a pretrained ResNet-18 model.
2. Used KMeans to group images into three clusters based on the embeddings.
3. Based on the cluster, ran a script that randomly picked one item in each cluster to form 700 complete outfits, and let LlaVa rate these outfits.
### RandomForestRegressor
1. Took the 700 complete outfits and splitted them into training, validation, and testing set. Saved the testing set and kept them untouched. 
2. Performed 5-fold cross-validation on the training data(X: outfits, Y: LlaVa score) to evaluate model performance and average the validation MSE across folds.
3. Saved the model.
### Evaluation
1. Ran a script to generate the complete outfit based on images in testing set.
2. Did coisine similarity between items. Kept the top 3 outfit that have nearest coisine similarity, let the saved RandomForestRegressor rate them and keep the one with highest rating.
3. Let LlaVa rate these outfits again and averaged the score of these complete items.
### Results
- On the testing set, the average LLaVA rating for outfits increased from **7.74 to 8.03** after implementing the RandomForestRegressor.
- Comparing the ratings from the trained RandomForestRegressor and LLaVA, the **MSE is 4.83**.
## Deep Learning Models
# Previous approaches 
Previous studies on outfit recommendation typically fall into two categories: retrieval-based methods and generation-based methods. Retrieval-based systems rely on visual similarity using image embeddings (e.g., from CNNs or CLIP) to find matching items, often leveraging co-occurrence patterns or compatibility graphs. Generation-based methods, on the other hand, use models such as GANs or Transformers to synthesize compatible outfit combinations or directly predict user preferences. Recent works have also explored graph neural networks, style embeddings, and multi-modal fusion to improve contextual matching. However, these methods often require large-scale labeled fashion datasets and careful training to avoid bias or overfitting.
# 4. Application
 This is our demo [Outfit_recommendation](https://huggingface.co/spaces/yiqing111/Outfit_recommendation) . 
 Users can upload a photo and receive personalized outfit suggestions, filtered by clothing categories, with product images and prices displayed.
# 5. Ethics Statement
This project is intended for educational and research purposes only. The outfit recommendations are generated based on image similarity and learned style patterns, and may not always reflect diverse cultural, gender, or body-style preferences.

No personal data is stored or used beyond temporary image processing, and all recommendations are non-commercial and anonymized.

We encourage responsible and inclusive use of AI models in fashion applications, and welcome further work to address fairness and representation in personalized recommendation systems.
