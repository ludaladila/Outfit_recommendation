# Outfit_recommendation

# 0. Project Overview
(jason)
## Key Features
- **Model Approaches:** (yiqing)
  - **Naive Mean Model:**
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques. We use a combination of feature extraction using ResNet and a Random Forest model to train our model.
  - **Deep Learning Models:** (jason)
## Evaluation Metric
Since the approaches we used on three models are really different and might not be fair to use a same method to evaluate across all of them, so we will provide the detail of evaluation of each model separately.
# 1. Running Instruction
# 2. Data
image_dataset:[image_dataset](https://drive.google.com/drive/folders/1hAGdJju8bOw-_z2MK0L9RHW1VW9qDCNZ)

We implemented a two-stage Selenium-based scraping pipeline to collect structured outfit data from the H&M website. The first stage collects product links across categories and genders, while the second stage visits each product page to extract detailed metadata and the top 2 recommended matching items. The final dataset (data.xlsx) includes product names, prices, descriptions, image URLs, and suggested pairings, which supports training and evaluation of our outfit recommendation models.

# 3. Approaches
## Naive Mean Model
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
# 4. Application
 This is our demo [Outfit_recommendation](https://huggingface.co/spaces/yiqing111/Outfit_recommendation) . 
 Users can upload a photo and receive personalized outfit suggestions, filtered by clothing categories, with product images and prices displayed.
