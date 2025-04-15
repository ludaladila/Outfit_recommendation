# Outfit_recommendation

# 0. Project Overview
## Key Features
## Evaluation Metric
# 1. Running Instruction
# 2. Data
image_dataset: https://drive.google.com/drive/folders/1hAGdJju8bOw-_z2MK0L9RHW1VW9qDCNZ
# 3. Approaches
## Naive Mean Model
## Non-Deep Learning Models
### Data Preparation
For the non-deep learning model approach, I define a complete outfit as one top + one bottom + one other items(e.g. sunglasses, shoes, perfume). 
1. Embed all images through a pretrained ResNet-18 model.
2. Use KMeans to group images into three clusters based on the embeddings.
3. Based on the cluster, run a script that randomly pick one item in each cluster to form 700 complete outfits, and let LlaVa rate these outfits.
### RandomForestRegressor
1. Take the 700 complete outfits and split them into training, validation, and testing set. Saved the testing set and keep them untouched. 
2. Perform 5-fold cross-validation on the training data to evaluate model performance and average the validation MSE across folds.
3. Saved the model.
### Evaluation
1. Run a script to generate the complete outfit based on images in testing set.
2. Do coisine similarity between items. Keep the top 3 outfit that have nearest coisine similarity, let the saved RandomForestRegressor rate them and keep the one with highest rating.
3. Let LlaVa rate these outfits again and average the score of these complete items.
### Results
- On the testing set, the average LLaVA rating for outfits increased from **7.74 to 8.03** after implementing the RandomForestRegressor.
- Comparing the ratings from the trained RandomForestRegressor and LLaVA, the **MSE is 4.83**.


## Deep Learning Models
# 4. Application
