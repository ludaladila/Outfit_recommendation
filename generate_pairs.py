import pandas as pd
import json
from itertools import combinations

def merge_details(existing_data, new_data):
    """
    Merge new_data into existing['details'].
    Always update image if available.
    Add new keys if they don't exist yet.
    Ensure existing keys are consistent.
    """

    for key, value in new_data.items():
        if key == 'image':
            # Always update the image
            existing_data[key] = value
        elif value not in [None, "", float('nan')] and key not in existing_data:
            # Add new information
            existing_data[key] = value
        elif key in existing_data and existing_data[key] != value and pd.notna(value):
            # Check consistency for existing keys (like price)
            if key == 'price':
                # Convert to float and compare
                try:
                    old_price = float(existing_data[key])
                    new_price = float(value)
                    if old_price != new_price:
                        print(f"⚠️ Price mismatch for {key} - keeping original: {old_price}, new: {new_price}")
                except:
                    pass
            else:
                print(f"⚠️ Inconsistent field '{key}' for product. Keeping original: {existing_data[key]} vs new: {value}")

def create_product_dict():
    # Load Excel file
    df = pd.read_excel("data.xlsx")

    # Dictionary to hold all product data
    product_objects = {}

    # Loop through each row in data
    for _, row in df.iterrows():
        # Main product
        main_name = row['product_name']
        main_data = {
            'gender': row['grnder'],
            'category': row['category'],
            'href': row['href'],
            'price': row['price'],
            'desc': row['description'],
            'image': row['main_image_url']
        }

        if main_name not in product_objects:
            product_objects[main_name] = main_data
        else:
            merge_details(product_objects[main_name], main_data)

        # Suggestion 1
        sugg1 = row['suggestion_1_name']
        if pd.notna(sugg1):
            sugg1_data = {
                'image': row['suggestion_1_image'],
                'price': row['suggestion_1_price']
            }
            if sugg1 not in product_objects:
                product_objects[sugg1] = sugg1_data
            else:
                merge_details(product_objects[sugg1], sugg1_data)

        # Suggestion 2
        sugg2 = row['suggestion_2_name']
        if pd.notna(sugg2):
            sugg2_data = {
                'image': row['suggestion_2_image'],
                'price': row['suggestion_2_price']
            }
            if sugg2 not in product_objects:
                    product_objects[sugg2] = sugg2_data
            else:
                merge_details(product_objects[sugg2], sugg2_data)


    with open("product_details.json", "w", encoding="utf-8") as f:
        json.dump(product_objects, f, indent=4)

    return product_objects


def make_pairs(product_dict):
    # Load Excel file
    df = pd.read_excel("data.xlsx")

    # Store all (product1, product2, label) rows
    pair_dataset = []

    # Set to track known complimentary pairs for checking later
    complimentary_pairs = set()

    # Store product metadata (gender, category) for filtering
    # product_meta = {}

    # First pass: build product_meta and record complimentary pairs
    for _, row in df.iterrows():
        product = row['product_name']
        # gender = row['grnder']
        # category = row['category']
        
        # # Store metadata
        # product_meta[product] = {'gender': gender, 'category': category}
        
        # Get suggestions
        suggestions = []
        if pd.notna(row['suggestion_1_name']):
            suggestions.append(row['suggestion_1_name'])
        if pd.notna(row['suggestion_2_name']):
            suggestions.append(row['suggestion_2_name'])

        for sugg in suggestions:
            # Record complimentary pair
            pair_dataset.append((product, sugg, 1))
            complimentary_pairs.add((product, sugg))

    # Get all unique products that appeared as a main product
    product_names = df['product_name'].dropna().unique()

    # Second pass: generate non-complimentary pairs
    for prod1, prod2 in combinations(product_names, 2):
        prod1_details = product_dict.get(prod1)
        prod2_details = product_dict.get(prod2)

        # Skip if missing metadata
        if not prod1_details or not prod2_details:
            continue

        # Rule: same gender
        if prod1_details['gender'] != prod2_details['gender']:
            continue

        # Rule: different categories
        if prod1_details['category'] == prod2_details['category']:
            continue

        # Rule: not already complimentary
        if (prod1, prod2) in complimentary_pairs or (prod2, prod1) in complimentary_pairs:
            continue

        # If all rules pass, it's non-complimentary
        pair_dataset.append((prod1, prod2, 0))

    pairs_df = pd.DataFrame(pair_dataset, columns=['product_1', 'product_2', 'label'])
    pairs_df.to_csv('product_pairs.csv', index=False)

if __name__ == "__main__":
    product_dict = create_product_dict()
    make_pairs(product_dict)