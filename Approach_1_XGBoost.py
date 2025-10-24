import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import spacy
from spacy.matcher import Matcher
import re

print("Prediction Script Started...")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

matcher = Matcher(nlp.vocab)
KNOWN_BRANDS = [
    'Charles Jacquin-St.Dalfour', 'Traditional Medicinals', 'Kirkland Signature', 'Celestial Seasonings', 'Lundberg Family Farms',
    'Stevia International', 'McCormick Gourmet', 'Stonewall Kitchen', 'Lehi Roller Mills', 'Community Coffee', 'Mystic Sprinkles',
    'Pacific Natural Foods', 'Marshalls Creek', 'Sweet Baby Rays', 'Pepperidge Farm', 'Crystal Light', 'Green Mountain',
    'Simply Organic', 'DaVinci Gourmet', 'Sweet Baby Ray', 'Bob\'s Red Mill', 'Nature Valley', 'Betty Crocker', 'Amazon Brand',
    'Blue Diamond', 'LA MOLISANA', 'Kettle Foods', 'Equal Exchange', 'Charleston Chew', 'Three Crabs Brand', 'Sunfood Superfood',
    'Blue Mountain', 'Imagine Foods', 'Taste of Thai', 'House Of Tsang', 'Monster Java', 'Amys Kitchen', 'Beach Cliff',
    'Amoretti', 'Goya Foods', 'Jelly Belly', 'Frontier Co', 'Special Tea', 'Trader Joe', 'Ocean Spray', 'Stash Tea',
    'Tiesta Tea', 'NOW Foods', 'Del Monte', 'Newmans Own', 'Underwood', 'Little Debbie', 'Wacky Mac', 'Private Selection',
    'Cbs Nuts', 'Spice World', 'Pickapeppa', 'Country Kitchen', 'Pace', 'Marukan', 'Martha White', 'Morton & Bassett',
    'Talk O Texas', 'Kernel Seasons', 'Dynamic Health', 'SUNRIDGE FARMS', 'Gold Peak', 'Uncle Lees', 'Grandmas',
    'Campbell', 'Kellogg', 'Davidson', 'Smucker', 'Annie\'s', 'HERSHEY', 'NY Spice', 'Beulah', 'Morton', 'Snyder',
    'Vintaj', 'Lawrys', 'Hain', 'Reese', 'Try Me', 'Kraft', 'Accent', 'Topps', 'Olivari', 'Pocky', 'Hereford',
    'GLUTINO', 'Sprite', 'Zapps', 'Manamim', 'Wegmans', 'Knorr', 'Iberia', 'Mentos', 'Peeps', 'Jiffy', 'Goya', 'Hunt'
]
patterns = [[{'TEXT': token.text} for token in nlp(brand)] for brand in KNOWN_BRANDS]
matcher.add("BRAND_PATTERNS", patterns)

print("Loading pre-trained sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- 2. Create Reusable Feature Engineering Functions ---

def find_brand_spacy(item_name):
    doc = nlp(item_name)
    matches = matcher(doc)
    if matches:
        return max([doc[start:end].text for _, start, end in matches], key=len)
    parts = item_name.split()
    return parts[0].replace(',', '').replace(':', '') if parts else "Unknown"

def process_dataframe(df, is_train=True):
    print(f"Processing {'training' if is_train else 'testing'} data...")

    def extract_base_features(text_content):
        text_content = str(text_content)
        name_match = re.search(r"Item Name:\s*(.*?)\n", text_content, re.DOTALL)
        item_name = name_match.group(1).strip() if name_match else ""
        brand = find_brand_spacy(item_name)
        value_match = re.search(r"Value:\s*([\d\.]+)", text_content)
        extracted_value = float(value_match.group(1)) if value_match else np.nan
        lower_text = text_content.lower()
        is_organic = 1 if 'organic' in lower_text else 0
        is_gluten_free = 1 if 'gluten free' in lower_text or ' gf' in lower_text else 0
        is_sugar_free = 1 if 'sugar free' in lower_text or 'no sugar' in lower_text else 0
        is_kosher = 1 if 'kosher' in lower_text else 0
        return pd.Series([item_name, brand, extracted_value, is_organic, is_gluten_free, is_sugar_free, is_kosher])

    df[['item_name', 'brand', 'final_value', 'is_organic', 'is_gluten_free', 'is_sugar_free', 'is_kosher']] = df['catalog_content'].apply(extract_base_features)
    df.dropna(subset=['item_name'], inplace=True)

    print(f"Generating embeddings for {len(df)} items...")
    embeddings = embedding_model.encode(df['item_name'].tolist(), show_progress_bar=True)
    embedding_df = pd.DataFrame(embeddings, index=df.index, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])

    numerical_features = df[['final_value', 'is_organic', 'is_gluten_free', 'is_sugar_free', 'is_kosher']]
    X = pd.concat([numerical_features.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)

    for col in numerical_features.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    if is_train:
        y = np.log1p(df['price'])
        return X, y
    else:
        return X

# --- 3. Load and Process Training Data ---
try:
    train_df = pd.read_csv('train_cleaned.csv')
    print(f"\nSuccessfully loaded 'train_cleaned.csv'. Shape: {train_df.shape}")
except FileNotFoundError:
    print("Error: 'train_cleaned.csv' not found. Please ensure it's in the same directory.")
    exit()

X_train, y_train = process_dataframe(train_df, is_train=True)
print(f"Training features created. Shape: {X_train.shape}")


# --- 4. Train the XGBoost Model ---
print("\nTraining XGBoost Regressor model on the full training dataset...")
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=7, random_state=42)
xgbr.fit(X_train, y_train, verbose=False)
print("Model training complete.")


# --- 5. Load and Process Test Data ---
try:
    test_df = pd.read_csv('test.csv')
    print(f"\nSuccessfully loaded 'test.csv'. Shape: {test_df.shape}")
except FileNotFoundError:
    print("Error: 'test.csv' not found. Please ensure it's in the same directory.")
    exit()

test_ids = test_df['sample_id']
X_test = process_dataframe(test_df, is_train=False)
print(f"Test features created. Shape: {X_test.shape}")


# --- 6. Make Predictions and Save Submission File ---
print("\nMaking predictions on the test set...")
test_pred_log = xgbr.predict(X_test)
test_pred_actual = np.expm1(test_pred_log)

submission_df = pd.DataFrame({'sample_id': test_ids, 'price': test_pred_actual})
submission_df['price'] = submission_df['price'].clip(lower=0)

output_filename = 'submission.csv'
submission_df.to_csv(output_filename, index=False)

print(f"\n=============================================")
print(f"Successfully generated predictions!")
print(f"Output saved to '{output_filename}'")
print("=============================================")
print("\n--- Sample of Predictions ---")
print(submission_df.head())
print("\nScript finished.")