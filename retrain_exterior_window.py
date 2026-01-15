"""
Retrain ONLY Exterior Window Model
Ensures it has N/A roof types
"""

import pandas as pd
import numpy as np
from gutter_price_model import GutterPricePredictor

print("=" * 80)
print("RETRAINING EXTERIOR WINDOW MODEL")
print("=" * 80)

csv_file = 'exteriorwindowtrain.csv'
price_column = 'Exterior Window Cleaning'
model_file = 'exterior_window_model.pkl'

# Load data
print(f"\n1. Loading {csv_file}...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"   ✅ Loaded {len(df)} records")
except:
    try:
        df = pd.read_csv(csv_file, encoding='windows-1252')
        print(f"   ✅ Loaded {len(df)} records (Windows-1252)")
    except:
        df = pd.read_csv(csv_file, encoding='latin-1')
        print(f"   ✅ Loaded {len(df)} records (Latin-1)")

# Check price column
if price_column not in df.columns:
    print(f"❌ Price column '{price_column}' not found!")
    print(f"Available: {df.columns.tolist()}")
    exit(1)

# Rename for model
df['Gutter Clearing'] = df[price_column]

# Handle roof type - ADD N/A VALUES
print(f"\n2. Adding N/A roof types...")

roof_col = None
if 'Roof Type' in df.columns:
    roof_col = 'Roof Type'
elif 'Roof Type/ Material' in df.columns:
    roof_col = 'Roof Type/ Material'

if roof_col:
    # Fill missing with N/A (don't use 'Unknown')
    df[roof_col] = df[roof_col].fillna('N/A')
    df[roof_col] = df[roof_col].replace(['', 'nan', 'NaN', 'none', 'None'], 'N/A')
    
    # Ensure at least 5% are N/A
    na_count = (df[roof_col] == 'N/A').sum()
    total_needed = max(int(len(df) * 0.05), 10)
    
    if na_count < total_needed:
        # Get random indices to convert to N/A
        non_na_indices = df[df[roof_col] != 'N/A'].index
        if len(non_na_indices) > 0:
            num_to_convert = min(total_needed - na_count, len(non_na_indices))
            convert_indices = np.random.choice(non_na_indices, size=num_to_convert, replace=False)
            df.loc[convert_indices, roof_col] = 'N/A'
            print(f"   ✅ Added {num_to_convert} 'N/A' roof types")
    
    # Show distribution
    print(f"\n   Roof Type Distribution:")
    roof_counts = df[roof_col].value_counts()
    for roof_type, count in roof_counts.head(10).items():
        print(f"      {roof_type:<20} {count:>6} ({count/len(df)*100:.1f}%)")
else:
    print("   ⚠️  No roof type column found!")

# Remove missing prices
df = df.dropna(subset=['Gutter Clearing'])
print(f"\n3. Valid records: {len(df)}")

if len(df) < 100:
    print("❌ Not enough data!")
    exit(1)

# Train model
print(f"\n4. Training model...")
predictor = GutterPricePredictor()
predictor.fit(df)

print(f"   ✅ Model: {predictor.best_model_name}")
print(f"   ✅ Features: {len(predictor.feature_names)}")

# Check if N/A is in encoder
if 'Roof Type' in predictor.label_encoders:
    classes = predictor.label_encoders['Roof Type'].classes_
    print(f"\n   Roof types model knows: {list(classes)}")
    
    if 'N/A' in classes:
        print(f"   ✅ N/A is in the model!")
    else:
        print(f"   ❌ N/A NOT in model - something went wrong!")

# Save model
print(f"\n5. Saving {model_file}...")
predictor.save_model(model_file)

# Get metrics
best_metrics = predictor.model_metrics.get(predictor.best_model_name, {})
print(f"\n✅ SUCCESS!")
print(f"   Model: {predictor.best_model_name}")
print(f"   MAE: ${best_metrics.get('MAE', 0):.2f}")
print(f"   R²: {best_metrics.get('R2', 0):.4f}")

print("\n" + "=" * 80)
print("Now test the API with exterior_window!")
print("=" * 80)