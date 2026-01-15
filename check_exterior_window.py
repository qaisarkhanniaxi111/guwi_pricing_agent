"""
Check Exterior Window Training CSV
"""

import pandas as pd
import joblib

print("=" * 80)
print("EXTERIOR WINDOW DIAGNOSTIC")
print("=" * 80)

# Check CSV
print("\n1. CHECKING CSV FILE:")
print("-" * 80)

try:
    df = pd.read_csv('exteriorwindowtrain.csv', encoding='utf-8', nrows=5)
except:
    try:
        df = pd.read_csv('exteriorwindowtrain.csv', encoding='windows-1252', nrows=5)
    except:
        df = pd.read_csv('exteriorwindowtrain.csv', encoding='latin-1', nrows=5)

print(f"Columns in CSV:")
for i, col in enumerate(df.columns, 1):
    if 'Roof' in col and 'Type' in col:
        print(f"  {i:2}. {col} ⭐⭐ ROOF TYPE COLUMN")
    else:
        print(f"  {i:2}. {col}")

# Check model
print("\n2. CHECKING MODEL FILE:")
print("-" * 80)

try:
    from gutter_price_model import GutterPricePredictor
    
    predictor = GutterPricePredictor()
    predictor.load_model('exterior_window_model.pkl')
    
    print(f"Model features ({len(predictor.feature_names)}):")
    for i, feat in enumerate(predictor.feature_names, 1):
        print(f"  {i:2}. {feat}")
    
    # Check label encoder
    if hasattr(predictor, 'label_encoders'):
        print(f"\nLabel encoders:")
        for key in predictor.label_encoders.keys():
            encoder = predictor.label_encoders[key]
            classes = list(encoder.classes_)
            print(f"  {key}:")
            print(f"    Classes: {classes}")
            
            if 'N/A' in classes:
                print(f"    ✅ N/A is present")
            else:
                print(f"    ❌ N/A is MISSING!")

except Exception as e:
    print(f"❌ Error: {e}")

# Test prediction
print("\n3. TEST PREDICTION:")
print("-" * 80)

test_data = {
    'Roof Type': 'N/A',
    'Roof Type/ Material': 'N/A',
    'Home Square Footage': 2000,
    'Home Value': 1800,
    'Average Home value in Zip code': 1700,
    'Number of Stories': 1
}

print("Test data:")
for k, v in test_data.items():
    print(f"  {k}: {v}")

try:
    from gutter_price_model import GutterPricePredictor
    
    predictor = GutterPricePredictor()
    predictor.load_model('exterior_window_model.pkl')
    
    df_test = pd.DataFrame([test_data])
    
    # Check columns after normalize
    df_normalized = predictor.normalize_columns(df_test)
    
    print(f"\nColumns after normalize:")
    for col in df_normalized.columns:
        if 'Roof' in col:
            print(f"  ⭐ {col}")
    
    # Try prediction
    prediction = predictor.predict(df_test)
    print(f"\n✅ SUCCESS! Predicted: ${prediction:.2f}")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)