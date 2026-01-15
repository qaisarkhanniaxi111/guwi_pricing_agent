"""
Compare API Prediction vs Direct Model Prediction
Find out why they give different results
"""

import pandas as pd
import requests
import json
from gutter_price_model import GutterPricePredictor

print("=" * 80)
print("API vs MODEL COMPARISON")
print("=" * 80)

# Your test data from Postman
test_data = {
    "Roof Type": "Composition",
    "Home Square Footage": 2267,
    "Home Value": 1753400,
    "Average Home value in Zip code": 918825,
    "Number of Stories": "N/A"
}

print("\n1. TEST DATA:")
print("-" * 80)
for key, val in test_data.items():
    print(f"   {key}: {val}")

# Test 1: Direct model prediction
print("\n2. DIRECT MODEL PREDICTION:")
print("-" * 80)

try:
    predictor = GutterPricePredictor()
    predictor.load_model('gutter_price_model.pkl')
    
    df = pd.DataFrame([test_data])
    
    # Show what goes into the model
    print("\n   Data going to model:")
    for col in df.columns:
        print(f"      {col}: {df[col].values[0]}")
    
    prediction = predictor.predict(df)
    print(f"\n   ✅ Direct prediction: ${prediction:.2f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: API prediction
print("\n3. API PREDICTION:")
print("-" * 80)

try:
    response = requests.post(
        'http://127.0.0.1:8080/predict/gutter_clearing',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        api_price = result.get('predicted_price', 0)
        print(f"\n   ✅ API prediction: ${api_price:.2f}")
        
        print(f"\n   Full API response:")
        print(f"   {json.dumps(result, indent=6)}")
    else:
        print(f"   ❌ API Error: {response.status_code}")
        print(f"   {response.text}")
        
except Exception as e:
    print(f"   ❌ Error calling API: {e}")

# Test 3: Check if models are different
print("\n4. MODEL FILE CHECK:")
print("-" * 80)

import os
import hashlib

model_file = 'gutter_price_model.pkl'
if os.path.exists(model_file):
    # Get file size and hash
    size = os.path.getsize(model_file)
    
    with open(model_file, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    print(f"   Model file: {model_file}")
    print(f"   Size: {size:,} bytes")
    print(f"   MD5: {file_hash}")
    
    # Load and check details
    predictor = GutterPricePredictor()
    predictor.load_model(model_file)
    
    print(f"\n   Model type: {predictor.best_model_name}")
    print(f"   Features: {len(predictor.feature_names)}")
    print(f"   Features list:")
    for i, feat in enumerate(predictor.feature_names, 1):
        print(f"      {i:2}. {feat}")

# Test 4: Check what the API actually sends
print("\n5. WHAT API ACTUALLY SENDS:")
print("-" * 80)

print("\n   Your request:")
print(f"   {json.dumps(test_data, indent=6)}")

print("\n   After API applies defaults (check app.py):")
print("   The API adds missing fields with defaults")
print("   This might change the prediction!")

# Diagnosis
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nPossible causes of difference:")
print("  1. API adds default values that change the prediction")
print("  2. API uses a different model file")
print("  3. Data transformation is different")
print("  4. Column mapping issues")

print("\n🔧 NEXT STEPS:")
print("  1. Check if API is loading the correct model file")
print("  2. Add ALL fields to your API request (don't rely on defaults)")
print("  3. Compare feature values that go into the model")

print("\n" + "=" * 80)