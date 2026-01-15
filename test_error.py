"""
ULTIMATE DIAGNOSTIC - Find the EXACT problem
"""

import pandas as pd
import joblib

print("=" * 70)
print("FINDING THE EXACT PROBLEM")
print("=" * 70)

# Step 1: Check the model file
print("\n1. CHECKING MODEL FILE")
print("-" * 70)
try:
    model = joblib.load('gutter_price_model.pkl')
    print("✅ Model loaded")
    
    if hasattr(model, 'feature_names'):
        print(f"\nModel expects {len(model.feature_names)} features:")
        for i, feat in enumerate(model.feature_names, 1):
            print(f"  {i:2}. {feat}")
        
        # Check for duplicates
        duplicates = []
        for i, feat in enumerate(model.feature_names):
            if model.feature_names.count(feat) > 1 and feat not in duplicates:
                duplicates.append(feat)
        
        if duplicates:
            print(f"\n❌ MODEL HAS DUPLICATE FEATURES: {duplicates}")
            print("\n🔧 THE FIX:")
            print("   del *.pkl")
            print("   python train_all_services.py")
        else:
            print("\n✅ No duplicates in model features")
    
    # Check label encoders
    if hasattr(model, 'label_encoders'):
        print(f"\nLabel encoders in model:")
        for key in model.label_encoders.keys():
            print(f"  - {key}")
            classes = model.label_encoders[key].classes_
            print(f"    Classes: {list(classes)}")
            
            if 'N/A' not in classes:
                print(f"    ⚠️  'N/A' NOT in classes!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Step 2: Simulate what API does
print("\n" + "=" * 70)
print("2. SIMULATING API REQUEST")
print("-" * 70)

# Your request
data = {'Roof Type': 'N/A', 'Home Square Footage': 2000, 'Number of Stories': 2}
print(f"\nOriginal request: {data}")

# Load the actual app.py defaults
print("\nLoading app.py to see what it sends...")
try:
    with open('app.py', 'r') as f:
        content = f.read()
        if "'Roof Type': " in content and "'Roof Type/ Material': " in content:
            print("⚠️  app.py has BOTH 'Roof Type' AND 'Roof Type/ Material' in defaults!")
            print("   This creates duplicate columns!")
        elif "'Roof Type/ Material': " in content:
            print("✅ app.py uses 'Roof Type/ Material' (good)")
        elif "'Roof Type': " in content:
            print("✅ app.py uses 'Roof Type' (good)")
except:
    print("❌ Could not read app.py")

# Step 3: Check what normalize_columns does
print("\n" + "=" * 70)
print("3. CHECKING COLUMN NORMALIZATION")
print("-" * 70)

test_df = pd.DataFrame([{
    'Roof Type': 'N/A',
    'Roof Type/ Material': 'N/A',
    'Home Square Footage': 2000
}])

print(f"\nTest DataFrame columns BEFORE normalize:")
for col in test_df.columns:
    if 'Roof' in col:
        print(f"  ⭐ {col}")
    else:
        print(f"     {col}")

try:
    from gutter_price_model import GutterPricePredictor
    predictor = GutterPricePredictor()
    
    normalized = predictor.normalize_columns(test_df)
    
    print(f"\nTest DataFrame columns AFTER normalize:")
    for col in normalized.columns:
        if 'Roof' in col:
            print(f"  ⭐ {col}")
        else:
            print(f"     {col}")
    
    roof_cols = [col for col in normalized.columns if 'Roof' in col and 'Type' in col]
    if len(roof_cols) > 1:
        print(f"\n❌ PROBLEM FOUND: {len(roof_cols)} roof type columns after normalization!")
        print(f"   Columns: {roof_cols}")
        print("\n🔧 THE FIX:")
        print("   The normalize_columns() function is creating duplicates")
        print("   Need to fix gutter_price_model.py line 39-49")
    elif len(roof_cols) == 1:
        print(f"\n✅ Only 1 roof type column: {roof_cols[0]}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Step 4: The real test
print("\n" + "=" * 70)
print("4. ACTUAL PREDICTION TEST")
print("-" * 70)

print("\nTrying prediction with 'Roof Type/ Material' ONLY...")
test_data1 = pd.DataFrame([{
    'Roof Type/ Material': 'N/A',
    'Home Square Footage': 2000,
    'Home Value': 1800,
    'Average Home value in Zip code': 1700,
    'Number of Stories': 2
}])

try:
    from gutter_price_model import GutterPricePredictor
    model = GutterPricePredictor()
    model.load_model('gutter_price_model.pkl')
    
    price = model.predict(test_data1)
    print(f"✅ SUCCESS with 'Roof Type/ Material': ${price:.2f}")
except Exception as e:
    print(f"❌ FAILED with 'Roof Type/ Material': {e}")

print("\nTrying prediction with 'Roof Type' ONLY...")
test_data2 = pd.DataFrame([{
    'Roof Type': 'N/A',
    'Home Square Footage': 2000,
    'Home Value': 1800,
    'Average Home value in Zip code': 1700,
    'Number of Stories': 2
}])

try:
    price = model.predict(test_data2)
    print(f"✅ SUCCESS with 'Roof Type': ${price:.2f}")
except Exception as e:
    print(f"❌ FAILED with 'Roof Type': {e}")

print("\nTrying prediction with BOTH columns...")
test_data3 = pd.DataFrame([{
    'Roof Type': 'N/A',
    'Roof Type/ Material': 'N/A',
    'Home Square Footage': 2000,
    'Home Value': 1800,
    'Average Home value in Zip code': 1700,
    'Number of Stories': 2
}])

try:
    price = model.predict(test_data3)
    print(f"✅ SUCCESS with BOTH: ${price:.2f}")
except Exception as e:
    print(f"❌ FAILED with BOTH columns: {e}")
    print("\n🎯 THIS IS YOUR PROBLEM!")
    print("   Your app.py is sending BOTH columns")

# Final diagnosis
print("\n" + "=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)

print("\nThe error 'y should be a 1d array, got an array of shape (1, 2)'")
print("means the model is receiving 2 values when it expects 1.")
print("\nThis happens when:")
print("  1. Your DataFrame has BOTH 'Roof Type' and 'Roof Type/ Material'")
print("  2. normalize_columns() maps both to 'Roof Type'")
print("  3. Now you have DUPLICATE 'Roof Type' columns")
print("  4. label_encoder sees array with shape (1, 2) instead of (1,)")
print("\n🔧 THE FIX:")
print("   Make sure app.py sends ONLY ONE roof type column")
print("   Check your app.py defaults - should have only one, not both")

print("\n" + "=" * 70)