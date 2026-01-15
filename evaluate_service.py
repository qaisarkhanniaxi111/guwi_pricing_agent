"""
Service-Specific Model Evaluator
Evaluates any service model with its corresponding test data
"""

import pandas as pd
import numpy as np
import sys
from gutter_price_model import GutterPricePredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_service_model(model_path, test_csv_path, price_column_name='Gutter Clearing'):
    """
    Evaluate a service model
    
    Args:
        model_path: Path to .pkl model file
        test_csv_path: Path to test CSV
        price_column_name: Name of price column in CSV
    """
    print("=" * 80)
    print("SERVICE MODEL EVALUATION")
    print("=" * 80)
    
    # Load the model
    print(f"\n1. Loading model from {model_path}...")
    try:
        predictor = GutterPricePredictor()
        predictor.load_model(model_path)
        print(f"   ✅ Model loaded: {predictor.best_model_name}")
        print(f"   ✅ Features used: {len(predictor.feature_names)}")
        print(f"\n   📋 Model features:")
        for i, feat in enumerate(predictor.feature_names, 1):
            print(f"      {i:2}. {feat}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Load test data
    print(f"\n2. Loading test data from {test_csv_path}...")
    try:
        df_test = None
        for encoding in ['utf-8', 'windows-1252', 'latin-1']:
            try:
                df_test = pd.read_csv(test_csv_path, encoding=encoding)
                print(f"   ✅ Test records: {len(df_test)} (encoding: {encoding})")
                break
            except:
                continue
        
        if df_test is None:
            print(f"   ❌ Could not read file")
            return
            
    except Exception as e:
        print(f"   ❌ Error loading test data: {e}")
        return
    
    # Check if price column exists
    if price_column_name not in df_test.columns:
        print(f"\n   ❌ ERROR: Price column '{price_column_name}' not found!")
        print(f"   Available columns: {df_test.columns.tolist()}")
        return
    
    # Get actual prices
    actual_prices = df_test[price_column_name].copy()
    valid_prices = actual_prices.dropna()
    
    if len(valid_prices) == 0:
        print(f"   ❌ No valid prices")
        return
    
    print(f"   ✅ Valid prices: {len(valid_prices)}")
    
    # Make predictions
    print(f"\n3. Making predictions...")
    try:
        df_predict = df_test.copy()
        df_predict['Gutter Clearing'] = df_predict[price_column_name]
        
        # CRITICAL FIX: Fill missing roof types with N/A
        roof_cols = ['Roof Type', 'Roof Type/ Material']
        for col in roof_cols:
            if col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna('N/A')
                # Also replace empty strings and 'nan' strings
                df_predict[col] = df_predict[col].replace(['', 'nan', 'NaN', 'none', 'None'], 'N/A')
        
        predicted_prices = predictor.predict(df_predict)
        print(f"   ✅ Predictions made: {len(predicted_prices)}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    valid_mask = ~actual_prices.isna()
    actual_valid = actual_prices[valid_mask].values
    predicted_valid = predicted_prices[valid_mask]
    
    mae = mean_absolute_error(actual_valid, predicted_valid)
    rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
    r2 = r2_score(actual_valid, predicted_valid)
    mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
    
    print(f"\n📊 Mean Absolute Error (MAE):     ${mae:.2f}")
    print(f"📊 Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"📊 R² Score:                       {r2:.4f}")
    print(f"📊 Mean Absolute % Error (MAPE):   {mape:.2f}%")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if mape < 10:
        print("✅ EXCELLENT: Very accurate")
    elif mape < 20:
        print("✅ GOOD: Reasonably accurate")
    elif mape < 30:
        print("⚠️  FAIR: Moderate accuracy")
    else:
        print("❌ POOR: Needs improvement")
    
    print(f"\n💡 Average error: ${mae:.2f} ({mape:.1f}%)")
    
    if r2 < 0.3:
        print(f"❌ Low R² ({r2:.2f})")
    elif r2 < 0.5:
        print(f"⚠️  Moderate R² ({r2:.2f})")
    else:
        print(f"✅ Good R² ({r2:.2f})")
    
    # Price comparison
    print("\n" + "=" * 80)
    print("PRICE COMPARISON")
    print("=" * 80)
    
    print(f"\nActual:    Min=${actual_valid.min():.2f}, Max=${actual_valid.max():.2f}, Mean=${actual_valid.mean():.2f}")
    print(f"Predicted: Min=${predicted_valid.min():.2f}, Max=${predicted_valid.max():.2f}, Mean=${predicted_valid.mean():.2f}")
    
    # Sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (First 20)")
    print("=" * 80)
    
    print(f"\n{'Index':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for i in range(min(20, len(actual_valid))):
        actual = actual_valid[i]
        predicted = predicted_valid[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        
        print(f"{i:<8} ${actual:<9.2f} ${predicted:<9.2f} ${error:<9.2f} {error_pct:<9.1f}%")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("=" * 80)
        print("SERVICE MODEL EVALUATOR")
        print("=" * 80)
        print("\nUsage: python evaluate_service.py <model_file> <test_csv> [price_column]")
        print("\nExamples:")
        print("  python evaluate_service.py gutter_price_model.pkl test.csv 'Gutter Clearing'")
        print("  python evaluate_service.py chemical_spray_model.pkl test.csv 'Chemical Spray'")
        print("=" * 80)
        sys.exit(1)
    
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    price_column = sys.argv[3] if len(sys.argv) > 3 else 'Gutter Clearing'
    
    evaluate_service_model(model_file, test_file, price_column)