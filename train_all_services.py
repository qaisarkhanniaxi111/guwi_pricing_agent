"""
Universal Multi-Service Price Prediction Trainer
Trains separate models for: Gutter Clearing, Chemical Spray, Zinc Treatment,
Exterior Window Cleaning, Interior Window Cleaning
"""

import pandas as pd
import numpy as np
from gutter_price_model import GutterPricePredictor
import os
import json

# Service configurations
SERVICES = {
    'gutter_clearing': {
        'csv_file': 'train2_fix.csv',
        'price_column': 'Gutter Clearing',
        'model_file': 'gutter_price_model.pkl',
        'description': 'Gutter Clearing Service',
        'required_columns': None  # Uses all standard columns
    },
    'chemical_spray': {
        'csv_file': 'chemicaltrain.csv',
        'price_column': 'Chemical Spray',
        'model_file': 'chemical_spray_model.pkl',
        'description': 'Chemical Spray Service',
        'required_columns': None  # Uses all standard columns
    },
    'zinc_treatment': {
        'csv_file': 'zinctrain.csv',
        'price_column': 'Zinc Treatment',
        'model_file': 'zinc_treatment_model.pkl',
        'description': 'Zinc Treatment Service',
        'required_columns': None  # Uses all standard columns
    },
    'exterior_window': {
        'csv_file': 'exteriorwindowtrain.csv',
        'price_column': 'Exterior Window Cleaning',
        'model_file': 'exterior_window_model.pkl',
        'description': 'Exterior Window Cleaning Service'
    },
    'interior_window': {
        'csv_file': 'interiorwindowtrain.csv',
        'price_column': 'Interior Window Cleaning',
        'model_file': 'interior_window_model.pkl',
        'description': 'Interior Window Cleaning Service',
        'required_columns': None  # Uses all standard columns
    }
}


def train_service_model(service_name, config):
    """Train a model for a specific service"""
    
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {config['description']}")
    print("=" * 80)
    
    csv_file = config['csv_file']
    price_column = config['price_column']
    model_file = config['model_file']
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ ERROR: File not found: {csv_file}")
        print(f"   Skipping {service_name}")
        return None
    
    try:
        # Load data - try different encodings
        print(f"\n1. Loading data from {csv_file}...")
        
        # Try UTF-8 first (standard)
        try:
            df = pd.read_csv(csv_file)
            print(f"   ✅ Loaded {len(df)} records (UTF-8 encoding)")
        except UnicodeDecodeError:
            # Try Windows-1252 encoding (common in Excel)
            try:
                df = pd.read_csv(csv_file, encoding='windows-1252')
                print(f"   ✅ Loaded {len(df)} records (Windows-1252 encoding)")
            except:
                # Try Latin-1 as last resort
                df = pd.read_csv(csv_file, encoding='latin-1')
                print(f"   ✅ Loaded {len(df)} records (Latin-1 encoding)")
        
        # CRITICAL FIX: Check for duplicate roof type columns
        roof_type_cols = [col for col in df.columns if 'Roof' in col and 'Type' in col and 'Details' not in col]
        if len(roof_type_cols) > 1:
            print(f"   ⚠️  Found multiple roof type columns: {roof_type_cols}")
            print(f"   → Keeping 'Roof Type/ Material' and removing others")
            # Keep 'Roof Type/ Material' if it exists, otherwise keep the first one
            if 'Roof Type/ Material' in roof_type_cols:
                keep_col = 'Roof Type/ Material'
            elif 'Roof Type' in roof_type_cols:
                keep_col = 'Roof Type'
            else:
                keep_col = roof_type_cols[0]
            
            # Remove other roof type columns
            for col in roof_type_cols:
                if col != keep_col:
                    df = df.drop(columns=[col])
                    print(f"   ✅ Removed duplicate column: {col}")
        
        # Check if price column exists
        if price_column not in df.columns:
            print(f"   ❌ ERROR: Column '{price_column}' not found!")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
        
        # Detect column configuration
        print(f"\n2. Analyzing column configuration...")
        has_roof_type = 'Roof Type/ Material' in df.columns or 'Roof Type' in df.columns
        has_square_footage = 'Home Square Footage' in df.columns
        has_home_value = 'Home Value' in df.columns
        has_avg_value = 'Average Home Value in Zip code' in df.columns or 'Average Home value in Zip code' in df.columns
        has_stories = 'Number of Stories' in df.columns
        has_steepness = 'Roof Details >> Roof: >> Steepness' in df.columns
        has_ladder_cols = any('Ladder' in col for col in df.columns)
        
        # Core columns check
        core_count = sum([has_roof_type, has_square_footage, has_home_value, has_avg_value, has_stories])
        print(f"   📊 Core columns present: {core_count}/5")
        
        if core_count == 5:
            print(f"   ✅ All 5 core columns found")
        else:
            missing = []
            if not has_roof_type: missing.append('Roof Type')
            if not has_square_footage: missing.append('Home Square Footage')
            if not has_home_value: missing.append('Home Value')
            if not has_avg_value: missing.append('Average Home Value in Zip code')
            if not has_stories: missing.append('Number of Stories')
            print(f"   ⚠️  Missing: {', '.join(missing)}")
        
        # Optional columns check
        optional_count = sum([has_steepness, has_ladder_cols])
        if optional_count > 0:
            print(f"   ℹ️  Optional columns: Steepness={has_steepness}, Ladder info={has_ladder_cols}")
            print(f"   → Training with STANDARD feature set ({core_count + optional_count*5}+ features)")
        else:
            print(f"   ℹ️  No steepness or ladder columns found")
            print(f"   → Training with MINIMAL feature set (~8 features)")
        
        # Rename price column to standard name for model
        df_renamed = df.copy()
        df_renamed['Gutter Clearing'] = df_renamed[price_column]
        
        # CRITICAL FIX: Handle N/A roof types
        roof_col = None
        if 'Roof Type' in df_renamed.columns:
            roof_col = 'Roof Type'
        elif 'Roof Type/ Material' in df_renamed.columns:
            roof_col = 'Roof Type/ Material'
        
        if roof_col:
            # Replace missing/empty with 'Unknown'
            df_renamed[roof_col] = df_renamed[roof_col].fillna('Unknown')
            df_renamed[roof_col] = df_renamed[roof_col].replace(['', 'nan', 'NaN'], 'Unknown')
            
            # Ensure we have some N/A values in training (5% minimum)
            na_count = (df_renamed[roof_col] == 'N/A').sum()
            if na_count < len(df_renamed) * 0.05:
                # Convert 5% of 'Unknown' to 'N/A' to ensure model sees it
                unknown_indices = df_renamed[df_renamed[roof_col] == 'Unknown'].index
                if len(unknown_indices) > 0:
                    num_to_convert = max(int(len(df_renamed) * 0.05), 10)
                    convert_indices = np.random.choice(unknown_indices, 
                                                      size=min(num_to_convert, len(unknown_indices)), 
                                                      replace=False)
                    df_renamed.loc[convert_indices, roof_col] = 'N/A'
                    print(f"   ✅ Added {len(convert_indices)} 'N/A' roof types for robust handling")
            
            # Show roof type distribution
            print(f"\n   📊 Roof Type Distribution:")
            roof_counts = df_renamed[roof_col].value_counts()
            for roof_type, count in roof_counts.head(10).items():
                print(f"      {roof_type:<20} {count:>6} ({count/len(df_renamed)*100:.1f}%)")
        
        # Remove rows with missing prices
        initial_count = len(df_renamed)
        df_renamed = df_renamed.dropna(subset=['Gutter Clearing'])
        removed_count = initial_count - len(df_renamed)
        
        if removed_count > 0:
            print(f"   ⚠️  Removed {removed_count} rows with missing prices")
        
        if len(df_renamed) < 100:
            print(f"   ❌ ERROR: Not enough data! Only {len(df_renamed)} samples")
            print(f"   Need at least 100 samples to train")
            return None
        
        # Price statistics
        prices = df_renamed['Gutter Clearing']
        print(f"\n3. Price Statistics for {config['description']}:")
        print(f"   Min:    ${prices.min():.2f}")
        print(f"   Max:    ${prices.max():.2f}")
        print(f"   Mean:   ${prices.mean():.2f}")
        print(f"   Median: ${prices.median():.2f}")
        print(f"   Std:    ${prices.std():.2f}")
        
        # Train model
        print(f"\n4. Training model...")
        predictor = GutterPricePredictor()
        predictor.fit(df_renamed)
        
        # Show features used
        print(f"   ✅ Model trained with {len(predictor.feature_names)} features")
        print(f"\n   📋 Features used in model:")
        for i, feat in enumerate(predictor.feature_names, 1):
            print(f"      {i:2}. {feat}")
        
        # Save model
        print(f"\n5. Saving model to {model_file}...")
        predictor.save_model(model_file)
        
        # Get best model metrics
        best_metrics = predictor.model_metrics.get(predictor.best_model_name, {})
        
        result = {
            'service': service_name,
            'description': config['description'],
            'model_file': model_file,
            'best_model': predictor.best_model_name,
            'training_samples': len(df_renamed),
            'features_used': len(predictor.feature_names),
            'mae': best_metrics.get('MAE', 0),
            'rmse': best_metrics.get('RMSE', 0),
            'r2': best_metrics.get('R2', 0),
            'cv_mae': best_metrics.get('CV_MAE', 0),
            'price_range': {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean())
            }
        }
        
        print(f"\n✅ SUCCESS: {config['description']} model trained!")
        print(f"   Model: {predictor.best_model_name}")
        print(f"   MAE: ${best_metrics.get('MAE', 0):.2f}")
        print(f"   R²: {best_metrics.get('R2', 0):.4f}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR training {service_name}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_all_services():
    """Train models for all services"""
    
    print("=" * 80)
    print("MULTI-SERVICE MODEL TRAINER")
    print("=" * 80)
    print("\nThis will train separate models for each service:")
    for service, config in SERVICES.items():
        print(f"  • {config['description']}")
    
    input("\nPress Enter to start training...")
    
    results = []
    successful = 0
    failed = 0
    
    for service_name, config in SERVICES.items():
        result = train_service_model(service_name, config)
        
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Successfully trained: {successful} models")
    print(f"❌ Failed: {failed} models")
    
    if results:
        print("\nTrained Models:")
        print("-" * 90)
        print(f"{'Service':<30} {'Model':<20} {'MAE':<12} {'R²':<10} {'Features':<10}")
        print("-" * 90)
        
        for result in results:
            print(f"{result['description']:<30} "
                  f"{result['best_model']:<20} "
                  f"${result['mae']:<11.2f} "
                  f"{result['r2']:<10.4f} "
                  f"{result.get('features_used', 'N/A'):<10}")
    
    # Save summary to JSON
    if results:
        summary_file = 'models_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    if successful > 0:
        print("\n📦 Model Files Created:")
        for result in results:
            print(f"   • {result['model_file']}")
        
        print("\n🚀 Next Steps:")
        print("   1. Test each model with test data")
        print("   2. Update API to support all services")
        print("   3. Deploy models to production")
    
    return results


if __name__ == "__main__":
    results = train_all_services()