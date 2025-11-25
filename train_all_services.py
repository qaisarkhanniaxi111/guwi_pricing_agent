"""
Universal Multi-Service Price Prediction Trainer
Trains separate models for: Gutter Clearing, Chemical Spray, Zinc Treatment,
Exterior Window Cleaning, Interior Window Cleaning
"""

import pandas as pd
import numpy as np
from gutter_price_model import GutterPricePredictor
from flexible_trainer import FlexibleServicePredictor, train_service_model_flexible
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
        'description': 'Exterior Window Cleaning Service',
        'required_columns': [
            'Address', 'City', 'State', 'Zip',
            'Home Square Footage', 'Home Value', 
            'Average Home Value in Zip code', 'Number of Stories',
            'Jobsite Ladders >>Gutter>>Number of Ladder Movements',
            'Jobsite Ladders >> Gutter >> ladder Size',
            'Jobsite Ladders >> Window >> Number of Ladder Movements',
            'Jobsite Ladders >> Window >> ladder Size'
        ]
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
        
        # Check if price column exists
        if price_column not in df.columns:
            print(f"   ❌ ERROR: Column '{price_column}' not found!")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
        
        # Rename price column to standard name for model
        df_renamed = df.copy()
        df_renamed['Gutter Clearing'] = df_renamed[price_column]
        
        # If service has specific required columns, keep only those
        required_cols = config.get('required_columns', None)
        if required_cols is not None:
            print(f"   ℹ️  Service uses custom column set ({len(required_cols)} columns)")
            
            # Keep only the required columns that exist
            available_required = [col for col in required_cols if col in df_renamed.columns]
            missing_required = [col for col in required_cols if col not in df_renamed.columns]
            
            if missing_required:
                print(f"   ⚠️  Some required columns missing: {missing_required}")
            
            # Keep required columns + the price column
            columns_to_keep = available_required + ['Gutter Clearing']
            df_renamed = df_renamed[columns_to_keep]
            
            print(f"   ✅ Using {len(available_required)} feature columns")
        else:
            print(f"   ℹ️  Service uses all available columns")
        
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
        print(f"\n2. Price Statistics for {config['description']}:")
        print(f"   Min:    ${prices.min():.2f}")
        print(f"   Max:    ${prices.max():.2f}")
        print(f"   Mean:   ${prices.mean():.2f}")
        print(f"   Median: ${prices.median():.2f}")
        print(f"   Std:    ${prices.std():.2f}")
        
        # Train model
        print(f"\n3. Training model...")
        predictor = GutterPricePredictor()
        predictor.fit(df_renamed)
        
        # Save model
        print(f"\n4. Saving model to {model_file}...")
        predictor.save_model(model_file)
        
        # Get best model metrics
        best_metrics = predictor.model_metrics.get(predictor.best_model_name, {})
        
        result = {
            'service': service_name,
            'description': config['description'],
            'model_file': model_file,
            'best_model': predictor.best_model_name,
            'training_samples': len(df_renamed),
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
        # Use flexible trainer for services with custom columns
        if config.get('required_columns') is not None:
            print(f"\n[Using Flexible Trainer for {service_name}]")
            result = train_service_model_flexible(service_name, config)
        else:
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
        print("-" * 80)
        print(f"{'Service':<30} {'Model':<20} {'MAE':<12} {'R²':<10} {'Samples':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['description']:<30} "
                  f"{result['best_model']:<20} "
                  f"${result['mae']:<11.2f} "
                  f"{result['r2']:<10.4f} "
                  f"{result['training_samples']:<10}")
    
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