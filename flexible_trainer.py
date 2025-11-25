"""
Service-Specific Model Trainer
Handles different column configurations for different services
"""

import pandas as pd
import numpy as np
import os
import json
from gutter_price_model import GutterPricePredictor


class FlexibleServicePredictor(GutterPricePredictor):
    """
    Extended predictor that adapts to different column configurations
    """
    
    def __init__(self, service_name=None):
        super().__init__()
        self.service_name = service_name
    
    def preprocess_data(self, df):
        """
        Flexible preprocessing that works with whatever columns are available
        """
        # Create a copy to avoid modifying original
        df_processed = self.normalize_columns(df)
        
        # Convert numeric columns to proper numeric types
        numeric_columns = [
            'Home Square Footage', 'Home Value', 
            'Average Home Value in Zip code', 'Average Home value in Zip code',
            'Number of Stories'
        ]
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle categorical variables if available
        if 'Roof Type' in df_processed.columns or 'Roof Type/ Material' in df_processed.columns:
            roof_col = 'Roof Type' if 'Roof Type' in df_processed.columns else 'Roof Type/ Material'
            if roof_col not in self.label_encoders:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoders[roof_col] = LabelEncoder()
                df_processed['Roof Type Encoded'] = self.label_encoders[roof_col].fit_transform(
                    df_processed[roof_col].astype(str).fillna('Unknown')
                )
            else:
                df_processed['Roof Type Encoded'] = self.label_encoders[roof_col].transform(
                    df_processed[roof_col].astype(str).fillna('Unknown')
                )
        
        # Extract numeric values from steepness if available
        if 'Roof Details >> Roof: >> Steepness' in df_processed.columns:
            df_processed['Roof Steepness Ratio'] = df_processed['Roof Details >> Roof: >> Steepness'].apply(
                self.parse_roof_steepness
            )
        
        # Process ladder movements - check for all possible variations
        ladder_movement_patterns = [
            'Jobsite Ladders >> Roof >> Number of Ladder Movements',
            'Jobsite Ladders >> Gutter >> Number of Ladder Movements',
            'Jobsite Ladders >>Gutter>>Number of Ladder Movements',  # No spaces version
            'Jobsite Ladders >> Window >> Number of Ladder Movements'
        ]
        
        for pattern in ladder_movement_patterns:
            if pattern in df_processed.columns:
                new_col = pattern.replace('Number of Ladder Movements', 'Movements Numeric').replace('>>', '_')
                df_processed[new_col] = df_processed[pattern].apply(self.parse_ladder_movements)
        
        # Extract ladder sizes - check for all possible variations
        ladder_size_patterns = [
            'Jobsite Ladders >> Roof >> ladder Size',
            'Jobsite Ladders >> Gutter >> ladder Size',
            'Jobsite Ladders >>Gutter>>ladder Size',  # No spaces version
            'Jobsite Ladders >> Window >> ladder Size'
        ]
        
        for pattern in ladder_size_patterns:
            if pattern in df_processed.columns:
                new_col = pattern.replace('ladder Size', 'Ladder Size Numeric').replace('>>', '_')
                df_processed[new_col] = df_processed[pattern].apply(self.parse_ladder_size)
        
        # Feature engineering - handle missing columns gracefully
        if 'Home Value' in df_processed.columns:
            home_value_col = 'Average Home Value in Zip code' if 'Average Home Value in Zip code' in df_processed.columns else 'Average Home value in Zip code'
            if home_value_col in df_processed.columns:
                df_processed['Home Value'] = df_processed['Home Value'].fillna(df_processed[home_value_col])
                df_processed[home_value_col] = df_processed[home_value_col].fillna(df_processed['Home Value'])
                df_processed['Home Value to Area Ratio'] = df_processed['Home Value'] / df_processed[home_value_col].replace(0, 1)
            else:
                df_processed['Home Value to Area Ratio'] = 1.0
        else:
            df_processed['Home Value to Area Ratio'] = 1.0
        
        if 'Home Square Footage' in df_processed.columns and 'Number of Stories' in df_processed.columns:
            df_processed['Home Square Footage'] = df_processed['Home Square Footage'].fillna(1500)
            df_processed['Number of Stories'] = df_processed['Number of Stories'].fillna(1)
            df_processed['Size per Story'] = df_processed['Home Square Footage'] / df_processed['Number of Stories'].replace(0, 1)
        else:
            df_processed['Size per Story'] = 0
        
        df_processed['Total Ladder Movements'] = 0
        df_processed['Max Ladder Size'] = 0
        df_processed['Difficulty Score'] = 0
        
        # Calculate total movements and max ladder size
        movement_cols = [col for col in df_processed.columns if 'Movements Numeric' in col or 'Movements_Numeric' in col]
        if movement_cols:
            df_processed['Total Ladder Movements'] = df_processed[movement_cols].sum(axis=1)
        
        size_cols = [col for col in df_processed.columns if 'Ladder Size Numeric' in col or 'Ladder_Size_Numeric' in col]
        if size_cols:
            df_processed['Max Ladder Size'] = df_processed[size_cols].max(axis=1)
        
        # Create difficulty score
        num_stories = df_processed['Number of Stories'].fillna(1) if 'Number of Stories' in df_processed.columns else 1
        roof_steepness = df_processed['Roof Steepness Ratio'].fillna(0.5) if 'Roof Steepness Ratio' in df_processed.columns else 0.5
        
        df_processed['Difficulty Score'] = (
            num_stories * 10 +
            roof_steepness * 15 +
            df_processed['Total Ladder Movements'] * 0.5 +
            df_processed['Max Ladder Size'] * 0.3
        )
        
        # Select features for model - only use what's available
        feature_columns = []
        
        # Core features
        possible_features = [
            'Home Square Footage',
            'Home Value',
            'Average Home value in Zip code',
            'Average Home Value in Zip code',
            'Number of Stories'
        ]
        
        for feat in possible_features:
            if feat in df_processed.columns:
                feature_columns.append(feat)
        
        # Engineered features (always include if we created them)
        feature_columns.extend([
            'Home Value to Area Ratio',
            'Size per Story',
            'Total Ladder Movements',
            'Max Ladder Size',
            'Difficulty Score'
        ])
        
        # Optional features
        if 'Roof Type Encoded' in df_processed.columns:
            feature_columns.append('Roof Type Encoded')
        if 'Roof Steepness Ratio' in df_processed.columns:
            feature_columns.append('Roof Steepness Ratio')
        
        # Add individual movement and size columns
        feature_columns.extend(movement_cols)
        feature_columns.extend(size_cols)
        
        # Keep only available columns
        available_features = [col for col in feature_columns if col in df_processed.columns]
        
        # Fill any remaining NaN values
        result_df = df_processed[available_features].fillna(0)
        
        return result_df


# Update the train_service_model function to use flexible predictor
def train_service_model_flexible(service_name, config):
    """Train a model for a specific service using flexible predictor"""
    
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
        
        df = None
        for encoding in ['utf-8', 'windows-1252', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"   ✅ Loaded {len(df)} records ({encoding} encoding)")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"   ❌ ERROR: Could not read file with any encoding")
            return None
        
        # Check if price column exists
        if price_column not in df.columns:
            print(f"   ❌ ERROR: Column '{price_column}' not found!")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
        
        # Show available columns
        print(f"\n2. Available columns in {csv_file}:")
        for i, col in enumerate(df.columns, 1):
            if col == price_column:
                print(f"   {i:2}. {col} ⭐ (price column)")
            else:
                print(f"   {i:2}. {col}")
        
        # Rename price column to standard name
        df_renamed = df.copy()
        df_renamed['Gutter Clearing'] = df_renamed[price_column]
        
        # Remove rows with missing prices
        initial_count = len(df_renamed)
        df_renamed = df_renamed.dropna(subset=['Gutter Clearing'])
        removed_count = initial_count - len(df_renamed)
        
        if removed_count > 0:
            print(f"\n   ⚠️  Removed {removed_count} rows with missing prices")
        
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
        
        # Train model using flexible predictor
        print(f"\n4. Training model with available features...")
        predictor = FlexibleServicePredictor(service_name=service_name)
        predictor.fit(df_renamed)
        
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
        print(f"   Features: {len(predictor.feature_names)}")
        print(f"   MAE: ${best_metrics.get('MAE', 0):.2f}")
        print(f"   R²: {best_metrics.get('R2', 0):.4f}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR training {service_name}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import os
    
    # Example: test with exterior window
    print("Testing flexible predictor with exterior window service...")
    
    config = {
        'csv_file': 'exteriorwindowtrain.csv',
        'price_column': 'Exterior Window Cleaning',
        'model_file': 'exterior_window_model.pkl',
        'description': 'Exterior Window Cleaning Service'
    }
    
    result = train_service_model_flexible('exterior_window', config)
    
    if result:
        print("\n✅ Test successful!")
    else:
        print("\n❌ Test failed!")