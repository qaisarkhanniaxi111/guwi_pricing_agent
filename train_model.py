"""
Train Model from CSV Data
This script loads your existing CSV data and trains
the gutter clearing price prediction model.
"""

import pandas as pd
import numpy as np
import sys
import os
from gutter_price_model import GutterPricePredictor

def load_and_validate_data(csv_path):
    """Load and validate the CSV data"""
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} records from {csv_path}")
        
        # Check for required columns
        required_columns = [
            'Gutter Clearing',  # Target variable (price)
            'Roof Type',
            'Home Square Footage',
            'Home Value',
            'Average Home value in Zip code',
            'Number of Stories'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            print("These columns will need to be added or the model may not work properly.")
        
        # Display data info
        print("\nData Overview:")
        print("-" * 50)
        print(f"Total records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nColumn names found:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Display price statistics
        if 'Gutter Clearing' in df.columns:
            print(f"\nPrice Statistics:")
            print(f"  Min: ${df['Gutter Clearing'].min():.2f}")
            print(f"  Max: ${df['Gutter Clearing'].max():.2f}")
            print(f"  Mean: ${df['Gutter Clearing'].mean():.2f}")
            print(f"  Median: ${df['Gutter Clearing'].median():.2f}")
        
        return df
        
    except FileNotFoundError:
        print(f"ERROR: File '{csv_path}' not found!")
        print("Please make sure your CSV file is in the correct location.")
        return None
    except Exception as e:
        print(f"ERROR loading CSV: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the data"""
    
    # Remove any rows with missing target values
    if 'Gutter Clearing' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['Gutter Clearing'])
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing price values")
    
    # Convert price column to numeric if it's not already
    if 'Gutter Clearing' in df.columns:
        df['Gutter Clearing'] = pd.to_numeric(df['Gutter Clearing'], errors='coerce')
    
    # Handle any other numeric columns
    numeric_columns = [
        'Home Square Footage', 
        'Home Value', 
        'Average Home value in Zip code',
        'Number of Stories'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("GUTTER CLEARING PRICE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Get CSV file path from command line or use default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try common file names
        possible_files = ['gutter_data.csv', 'training_data.csv', 'data.csv']
        csv_path = None
        for file in possible_files:
            if os.path.exists(file):
                csv_path = file
                break
        
        if not csv_path:
            print("\nPlease provide the path to your CSV file.")
            print("Usage: python train_model.py <path_to_csv>")
            print("\nExample: python train_model.py gutter_data.csv")
            csv_path = input("\nEnter the path to your CSV file: ").strip()
    
    # Load the data
    df = load_and_validate_data(csv_path)
    
    if df is None:
        sys.exit(1)
    
    # Clean the data
    df = clean_data(df)
    
    print("\n" + "=" * 50)
    print("Training model...")
    print("=" * 50)
    
    # Initialize and train model
    predictor = GutterPricePredictor()
    predictor.fit(df)
    
    # Save the model
    predictor.save_model('gutter_price_model.pkl')
    
    print("\n" + "=" * 50)
    print("Model training complete!")
    print("=" * 50)
    
    # Display model performance metrics
    if hasattr(predictor, 'model_metrics'):
        print("\nModel Performance Summary:")
        print("-" * 50)
        best_metrics = predictor.model_metrics.get(predictor.best_model_name, {})
        print(f"Best Model: {predictor.best_model_name}")
        print(f"Mean Absolute Error: ${best_metrics.get('MAE', 0):.2f}")
        print(f"R² Score: {best_metrics.get('R2', 0):.3f}")
    
    # Test on a few examples from the dataset
    print("\nValidation on sample records from your data:")
    print("-" * 50)
    
    # Take a few random samples for testing
    if len(df) > 0:
        sample_size = min(3, len(df))
        sample_df = df.sample(n=sample_size)
        
        for idx, row in sample_df.iterrows():
            test_record = df.loc[[idx]].copy()
            actual_price = row['Gutter Clearing']
            
            # Predict using the model
            predicted_price = predictor.predict(test_record)
            
            print(f"\nProperty: {row.get('Address', 'Unknown')}")
            if 'Home Square Footage' in row:
                print(f"  Square Footage: {row['Home Square Footage']}")
            if 'Number of Stories' in row:
                print(f"  Stories: {row['Number of Stories']}")
            if 'Roof Type' in row:
                print(f"  Roof Type: {row['Roof Type']}")
            print(f"  Actual Price: ${actual_price:.2f}")
            print(f"  Predicted Price: ${predicted_price:.2f}")
            print(f"  Difference: ${abs(predicted_price - actual_price):.2f}")
    
    print("\n" + "=" * 60)
    print("MODEL READY FOR API DEPLOYMENT")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt --break-system-packages")
    print("2. Start the API server: python app.py")
    print("3. The API will be available at http://localhost:5000")
    print("4. Use the test_client.py to test the API endpoints")
    print("\nModel file saved as: gutter_price_model.pkl")
    print("\nYour CSV data has been used to train the model successfully!")
