"""
Gutter Clearing Price Prediction Model
This script trains multiple ML models to predict gutter clearing prices
based on property characteristics and selects the best performer.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class GutterPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.model_metrics = {}
        self.feature_names = None
        self.label_encoders = {}
        self.column_mapping = {}
    
    def normalize_columns(self, df):
        """Normalize column names to handle variations"""
        df = df.copy()
        
        # Define column name mappings (variations -> standard name)
        mappings = {
            'Roof Type/ Material': 'Roof Type',
            'Roof Type/Material': 'Roof Type',
            'Average Home Value in Zip code': 'Average Home value in Zip code',
            'Average Home value in zip code': 'Average Home value in Zip code',
            'average home value in zip code': 'Average Home value in Zip code',
        }
        
        # Apply mappings
        for old_name, new_name in mappings.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                self.column_mapping[old_name] = new_name
        
        return df
    
    def parse_roof_steepness(self, x):
        """
        Parse roof steepness values like:
        - '4/12' -> 0.333
        - '4 / 12' -> 0.333
        - '4/12 or less' -> 0.333
        - 'Steeper than 12/12' -> 1.0
        - etc.
        """
        if pd.isna(x):
            return 0.5  # default value
        
        x_str = str(x).strip()
        
        # Try to find a fraction pattern (digits/digits)
        match = re.search(r'(\d+)\s*/\s*(\d+)', x_str)
        
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator != 0:
                return numerator / denominator
        
        return 0.5  # default if no fraction found
        
    def preprocess_data(self, df):
        """Preprocess the data and create features"""
        # Normalize column names first
        df_processed = self.normalize_columns(df)
        
        # Convert numeric columns to proper numeric types
        numeric_columns = [
            'Home Square Footage', 'Home Value', 'Average Home value in Zip code', 'Number of Stories'
        ]
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle categorical variables
        if 'Roof Type' in df_processed.columns:
            if 'Roof Type' not in self.label_encoders:
                self.label_encoders['Roof Type'] = LabelEncoder()
                df_processed['Roof Type Encoded'] = self.label_encoders['Roof Type'].fit_transform(df_processed['Roof Type'].astype(str))
            else:
                df_processed['Roof Type Encoded'] = self.label_encoders['Roof Type'].transform(df_processed['Roof Type'].astype(str))
        
        # Extract numeric values from steepness using the new parser
        if 'Roof Details >> Roof: >> Steepness' in df_processed.columns:
            df_processed['Roof Steepness Ratio'] = df_processed['Roof Details >> Roof: >> Steepness'].apply(
                self.parse_roof_steepness
            )
        
        # Process ladder movements (extract numeric values from ranges)
        ladder_movement_cols = [
            'Jobsite Ladders >> Roof >> Number of Ladder Movements',
            'Jobsite Ladders >> Gutter >> Number of Ladder Movements',
            'Jobsite Ladders >> Window >> Number of Ladder Movements'
        ]
        
        for col in ladder_movement_cols:
            if col in df_processed.columns:
                new_col = col.replace('Number of Ladder Movements', 'Movements Numeric')
                df_processed[new_col] = df_processed[col].apply(self.parse_ladder_movements)
        
        # Extract ladder sizes
        ladder_size_cols = [
            'Jobsite Ladders >> Roof >> ladder Size',
            'Jobsite Ladders >> Gutter >> ladder Size',
            'Jobsite Ladders >> Window >> ladder Size'
        ]
        
        for col in ladder_size_cols:
            if col in df_processed.columns:
                new_col = col.replace('ladder Size', 'Ladder Size Numeric')
                df_processed[new_col] = df_processed[col].apply(self.parse_ladder_size)
        
        # Feature engineering - handle missing columns gracefully
        if 'Home Value' in df_processed.columns and 'Average Home value in Zip code' in df_processed.columns:
            # Fill NaN values before division
            df_processed['Home Value'] = df_processed['Home Value'].fillna(0)
            df_processed['Average Home value in Zip code'] = df_processed['Average Home value in Zip code'].fillna(1)
            # Avoid division by zero
            df_processed['Home Value to Area Ratio'] = df_processed['Home Value'] / df_processed['Average Home value in Zip code'].replace(0, 1)
        else:
            df_processed['Home Value to Area Ratio'] = 1.0
        
        if 'Home Square Footage' in df_processed.columns and 'Number of Stories' in df_processed.columns:
            df_processed['Home Square Footage'] = df_processed['Home Square Footage'].fillna(0)
            df_processed['Number of Stories'] = df_processed['Number of Stories'].fillna(1)
            # Avoid division by zero
            df_processed['Size per Story'] = df_processed['Home Square Footage'] / df_processed['Number of Stories'].replace(0, 1)
        else:
            df_processed['Size per Story'] = 0
            
        df_processed['Total Ladder Movements'] = 0
        df_processed['Max Ladder Size'] = 0
        df_processed['Difficulty Score'] = 0
        
        # Calculate total movements and max ladder size
        movement_cols = [col for col in df_processed.columns if 'Movements Numeric' in col]
        if movement_cols:
            df_processed['Total Ladder Movements'] = df_processed[movement_cols].sum(axis=1)
        
        size_cols = [col for col in df_processed.columns if 'Ladder Size Numeric' in col]
        if size_cols:
            df_processed['Max Ladder Size'] = df_processed[size_cols].max(axis=1)
        
        # Create difficulty score - use fillna for safety
        df_processed['Difficulty Score'] = (
            df_processed.get('Number of Stories', pd.Series([1]*len(df_processed))).fillna(1) * 10 +
            df_processed.get('Roof Steepness Ratio', pd.Series([0.5]*len(df_processed))).fillna(0.5) * 15 +
            df_processed.get('Total Ladder Movements', pd.Series([0]*len(df_processed))).fillna(0) * 0.5 +
            df_processed.get('Max Ladder Size', pd.Series([20]*len(df_processed))).fillna(20) * 0.3
        )
        
        # Select features for model
        feature_columns = []
        
        # Add core features if available
        if 'Home Square Footage' in df_processed.columns:
            feature_columns.append('Home Square Footage')
        if 'Home Value' in df_processed.columns:
            feature_columns.append('Home Value')
        if 'Average Home value in Zip code' in df_processed.columns:
            feature_columns.append('Average Home value in Zip code')
        if 'Number of Stories' in df_processed.columns:
            feature_columns.append('Number of Stories')
            
        # Add engineered features
        feature_columns.extend(['Home Value to Area Ratio', 'Size per Story',
                               'Total Ladder Movements', 'Max Ladder Size', 'Difficulty Score'])
        
        # Add encoded categorical features if available
        if 'Roof Type Encoded' in df_processed.columns:
            feature_columns.append('Roof Type Encoded')
        if 'Roof Steepness Ratio' in df_processed.columns:
            feature_columns.append('Roof Steepness Ratio')
        
        # Add individual movement columns if available
        for col in movement_cols:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        # Add individual ladder size columns if available
        for col in size_cols:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        # Keep only available columns and fill any remaining NaN values
        available_features = [col for col in feature_columns if col in df_processed.columns]
        result_df = df_processed[available_features].fillna(0)
        
        return result_df
    
    def parse_ladder_movements(self, value):
        """Parse ladder movement ranges to numeric values"""
        if pd.isna(value):
            return 5
        
        value = str(value).strip()
        
        if '+' in value:
            # Handle "20+" format
            return float(value.replace('+', '')) + 5
        elif '-' in value:
            # Handle "4 - 8" format
            parts = value.split('-')
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return (low + high) / 2
                except:
                    return 10
        else:
            # Try to parse as single number
            try:
                return float(value)
            except:
                return 10
    
    def parse_ladder_size(self, value):
        """
        Parse ladder size values like:
        - '20ft' -> 20.0
        - '4 - 8' -> 6.0 (average)
        - '32 ft' -> 32.0
        - '20+' -> 25.0
        """
        if pd.isna(value):
            return 20  # default value
        
        value_str = str(value).strip()
        
        # Remove 'ft' and extra spaces
        value_str = value_str.replace('ft', '').replace('  ', ' ').strip()
        
        # Handle range like "4 - 8"
        if '-' in value_str:
            parts = value_str.split('-')
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return (low + high) / 2
                except:
                    return 20
        
        # Handle "20+" format
        if '+' in value_str:
            try:
                return float(value_str.replace('+', '').strip()) + 5
            except:
                return 20
        
        # Try to parse as single number
        try:
            return float(value_str)
        except:
            return 20
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            ),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.001),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        }
        
        best_score = float('-inf')
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                
                # Store metrics
                self.model_metrics[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'CV_MAE': cv_mae
                }
                
                # Select best model based on MAE
                if mae < best_score or best_score == float('-inf'):
                    best_score = mae
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"{name}:")
                print(f"  MAE: ${mae:.2f}")
                print(f"  RMSE: ${rmse:.2f}")
                print(f"  R²: {r2:.3f}")
                print(f"  Cross-Val MAE: ${cv_mae:.2f}")
                print()
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Fine-tune the best model's hyperparameters"""
        
        if isinstance(self.best_model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(self.best_model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        elif isinstance(self.best_model, xgb.XGBRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        else:
            print("Skipping hyperparameter tuning for this model type")
            return
        
        print(f"Tuning hyperparameters for {self.best_model_name}...")
        
        grid_search = GridSearchCV(
            self.best_model.__class__(),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: ${-grid_search.best_score_:.2f}")
    
    def fit(self, df):
        """Main training pipeline"""
        
        # Separate features and target
        X = self.preprocess_data(df)
        y = df['Gutter Clearing']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("Training multiple models...")
        print("=" * 50)
        self.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        print("=" * 50)
        print(f"Best Model: {self.best_model_name}")
        print("=" * 50)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train_scaled, y_train)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return self
    
    def predict(self, df):
        """Predict gutter clearing price for new data"""
        X = self.preprocess_data(df)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.best_model.predict(X_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def save_model(self, filepath='gutter_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'model_name': self.best_model_name,
            'metrics': self.model_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='gutter_price_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        self.best_model_name = model_data['model_name']
        self.model_metrics = model_data['metrics']
        return self


# Example usage and testing
if __name__ == "__main__":
    # Create sample training data
    sample_data = {
        'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr',
                    '987 Cedar Ln', '147 Birch Way', '258 Spruce Ct', '369 Willow Pl', '741 Ash Blvd'],
        'City': ['Austin', 'Dallas', 'Houston', 'Austin', 'Dallas',
                 'Houston', 'Austin', 'Dallas', 'Houston', 'Austin'],
        'State': ['TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX'],
        'Zip': ['78701', '75201', '77001', '78702', '75202',
                '77002', '78703', '75203', '77003', '78704'],
        'Gutter Clearing': [215, 285, 195, 245, 325, 175, 265, 295, 225, 235],
        'Roof Type': ['Composition', 'Tile', 'Composition', 'Metal', 'Tile',
                      'Composition', 'Metal', 'Tile', 'Composition', 'Metal'],
        'Home Square Footage': [1300, 2200, 1100, 1500, 2800, 900, 1800, 2400, 1400, 1600],
        'Home Value': [1295, 2100, 950, 1400, 2500, 850, 1650, 2200, 1200, 1450],
        'Average Home value in Zip code': [1717, 1950, 1200, 1650, 2100, 1100, 1800, 2000, 1350, 1700],
        'Number of Stories': [1, 2, 1, 1, 2, 1, 2, 2, 1, 1],
        'Roof Details >> Roof: >> Steepness': ['10 / 12', '8 / 12', '6 / 12', '12 / 12', '10 / 12',
                                                '4 / 12', '9 / 12', '11 / 12', '7 / 12', '8 / 12'],
        'Jobsite Ladders >> Roof >> Number of Ladder Movements': ['4 - 8', '9 - 20', '4 - 8', '4 - 8', '20+',
                                                                   '1 - 3', '9 - 20', '20+', '4 - 8', '9 - 20'],
        'Jobsite Ladders >> Roof >> ladder Size': ['40ft', '40ft', '28ft', '32ft', '40ft',
                                                    '24ft', '40ft', '40ft', '32ft', '36ft'],
        'Jobsite Ladders >> Gutter >> Number of Ladder Movements': ['9 - 20', '20+', '4 - 8', '9 - 20', '20+',
                                                                      '4 - 8', '9 - 20', '20+', '9 - 20', '9 - 20'],
        'Jobsite Ladders >> Gutter >> ladder Size': ['40ft', '40ft', '28ft', '36ft', '40ft',
                                                      '24ft', '40ft', '40ft', '32ft', '36ft'],
        'Jobsite Ladders >> Window >> Number of Ladder Movements': ['20+', '20+', '9 - 20', '20+', '20+',
                                                                      '4 - 8', '20+', '20+', '9 - 20', '20+'],
        'Jobsite Ladders >> Window >> ladder Size': ['40ft', '40ft', '32ft', '40ft', '40ft',
                                                      '28ft', '40ft', '40ft', '36ft', '40ft']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and train model
    predictor = GutterPricePredictor()
    predictor.fit(df)
    
    # Save the model
    predictor.save_model('gutter_price_model.pkl')
    
    # Test prediction on new data
    print("\n" + "=" * 50)
    print("Testing prediction on new property:")
    print("=" * 50)
    
    new_property = pd.DataFrame({
        'Address': ['999 Test St'],
        'City': ['Austin'],
        'State': ['TX'],
        'Zip': ['78705'],
        'Roof Type': ['Composition'],
        'Home Square Footage': [1500],
        'Home Value': [1400],
        'Average Home value in Zip code': [1600],
        'Number of Stories': [2],
        'Roof Details >> Roof: >> Steepness': ['8 / 12'],
        'Jobsite Ladders >> Roof >> Number of Ladder Movements': ['9 - 20'],
        'Jobsite Ladders >> Roof >> ladder Size': ['40ft'],
        'Jobsite Ladders >> Gutter >> Number of Ladder Movements': ['9 - 20'],
        'Jobsite Ladders >> Gutter >> ladder Size': ['40ft'],
        'Jobsite Ladders >> Window >> Number of Ladder Movements': ['20+'],
        'Jobsite Ladders >> Window >> ladder Size': ['40ft']
    })
    
    predicted_price = predictor.predict(new_property)
    print(f"Predicted Gutter Clearing Price: ${predicted_price:.2f}")