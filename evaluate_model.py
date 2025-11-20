"""
Model Evaluation and Debugging Script
This script helps diagnose why predictions might be inaccurate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gutter_price_model import GutterPricePredictor
import joblib

def evaluate_model_performance(model_path, test_csv_path):
    """
    Comprehensive model evaluation
    """
    print("=" * 70)
    print("GUTTER PRICE MODEL EVALUATION")
    print("=" * 70)
    
    # Load the model
    print("\n1. Loading model...")
    predictor = GutterPricePredictor()
    predictor.load_model(model_path)
    print(f"   Model loaded: {predictor.best_model_name}")
    
    # Load test data
    print("\n2. Loading test data...")
    df_test = pd.read_csv(test_csv_path)
    print(f"   Test records: {len(df_test)}")
    
    # Check if we have actual prices
    if 'Gutter Clearing' not in df_test.columns:
        print("   WARNING: No 'Gutter Clearing' column found for comparison!")
        return
    
    # Separate features and actual prices
    actual_prices = df_test['Gutter Clearing'].copy()
    
    # Make predictions
    print("\n3. Making predictions...")
    try:
        predicted_prices = predictor.predict(df_test)
        print(f"   Predictions made: {len(predicted_prices)}")
    except Exception as e:
        print(f"   ERROR making predictions: {e}")
        return
    
    # Calculate metrics
    print("\n4. Performance Metrics:")
    print("   " + "-" * 60)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    r2 = r2_score(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
    print(f"   Mean Absolute Error (MAE):     ${mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"   R² Score:                       {r2:.4f}")
    print(f"   Mean Absolute % Error (MAPE):   {mape:.2f}%")
    
    # Interpretation
    print("\n5. Interpretation:")
    print("   " + "-" * 60)
    if mape < 10:
        print("   ✅ EXCELLENT: Predictions are very accurate")
    elif mape < 20:
        print("   ✅ GOOD: Predictions are reasonably accurate")
    elif mape < 30:
        print("   ⚠️  FAIR: Predictions have moderate accuracy")
    elif mape < 50:
        print("   ❌ POOR: Predictions are not very accurate")
    else:
        print("   ❌ VERY POOR: Model needs significant improvement")
    
    print(f"\n   On average, predictions are off by ${mae:.2f}")
    print(f"   That's about {mape:.1f}% error on average")
    
    # Price range analysis
    print("\n6. Price Range Analysis:")
    print("   " + "-" * 60)
    print(f"   Actual prices:    ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")
    print(f"   Predicted prices: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
    print(f"   Actual mean:      ${actual_prices.mean():.2f}")
    print(f"   Predicted mean:   ${predicted_prices.mean():.2f}")
    
    # Sample comparisons
    print("\n7. Sample Predictions (First 20):")
    print("   " + "-" * 60)
    comparison_df = pd.DataFrame({
        'Actual': actual_prices.head(20),
        'Predicted': predicted_prices[:20],
        'Difference': actual_prices.head(20).values - predicted_prices[:20],
        'Error %': ((actual_prices.head(20).values - predicted_prices[:20]) / actual_prices.head(20).values * 100)
    })
    print(comparison_df.to_string(index=False))
    
    # Identify problematic predictions
    print("\n8. Worst Predictions (Top 10 Errors):")
    print("   " + "-" * 60)
    errors = np.abs(actual_prices.values - predicted_prices)
    worst_indices = np.argsort(errors)[-10:][::-1]
    
    for idx in worst_indices:
        actual = actual_prices.iloc[idx]
        predicted = predicted_prices[idx]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        print(f"   Index {idx}: Actual=${actual:.2f}, Predicted=${predicted:.2f}, Error=${error:.2f} ({error_pct:.1f}%)")
    
    # Feature importance
    print("\n9. Feature Importance:")
    print("   " + "-" * 60)
    if hasattr(predictor.best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': predictor.feature_names,
            'Importance': predictor.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.head(10).to_string(index=False))
        
        # Check for low-importance features
        low_importance = importance_df[importance_df['Importance'] < 0.01]
        if len(low_importance) > 0:
            print(f"\n   ⚠️  {len(low_importance)} features have very low importance (<1%)")
            print("   Consider removing these features to improve the model")
    
    # Distribution analysis
    print("\n10. Error Distribution:")
    print("   " + "-" * 60)
    errors = actual_prices.values - predicted_prices
    
    print(f"   Mean error:        ${np.mean(errors):.2f}")
    print(f"   Std deviation:     ${np.std(errors):.2f}")
    print(f"   Min error:         ${np.min(errors):.2f} (over-predicted)")
    print(f"   Max error:         ${np.max(errors):.2f} (under-predicted)")
    
    # Check for bias
    if np.mean(errors) > 10:
        print("\n   ⚠️  MODEL IS UNDER-PREDICTING (predictions too low)")
    elif np.mean(errors) < -10:
        print("\n   ⚠️  MODEL IS OVER-PREDICTING (predictions too high)")
    else:
        print("\n   ✅ No significant bias detected")
    
    # Training data statistics
    print("\n11. Model Training Info:")
    print("   " + "-" * 60)
    if hasattr(predictor, 'model_metrics'):
        for model_name, metrics in predictor.model_metrics.items():
            if model_name == predictor.best_model_name:
                print(f"   Best Model: {model_name}")
                print(f"   Training MAE:     ${metrics.get('MAE', 0):.2f}")
                print(f"   Training RMSE:    ${metrics.get('RMSE', 0):.2f}")
                print(f"   Training R²:      {metrics.get('R2', 0):.4f}")
                print(f"   Cross-Val MAE:    ${metrics.get('CV_MAE', 0):.2f}")
    
    # Data quality checks
    print("\n12. Data Quality Checks:")
    print("   " + "-" * 60)
    
    # Check for missing values
    missing = df_test.isnull().sum()
    if missing.sum() > 0:
        print("   ⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count} missing")
    else:
        print("   ✅ No missing values")
    
    # Check price range
    if actual_prices.min() < 50 or actual_prices.max() > 2000:
        print("   ⚠️  Unusual price range detected")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print("=" * 70)
    
    recommendations = []
    
    if mape > 30:
        recommendations.append("❌ Model accuracy is poor. Consider:")
        recommendations.append("   • Collecting more training data")
        recommendations.append("   • Adding more relevant features")
        recommendations.append("   • Checking for data quality issues")
    
    if r2 < 0.3:
        recommendations.append("❌ Low R² score indicates weak explanatory power:")
        recommendations.append("   • Current features may not predict price well")
        recommendations.append("   • Consider adding: location features, season, complexity metrics")
    
    if abs(np.mean(errors)) > 20:
        recommendations.append("❌ Model has significant bias:")
        recommendations.append("   • Re-train with more balanced data")
        recommendations.append("   • Check for outliers in training data")
    
    if len(df_test) < 100:
        recommendations.append("⚠️  Small test set - results may not be reliable")
        recommendations.append("   • Use larger test set (at least 200+ samples)")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ Model performance looks reasonable!")
        print("   For further improvement:")
        print("   • Collect more training data")
        print("   • Fine-tune hyperparameters")
        print("   • Try ensemble methods")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    # Return results for further analysis
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'actual': actual_prices,
        'predicted': predicted_prices,
        'errors': errors
    }


def plot_predictions(actual, predicted, save_path='prediction_analysis.png'):
    """
    Create visualization plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted scatter
        axes[0, 0].scatter(actual, predicted, alpha=0.5)
        axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = actual.values - predicted
        axes[0, 1].scatter(predicted, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        pct_errors = ((actual.values - predicted) / actual.values) * 100
        axes[1, 1].hist(pct_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Percentage Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping visualization")


if __name__ == "__main__":
    # Run evaluation
    print("Starting model evaluation...")
    print("\nUsage: python evaluate_model.py")
    print("\nMake sure you have:")
    print("  1. gutter_price_model.pkl (trained model)")
    print("  2. test data CSV with 'Gutter Clearing' column")
    print()
    
    # Example usage
    model_path = 'gutter_price_model.pkl'
    test_data_path = 'train2.csv'  # Replace with your test file
    
    results = evaluate_model_performance(model_path, test_data_path)
    
    if results:
        # Create visualization
        plot_predictions(results['actual'], results['predicted'])
