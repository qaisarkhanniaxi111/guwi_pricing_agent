"""
Flask API for Gutter Clearing Price Prediction
This API provides endpoints for predicting gutter clearing prices
and getting model information.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import os
import sys

# Add the current directory to path to import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gutter_price_model import GutterPricePredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for the model
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = GutterPricePredictor()
        model.load_model('gutter_price_model.pkl')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Gutter Clearing Price Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Predict gutter clearing price (POST)',
            '/predict_batch': 'Predict prices for multiple properties (POST)',
            '/model_info': 'Get model information',
            '/features': 'Get required features list'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get the list of required features"""
    required_features = {
        'required': [
            'Roof Type',
            'Home Square Footage',
            'Home Value',
            'Average Home value in Zip code',
            'Number of Stories'
        ],
        'optional': [
            'Address',
            'City',
            'State',
            'Zip',
            'Roof Details >> Roof: >> Steepness',
            'Jobsite Ladders >> Roof >> Number of Ladder Movements',
            'Jobsite Ladders >> Roof >> ladder Size',
            'Jobsite Ladders >> Gutter >> Number of Ladder Movements',
            'Jobsite Ladders >> Gutter >> ladder Size',
            'Jobsite Ladders >> Window >> Number of Ladder Movements',
            'Jobsite Ladders >> Window >> ladder Size'
        ],
        'roof_types': ['Composition', 'Tile', 'Metal', 'Slate', 'Wood'],
        'example_steepness': '10 / 12',
        'example_ladder_movements': '4 - 8 or 20+',
        'example_ladder_size': '40ft'
    }
    return jsonify(required_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict gutter clearing price for a single property"""
    try:
        # Get JSON data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Roof Type',
            'Home Square Footage',
            'Home Value',
            'Average Home value in Zip code',
            'Number of Stories'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Set default values for optional fields
        defaults = {
            'Address': 'Unknown',
            'City': 'Unknown',
            'State': 'Unknown',
            'Zip': '00000',
            'Roof Details >> Roof: >> Steepness': '6 / 12',
            'Jobsite Ladders >> Roof >> Number of Ladder Movements': '4 - 8',
            'Jobsite Ladders >> Roof >> ladder Size': '32ft',
            'Jobsite Ladders >> Gutter >> Number of Ladder Movements': '9 - 20',
            'Jobsite Ladders >> Gutter >> ladder Size': '32ft',
            'Jobsite Ladders >> Window >> Number of Ladder Movements': '9 - 20',
            'Jobsite Ladders >> Window >> ladder Size': '32ft'
        }
        
        # Apply defaults for missing optional fields
        for field, default_value in defaults.items():
            if field not in data:
                data[field] = default_value
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        predicted_price = model.predict(df)
        
        # Calculate confidence interval (simplified approach)
        # In production, you might want to use more sophisticated methods
        price_std = predicted_price * 0.1  # 10% standard deviation
        confidence_interval = {
            'low': max(predicted_price - 1.96 * price_std, 50),  # Minimum $50
            'high': predicted_price + 1.96 * price_std
        }
        
        # Prepare response
        response = {
            'predicted_price': float(predicted_price),
            'formatted_price': f"${predicted_price:.2f}",
            'confidence_interval': {
                'low': float(confidence_interval['low']),
                'high': float(confidence_interval['high']),
                'formatted': f"${confidence_interval['low']:.2f} - ${confidence_interval['high']:.2f}"
            },
            'factors': {
                'home_size': data['Home Square Footage'],
                'stories': data['Number of Stories'],
                'roof_type': data['Roof Type'],
                'area_value_ratio': float(data['Home Value'] / data['Average Home value in Zip code'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction
        logger.info(f"Prediction made: ${predicted_price:.2f} for {data.get('Address', 'Unknown')}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict gutter clearing prices for multiple properties"""
    try:
        # Get JSON data from request
        data = request.json
        
        if not data or 'properties' not in data:
            return jsonify({'error': 'No properties data provided'}), 400
        
        properties = data['properties']
        
        if not isinstance(properties, list):
            return jsonify({'error': 'Properties must be a list'}), 400
        
        results = []
        
        for idx, property_data in enumerate(properties):
            try:
                # Validate required fields
                required_fields = [
                    'Roof Type',
                    'Home Square Footage',
                    'Home Value',
                    'Average Home value in Zip code',
                    'Number of Stories'
                ]
                
                missing_fields = [field for field in required_fields if field not in property_data]
                if missing_fields:
                    results.append({
                        'index': idx,
                        'error': f'Missing required fields: {missing_fields}',
                        'address': property_data.get('Address', 'Unknown')
                    })
                    continue
                
                # Set default values for optional fields
                defaults = {
                    'Address': f'Property {idx + 1}',
                    'City': 'Unknown',
                    'State': 'Unknown',
                    'Zip': '00000',
                    'Roof Details >> Roof: >> Steepness': '6 / 12',
                    'Jobsite Ladders >> Roof >> Number of Ladder Movements': '4 - 8',
                    'Jobsite Ladders >> Roof >> ladder Size': '32ft',
                    'Jobsite Ladders >> Gutter >> Number of Ladder Movements': '9 - 20',
                    'Jobsite Ladders >> Gutter >> ladder Size': '32ft',
                    'Jobsite Ladders >> Window >> Number of Ladder Movements': '9 - 20',
                    'Jobsite Ladders >> Window >> ladder Size': '32ft'
                }
                
                # Apply defaults for missing optional fields
                for field, default_value in defaults.items():
                    if field not in property_data:
                        property_data[field] = default_value
                
                # Create DataFrame
                df = pd.DataFrame([property_data])
                
                # Make prediction
                predicted_price = model.predict(df)
                
                results.append({
                    'index': idx,
                    'address': property_data.get('Address', f'Property {idx + 1}'),
                    'predicted_price': float(predicted_price),
                    'formatted_price': f"${predicted_price:.2f}"
                })
            
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e),
                    'address': property_data.get('Address', f'Property {idx + 1}')
                })
        
        # Calculate summary statistics
        successful_predictions = [r['predicted_price'] for r in results if 'predicted_price' in r]
        
        summary = {
            'total_properties': len(properties),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(properties) - len(successful_predictions)
        }
        
        if successful_predictions:
            summary.update({
                'average_price': float(np.mean(successful_predictions)),
                'min_price': float(np.min(successful_predictions)),
                'max_price': float(np.max(successful_predictions)),
                'total_estimated_revenue': float(np.sum(successful_predictions))
            })
        
        response = {
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        info = {
            'model_type': model.best_model_name if hasattr(model, 'best_model_name') else 'Unknown',
            'features_used': model.feature_names if hasattr(model, 'feature_names') else [],
            'metrics': model.model_metrics if hasattr(model, 'model_metrics') else {},
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Please train the model first.")
        sys.exit(1)
