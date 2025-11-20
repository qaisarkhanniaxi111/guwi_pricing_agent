"""
Multi-Service Price Prediction API
Supports: Gutter Clearing, Chemical Spray, Zinc Treatment, 
Exterior Window Cleaning, Interior Window Cleaning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gutter_price_model import GutterPricePredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configurations
SERVICES = {
    'gutter_clearing': {
        'model_file': 'gutter_price_model.pkl',
        'description': 'Gutter Clearing Service',
        'model': None
    },
    'chemical_spray': {
        'model_file': 'chemical_spray_model.pkl',
        'description': 'Chemical Spray Service',
        'model': None
    },
    'zinc_treatment': {
        'model_file': 'zinc_treatment_model.pkl',
        'description': 'Zinc Treatment Service',
        'model': None
    },
    'exterior_window': {
        'model_file': 'exterior_window_model.pkl',
        'description': 'Exterior Window Cleaning Service',
        'model': None
    },
    'interior_window': {
        'model_file': 'interior_window_model.pkl',
        'description': 'Interior Window Cleaning Service',
        'model': None
    }
}


def load_all_models():
    """Load all available service models"""
    loaded_count = 0
    
    for service_name, config in SERVICES.items():
        model_file = config['model_file']
        
        if os.path.exists(model_file):
            try:
                predictor = GutterPricePredictor()
                predictor.load_model(model_file)
                config['model'] = predictor
                loaded_count += 1
                logger.info(f"Loaded {config['description']} model")
            except Exception as e:
                logger.error(f"Failed to load {service_name}: {e}")
        else:
            logger.warning(f"Model file not found: {model_file}")
    
    logger.info(f"Loaded {loaded_count}/{len(SERVICES)} models")
    return loaded_count


@app.route('/', methods=['GET'])
def home():
    """API information"""
    available_services = [
        {
            'service': name,
            'description': config['description'],
            'available': config['model'] is not None
        }
        for name, config in SERVICES.items()
    ]
    
    return jsonify({
        'message': 'Multi-Service Price Prediction API',
        'version': '2.0.0',
        'services': available_services,
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/services': 'List available services',
            '/predict/<service>': 'Predict price for a service (POST)',
            '/predict_batch/<service>': 'Batch predictions (POST)',
            '/predict_all': 'Get prices for all services (POST)',
            '/model_info/<service>': 'Get model information'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    loaded_services = sum(1 for config in SERVICES.values() if config['model'] is not None)
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': f"{loaded_services}/{len(SERVICES)}",
        'timestamp': datetime.now().isoformat()
    })


@app.route('/services', methods=['GET'])
def list_services():
    """List all available services"""
    services_list = []
    
    for name, config in SERVICES.items():
        service_info = {
            'service': name,
            'description': config['description'],
            'available': config['model'] is not None,
            'endpoint': f'/predict/{name}'
        }
        services_list.append(service_info)
    
    return jsonify({
        'services': services_list,
        'total': len(services_list),
        'available': sum(1 for s in services_list if s['available'])
    })


@app.route('/predict/<service>', methods=['POST'])
def predict(service):
    """Predict price for a specific service"""
    
    # Validate service
    if service not in SERVICES:
        return jsonify({
            'error': f'Unknown service: {service}',
            'available_services': list(SERVICES.keys())
        }), 400
    
    # Check if model is loaded
    if SERVICES[service]['model'] is None:
        return jsonify({
            'error': f'Model not available for {service}',
            'message': 'Model file not found or failed to load'
        }), 503
    
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Set defaults for optional fields
        defaults = {
            'Address': 'Unknown',
            'City': 'Unknown',
            'State': 'Unknown',
            'Zip': '00000',
            'Roof Type': 'Composition',
            'Home Square Footage': 1500,
            'Home Value': 1400,
            'Average Home value in Zip code': 1600,
            'Number of Stories': 1,
            'Roof Details >> Roof: >> Steepness': '6 / 12',
            'Jobsite Ladders >> Roof >> Number of Ladder Movements': '4 - 8',
            'Jobsite Ladders >> Roof >> ladder Size': '32ft',
            'Jobsite Ladders >> Gutter >> Number of Ladder Movements': '9 - 20',
            'Jobsite Ladders >> Gutter >> ladder Size': '32ft',
            'Jobsite Ladders >> Window >> Number of Ladder Movements': '9 - 20',
            'Jobsite Ladders >> Window >> ladder Size': '32ft'
        }
        
        # Apply defaults
        for field, default_value in defaults.items():
            if field not in data:
                data[field] = default_value
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        model = SERVICES[service]['model']
        predicted_price = model.predict(df)
        
        # Calculate confidence based on price range
        confidence = 'high' if predicted_price < 600 else 'medium'
        if predicted_price > 800:
            confidence = 'low - manual review recommended'
        
        # Prepare response
        response = {
            'service': service,
            'description': SERVICES[service]['description'],
            'predicted_price': float(predicted_price),
            'formatted_price': f"${predicted_price:.2f}",
            'confidence': confidence,
            'model': model.best_model_name if hasattr(model, 'best_model_name') else 'Unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction for {service}: ${predicted_price:.2f}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in prediction for {service}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Get price predictions for all available services"""
    
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        predictions = {}
        total_price = 0
        
        # Get predictions from all available models
        for service_name, config in SERVICES.items():
            if config['model'] is not None:
                try:
                    # Set defaults
                    defaults = {
                        'Address': 'Unknown',
                        'City': 'Unknown',
                        'State': 'Unknown',
                        'Zip': '00000',
                        'Roof Type': 'Composition',
                        'Home Square Footage': 1500,
                        'Home Value': 1400,
                        'Average Home value in Zip code': 1600,
                        'Number of Stories': 1,
                        'Roof Details >> Roof: >> Steepness': '6 / 12',
                        'Jobsite Ladders >> Roof >> Number of Ladder Movements': '4 - 8',
                        'Jobsite Ladders >> Roof >> ladder Size': '32ft',
                        'Jobsite Ladders >> Gutter >> Number of Ladder Movements': '9 - 20',
                        'Jobsite Ladders >> Gutter >> ladder Size': '32ft',
                        'Jobsite Ladders >> Window >> Number of Ladder Movements': '9 - 20',
                        'Jobsite Ladders >> Window >> ladder Size': '32ft'
                    }
                    
                    data_copy = data.copy()
                    for field, default_value in defaults.items():
                        if field not in data_copy:
                            data_copy[field] = default_value
                    
                    df = pd.DataFrame([data_copy])
                    predicted_price = config['model'].predict(df)
                    
                    predictions[service_name] = {
                        'description': config['description'],
                        'price': float(predicted_price),
                        'formatted': f"${predicted_price:.2f}"
                    }
                    
                    total_price += predicted_price
                    
                except Exception as e:
                    predictions[service_name] = {
                        'error': str(e)
                    }
        
        # Calculate bundle discount (10% off if getting all services)
        bundle_discount = total_price * 0.10 if len(predictions) >= 3 else 0
        bundle_price = total_price - bundle_discount
        
        response = {
            'predictions': predictions,
            'summary': {
                'total_individual': float(total_price),
                'bundle_discount': float(bundle_discount),
                'bundle_price': float(bundle_price),
                'formatted_total': f"${total_price:.2f}",
                'formatted_bundle': f"${bundle_price:.2f}"
            },
            'property': {
                'address': data.get('Address', 'Unknown'),
                'city': data.get('City', 'Unknown'),
                'state': data.get('State', 'Unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in predict_all: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info/<service>', methods=['GET'])
def model_info(service):
    """Get information about a specific service model"""
    
    if service not in SERVICES:
        return jsonify({
            'error': f'Unknown service: {service}',
            'available_services': list(SERVICES.keys())
        }), 400
    
    if SERVICES[service]['model'] is None:
        return jsonify({
            'error': f'Model not available for {service}'
        }), 503
    
    try:
        model = SERVICES[service]['model']
        
        info = {
            'service': service,
            'description': SERVICES[service]['description'],
            'model_type': model.best_model_name if hasattr(model, 'best_model_name') else 'Unknown',
            'features_used': model.feature_names if hasattr(model, 'feature_names') else [],
            'metrics': model.model_metrics.get(model.best_model_name, {}) if hasattr(model, 'model_metrics') else {},
            'status': 'active'
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load all models on startup
    loaded = load_all_models()
    
    if loaded == 0:
        logger.error("No models loaded! Please train models first.")
        print("\n❌ ERROR: No models found!")
        print("Please run: python train_all_services.py")
        sys.exit(1)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') != 'production'
    
    print(f"\n{'='*70}")
    print(f"Multi-Service Price Prediction API")
    print(f"{'='*70}")
    print(f"Loaded {loaded}/{len(SERVICES)} models")
    print(f"Starting on port {port}")
    print(f"{'='*70}\n")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)