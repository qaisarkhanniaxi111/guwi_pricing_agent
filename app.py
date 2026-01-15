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
import requests

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

# RapidAPI configuration for property data
RAPIDAPI_KEY = 'edb85c1e18msh5a90a5d85106214p1f8f6djsn3cbe051d609f'

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


def get_zillow_data(address):
    """
    Get comprehensive property data from Zillow API
    Returns both ZPID and property details
    """
    url = f'https://private-zillow.p.rapidapi.com/pro/byaddress?propertyaddress={address}'
    
    headers = {
        'x-rapidapi-host': 'private-zillow.p.rapidapi.com',
        'x-rapidapi-key': RAPIDAPI_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        property_details = data.get('propertyDetails', {})
        reso_facts = property_details.get('resoFacts', {})
        
        # Calculate average from nearby homes - EXACTLY like your JavaScript
        avg_nearby_value = None
        nearby_homes = data.get('nearbyHomes', [])  # From main data, not propertyDetails
        if nearby_homes and len(nearby_homes) > 0:
            # Get all valid prices (filter out None and <= 0)
            valid_prices = [h.get('price') for h in nearby_homes if h.get('price') and h.get('price') > 0]
            
            if len(valid_prices) > 0:
                # Calculate average and round
                avg_nearby_value = round(sum(valid_prices) / len(valid_prices))
        
        # Extract data from Zillow response - matching your JavaScript exactly
        zillow_data = {
            'zpid': str(property_details.get('zpid', '')),
            'squareFootage': clean_numeric_value(data.get('livingAreaValue') or data.get('livingArea')),
            'homeValue': clean_numeric_value(data.get('price')),
            'roofType': reso_facts.get('roofType'),
            'stories': clean_numeric_value(reso_facts.get('stories') or reso_facts.get('storiesTotal')),
            'avgNearbyValue': avg_nearby_value  # Already numeric
        }
        
        logger.info(f"Zillow data extracted: {zillow_data}")
        return zillow_data
        
    except Exception as e:
        logger.error(f"Error getting Zillow data: {e}")
        return None


def get_us_property_data(zpid):
    """Get property details from US Property Market API and extract fields like JavaScript"""
    url = f'https://us-property-market1.p.rapidapi.com/property?zpid={zpid}'
    
    headers = {
        'x-rapidapi-host': 'us-property-market1.p.rapidapi.com',
        'x-rapidapi-key': RAPIDAPI_KEY
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Get resoFacts from data
        reso_facts = data.get('resoFacts', {})
        
        # Calculate average from nearby homes - EXACTLY like your JavaScript
        avg_nearby_value = None
        nearby_homes = data.get('nearbyHomes', [])
        if nearby_homes and len(nearby_homes) > 0:
            # Get all valid prices (filter out None and <= 0)
            valid_prices = [h.get('price') for h in nearby_homes if h.get('price') and h.get('price') > 0]
            
            if len(valid_prices) > 0:
                # Calculate average and round
                avg_nearby_value = round(sum(valid_prices) / len(valid_prices))
        
        # Extract fields EXACTLY like your JavaScript
        us_data = {
            'squareFootage': clean_numeric_value(data.get('livingAreaValue') or data.get('livingArea')),
            'homeValue': clean_numeric_value(data.get('price')),
            'roofType': reso_facts.get('roofType'),
            'stories': clean_numeric_value(reso_facts.get('stories') or reso_facts.get('storiesTotal')),
            'avgNearbyValue': avg_nearby_value  # Already numeric
        }
        
        logger.info(f"US Property data extracted: {us_data}")
        return us_data
        
    except Exception as e:
        logger.error(f"Error getting US property data: {e}")
        return None


def merge_property_data(zillow_data, us_data):
    """
    Merge data from both APIs, preferring non-null values
    Priority: Use whichever API has data for each field
    """
    merged = {}
    
    # Get zpid from Zillow data
    merged['zpid'] = zillow_data.get('zpid', '')
    
    # For each field, use non-null value from either source
    fields = ['roofType', 'squareFootage', 'homeValue', 'avgNearbyValue', 'stories']
    
    for field in fields:
        zillow_val = zillow_data.get(field)
        us_val = us_data.get(field) if us_data else None
        
        # Use Zillow value if available, otherwise US Property value
        if zillow_val is not None:
            merged[field] = zillow_val
        elif us_val is not None:
            merged[field] = us_val
        else:
            merged[field] = None
    
    logger.info(f"Merged property data: {merged}")
    return merged


def clean_numeric_value(value):
    """
    Clean numeric values from API responses
    Converts: "1,310 sqft" -> 1310
    Converts: "$1,500,000" -> 1500000
    """
    if value is None:
        return None
    
    # If already a number, return it
    if isinstance(value, (int, float)):
        return value
    
    # Convert to string and clean
    value_str = str(value)
    
    # Remove common text: sqft, sq ft, $, commas
    value_str = value_str.lower()
    value_str = value_str.replace('sqft', '').replace('sq ft', '').replace('$', '').replace(',', '').strip()
    
    # Try to convert to number
    try:
        # Try int first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except:
        return None


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
            '/predict_all': 'Get prices for all services (POST)',
            '/predict_by_address/<service>': 'Predict by address (POST)',
            '/predict_all_by_address': 'Get all prices by address (POST)',
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
        
        # MINIMAL COLUMNS - Only 5 core fields
        defaults = {
            'Home Square Footage': 1500,
            'Home Value': 1400,
            'Average Home value in Zip code': 1600,
            'Number of Stories': 1
        }
        
        # Apply defaults for missing fields
        for field, default_value in defaults.items():
            if field not in data:
                data[field] = default_value
        
        # Handle Roof Type - ensure ONLY ONE column exists
        roof_type_value = None
        
        # Get roof type from either column name
        if 'Roof Type' in data:
            roof_type_value = data['Roof Type']
        elif 'Roof Type/ Material' in data:
            roof_type_value = data['Roof Type/ Material']
        
        # Default to N/A if not provided or empty
        if not roof_type_value or roof_type_value in ['', 'null', 'NULL', 'None', None]:
            roof_type_value = 'N/A'
        
        # Clear any existing roof type columns
        data.pop('Roof Type', None)
        data.pop('Roof Type/ Material', None)
        
        # Add back as SINGLE column
        data['Roof Type/ Material'] = roof_type_value
        
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
                    # MINIMAL COLUMNS - Only 5 core fields
                    defaults = {
                        'Home Square Footage': 1500,
                        'Home Value': 1400,
                        'Average Home value in Zip code': 1600,
                        'Number of Stories': 1
                    }
                    
                    data_copy = data.copy()
                    
                    # Apply defaults
                    for field, default_value in defaults.items():
                        if field not in data_copy:
                            data_copy[field] = default_value
                    
                    # Handle Roof Type - ensure ONLY ONE column
                    roof_type_value = None
                    if 'Roof Type' in data_copy:
                        roof_type_value = data_copy['Roof Type']
                    elif 'Roof Type/ Material' in data_copy:
                        roof_type_value = data_copy['Roof Type/ Material']
                    
                    if not roof_type_value or roof_type_value in ['', 'null', 'NULL', 'None', None]:
                        roof_type_value = 'N/A'
                    
                    # Clear and set single column
                    data_copy.pop('Roof Type', None)
                    data_copy.pop('Roof Type/ Material', None)
                    data_copy['Roof Type/ Material'] = roof_type_value
                    
                    df = pd.DataFrame([data_copy])
                    predicted_price = config['model'].predict(df)
                    
                    predictions[service_name] = {
                        'description': config['description'],
                        'price': float(predicted_price),
                        'formatted': f"${predicted_price:.2f}"
                    }
                    
                    total_price += predicted_price
                    
                except Exception as e:
                    logger.error(f"Error predicting {service_name}: {str(e)}")
                    predictions[service_name] = {
                        'description': config['description'],
                        'error': str(e),
                        'status': 'failed'
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


@app.route('/predict_by_address/<service>', methods=['POST'])
def predict_by_address(service):
    """
    Predict price using just address - automatically fetches property data
    
    Request body:
    {
        "address": "100 Northwest 79th Street, Seattle, Washington, 98117"
    }
    """
    
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
        # Get address from request
        data = request.json
        if not data or 'address' not in data:
            return jsonify({'error': 'No address provided'}), 400
        
        address = data['address']
        logger.info(f"Getting property data for: {address}")
        
        # Step 1: Get data from Zillow API
        zillow_data = get_zillow_data(address)
        if not zillow_data or not zillow_data.get('zpid'):
            return jsonify({'error': 'Could not get property data from Zillow'}), 400
        
        zpid = zillow_data['zpid']
        logger.info(f"Found ZPID: {zpid}")
        
        # Step 2: Get data from US Property Market API
        us_data = get_us_property_data(zpid)
        
        # Step 3: Merge data from both APIs
        property_data = merge_property_data(zillow_data, us_data)
        
        logger.info(f"Final merged property data: {property_data}")
        
        # Step 4: Prepare data for prediction with fallbacks ONLY if both APIs return None
        api_data = {
            'Roof Type': property_data.get('roofType') or 'N/A',
            'Home Square Footage': property_data.get('squareFootage') or 1500,
            'Home Value': property_data.get('homeValue') or 1400,
            'Average Home value in Zip code': property_data.get('avgNearbyValue') or 1600,
            'Number of Stories': property_data.get('stories') or 1
        }
        
        # Handle Roof Type - ensure ONLY ONE column
        roof_type_value = api_data['Roof Type']
        api_data.pop('Roof Type', None)
        api_data['Roof Type/ Material'] = roof_type_value
        
        # Step 5: Make prediction
        df = pd.DataFrame([api_data])
        model = SERVICES[service]['model']
        predicted_price = model.predict(df)
        
        # Calculate confidence
        confidence = 'high' if predicted_price < 600 else 'medium'
        if predicted_price > 800:
            confidence = 'low - manual review recommended'
        
        # Prepare response
        response = {
            'service': service,
            'description': SERVICES[service]['description'],
            'address': address,
            'zpid': zpid,
            'property_data': property_data,
            'data_sources': {
                'zillow': 'Zillow API provided initial data',
                'us_property': 'US Property Market API provided additional data',
                'merged': 'Used non-null values from both sources'
            },
            'predicted_price': float(predicted_price),
            'formatted_price': f"${predicted_price:.2f}",
            'confidence': confidence,
            'model': model.best_model_name if hasattr(model, 'best_model_name') else 'Unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction by address for {service}: ${predicted_price:.2f}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in predict_by_address for {service}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_all_by_address', methods=['POST'])
def predict_all_by_address():
    """
    Get prices for ALL services using just address
    
    Request body:
    {
        "address": "100 Northwest 79th Street, Seattle, Washington, 98117"
    }
    """
    
    try:
        # Get address from request
        data = request.json
        if not data or 'address' not in data:
            return jsonify({'error': 'No address provided'}), 400
        
        address = data['address']
        logger.info(f"Getting all service prices for: {address}")
        
        # Step 1: Get data from Zillow API
        zillow_data = get_zillow_data(address)
        if not zillow_data or not zillow_data.get('zpid'):
            return jsonify({'error': 'Could not get property data from Zillow'}), 400
        
        zpid = zillow_data['zpid']
        logger.info(f"Found ZPID: {zpid}")
        
        # Step 2: Get data from US Property Market API
        us_data = get_us_property_data(zpid)
        
        # Step 3: Merge data from both APIs
        property_data = merge_property_data(zillow_data, us_data)
        
        logger.info(f"Final merged property data: {property_data}")
        
        # Step 4: Prepare data for prediction with fallbacks ONLY if both APIs return None
        api_data = {
            'Roof Type/ Material': property_data.get('roofType') or 'N/A',
            'Home Square Footage': property_data.get('squareFootage') or 1500,
            'Home Value': property_data.get('homeValue') or 1400,
            'Average Home value in Zip code': property_data.get('avgNearbyValue') or 1600,
            'Number of Stories': property_data.get('stories') or 1
        }
        
        # Step 5: Get predictions for all services
        predictions = {}
        total_price = 0
        
        for service_name, config in SERVICES.items():
            if config['model'] is not None:
                try:
                    df = pd.DataFrame([api_data])
                    predicted_price = config['model'].predict(df)
                    
                    predictions[service_name] = {
                        'description': config['description'],
                        'price': float(predicted_price),
                        'formatted': f"${predicted_price:.2f}"
                    }
                    
                    total_price += predicted_price
                    
                except Exception as e:
                    logger.error(f"Error predicting {service_name}: {str(e)}")
                    predictions[service_name] = {
                        'description': config['description'],
                        'error': str(e),
                        'status': 'failed'
                    }
        
        # Calculate bundle discount
        bundle_discount = total_price * 0.10 if len(predictions) >= 3 else 0
        bundle_price = total_price - bundle_discount
        
        response = {
            'address': address,
            'zpid': zpid,
            'property_data': property_data,
            'data_sources': {
                'zillow': 'Zillow API provided initial data',
                'us_property': 'US Property Market API provided additional data',
                'merged': 'Used non-null values from both sources'
            },
            'predictions': predictions,
            'summary': {
                'total_individual': float(total_price),
                'bundle_discount': float(bundle_discount),
                'bundle_price': float(bundle_price),
                'formatted_total': f"${total_price:.2f}",
                'formatted_bundle': f"${bundle_price:.2f}"
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"All predictions by address complete: {len(predictions)} services")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in predict_all_by_address: {str(e)}")
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
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_ENV', 'production') != 'production'
    
    print(f"\n{'='*70}")
    print(f"Multi-Service Price Prediction API")
    print(f"{'='*70}")
    print(f"Loaded {loaded}/{len(SERVICES)} models")
    print(f"Starting on port {port}")
    print(f"Using MINIMAL 5-column approach")
    print(f"{'='*70}\n")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)