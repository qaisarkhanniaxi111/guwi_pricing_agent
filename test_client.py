"""
API Test Client
This script demonstrates how to use the Gutter Clearing Price Prediction API
"""

import requests
import json
from typing import Dict, List, Optional

class GutterPriceAPIClient:
    """Client for interacting with the Gutter Price Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def health_check(self) -> Dict:
        """Check if the API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_features(self) -> Dict:
        """Get the list of required and optional features"""
        response = self.session.get(f"{self.base_url}/features")
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        response = self.session.get(f"{self.base_url}/model_info")
        return response.json()
    
    def predict_single(self, property_data: Dict) -> Dict:
        """Predict price for a single property"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=property_data
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}
    
    def predict_batch(self, properties: List[Dict]) -> Dict:
        """Predict prices for multiple properties"""
        response = self.session.post(
            f"{self.base_url}/predict_batch",
            json={"properties": properties}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}

def print_separator():
    """Print a visual separator"""
    print("=" * 60)

def main():
    """Main function to demonstrate API usage"""
    
    # Initialize client
    client = GutterPriceAPIClient()
    
    print_separator()
    print("GUTTER CLEARING PRICE PREDICTION API - TEST CLIENT")
    print_separator()
    
    # 1. Health Check
    print("\n1. Health Check")
    print("-" * 30)
    try:
        health = client.health_check()
        print(f"Status: {health.get('status')}")
        print(f"Model Loaded: {health.get('model_loaded')}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API server is running (python app.py)")
        return
    
    # 2. Get Features Info
    print("\n2. Required Features")
    print("-" * 30)
    features = client.get_features()
    print("Required fields:")
    for field in features.get('required', []):
        print(f"  - {field}")
    
    # 3. Get Model Info
    print("\n3. Model Information")
    print("-" * 30)
    model_info = client.get_model_info()
    print(f"Model Type: {model_info.get('model_type')}")
    if 'metrics' in model_info and model_info['metrics']:
        best_model = model_info.get('model_type')
        if best_model in model_info['metrics']:
            metrics = model_info['metrics'][best_model]
            print(f"MAE: ${metrics.get('MAE', 0):.2f}")
            print(f"R² Score: {metrics.get('R2', 0):.3f}")
    
    # 4. Single Property Prediction
    print("\n4. Single Property Prediction")
    print("-" * 30)
    
    test_property = {
        'Address': '123 Test Street',
        'City': 'Austin',
        'State': 'TX',
        'Zip': '78701',
        'Roof Type': 'Composition',
        'Home Square Footage': 1500,
        'Home Value': 1400,
        'Average Home value in Zip code': 1600,
        'Number of Stories': 1,
        'Roof Details >> Roof: >> Steepness': '8 / 12',
        'Jobsite Ladders >> Roof >> Number of Ladder Movements': '4 - 8',
        'Jobsite Ladders >> Roof >> ladder Size': '32ft',
        'Jobsite Ladders >> Gutter >> Number of Ladder Movements': '9 - 20',
        'Jobsite Ladders >> Gutter >> ladder Size': '32ft',
        'Jobsite Ladders >> Window >> Number of Ladder Movements': '9 - 20',
        'Jobsite Ladders >> Window >> ladder Size': '32ft'
    }
    
    print("Property Details:")
    print(f"  Address: {test_property['Address']}")
    print(f"  Square Footage: {test_property['Home Square Footage']}")
    print(f"  Stories: {test_property['Number of Stories']}")
    print(f"  Roof Type: {test_property['Roof Type']}")
    
    result = client.predict_single(test_property)
    
    if 'error' not in result:
        print(f"\nPredicted Price: {result.get('formatted_price')}")
        print(f"Confidence Interval: {result.get('confidence_interval', {}).get('formatted')}")
    else:
        print(f"Error: {result['error']}")
    
    # 5. Batch Prediction
    print("\n5. Batch Prediction")
    print("-" * 30)
    
    batch_properties = [
        {
            'Roof Type': 'Composition',
            'Home Square Footage': 1200,
            'Home Value': 900,
            'Average Home value in Zip code': 1100,
            'Number of Stories': 1
        },
        {
            'Roof Type': 'Tile',
            'Home Square Footage': 2500,
            'Home Value': 2200,
            'Average Home value in Zip code': 2000,
            'Number of Stories': 2
        },
        {
            'Roof Type': 'Metal',
            'Home Square Footage': 1800,
            'Home Value': 1600,
            'Average Home value in Zip code': 1700,
            'Number of Stories': 1
        }
    ]
    
    batch_result = client.predict_batch(batch_properties)
    
    if 'error' not in batch_result:
        print("Batch Results:")
        for result in batch_result.get('results', []):
            if 'predicted_price' in result:
                print(f"  Property {result['index'] + 1}: {result['formatted_price']}")
            else:
                print(f"  Property {result['index'] + 1}: Error - {result.get('error')}")
        
        summary = batch_result.get('summary', {})
        if 'average_price' in summary:
            print(f"\nSummary:")
            print(f"  Average Price: ${summary.get('average_price', 0):.2f}")
            print(f"  Total Revenue: ${summary.get('total_estimated_revenue', 0):.2f}")
    else:
        print(f"Error: {batch_result['error']}")
    
    print("\n" + "=" * 60)
    print("API TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()