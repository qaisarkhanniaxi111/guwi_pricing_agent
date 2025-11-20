"""
Test Client for Multi-Service API
Tests all services: Gutter, Chemical, Zinc, Exterior/Interior Window
"""

import requests
import json

# Change this to your API URL
API_URL = "http://localhost:5000"

def test_services_list():
    """Test listing all services"""
    print("=" * 70)
    print("TEST: List Available Services")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/services")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def test_single_service(service):
    """Test prediction for a single service"""
    print("=" * 70)
    print(f"TEST: Predict {service}")
    print("=" * 70)
    
    data = {
        "Address": "123 Main St",
        "City": "Austin",
        "State": "TX",
        "Zip": "78701",
        "Roof Type": "Composition",
        "Home Square Footage": 2000,
        "Home Value": 1800,
        "Average Home value in Zip code": 1700,
        "Number of Stories": 2
    }
    
    response = requests.post(f"{API_URL}/predict/{service}", json=data)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def test_all_services():
    """Test getting prices for all services at once"""
    print("=" * 70)
    print("TEST: Predict All Services (Bundle Quote)")
    print("=" * 70)
    
    data = {
        "Address": "456 Oak Avenue",
        "City": "Dallas",
        "State": "TX",
        "Zip": "75201",
        "Roof Type": "Tile",
        "Home Square Footage": 2500,
        "Home Value": 2200,
        "Average Home value in Zip code": 2000,
        "Number of Stories": 2,
        "Roof Details >> Roof: >> Steepness": "10 / 12"
    }
    
    response = requests.post(f"{API_URL}/predict_all", json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n📋 PROPERTY:")
        print(f"   {result['property']['address']}")
        print(f"   {result['property']['city']}, {result['property']['state']}")
        
        print("\n💰 INDIVIDUAL SERVICE PRICES:")
        for service, details in result['predictions'].items():
            if 'error' not in details:
                print(f"   {details['description']:<35} {details['formatted']:>10}")
        
        print(f"\n   {'─'*45}")
        print(f"   {'Total (Individual):':<35} ${result['summary']['total_individual']:>9.2f}")
        print(f"   {'Bundle Discount (10%):':<35} -${result['summary']['bundle_discount']:>8.2f}")
        print(f"   {'─'*45}")
        print(f"   {'Bundle Price:':<35} ${result['summary']['bundle_price']:>9.2f}")
        print(f"   {'You Save:':<35} ${result['summary']['bundle_discount']:>9.2f}")
    else:
        print(json.dumps(response.json(), indent=2))
    
    print()


def test_model_info(service):
    """Test getting model information"""
    print("=" * 70)
    print(f"TEST: Model Info for {service}")
    print("=" * 70)
    
    response = requests.get(f"{API_URL}/model_info/{service}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nService: {result['description']}")
        print(f"Model: {result['model_type']}")
        print(f"Features: {len(result['features_used'])}")
        
        metrics = result.get('metrics', {})
        if metrics:
            print(f"\nPerformance Metrics:")
            print(f"  MAE:  ${metrics.get('MAE', 0):.2f}")
            print(f"  RMSE: ${metrics.get('RMSE', 0):.2f}")
            print(f"  R²:   {metrics.get('R2', 0):.4f}")
    else:
        print(json.dumps(response.json(), indent=2))
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTI-SERVICE API TEST CLIENT")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    print()
    
    try:
        # Test health check
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API is running\n")
        else:
            print("❌ API health check failed\n")
            exit(1)
        
        # List services
        test_services_list()
        
        # Test individual services
        services = [
            'gutter_clearing',
            'chemical_spray', 
            'zinc_treatment',
            'exterior_window',
            'interior_window'
        ]
        
        for service in services:
            test_single_service(service)
        
        # Test bundle quote
        test_all_services()
        
        # Test model info
        test_model_info('gutter_clearing')
        
        print("=" * 70)
        print("ALL TESTS COMPLETED!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API.")
        print(f"   Make sure the API is running at: {API_URL}")
    except Exception as e:
        print(f"❌ ERROR: {e}")