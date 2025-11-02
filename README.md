# Gutter Clearing Price Prediction System

A machine learning-powered API for predicting gutter clearing service prices based on property characteristics.

## Features

- **Multiple ML Models**: Compares 11 different models (Random Forest, XGBoost, LightGBM, Neural Networks, etc.) and automatically selects the best performer
- **Advanced Feature Engineering**: Creates derived features like difficulty scores, ladder complexity metrics, and area value ratios
- **RESTful API**: Easy-to-use Flask API with multiple endpoints for single and batch predictions
- **Comprehensive Validation**: Cross-validation and hyperparameter tuning for optimal performance
- **Production Ready**: Includes error handling, logging, and model persistence

## Project Structure

```
├── gutter_price_model.py   # Core ML model with preprocessing and training
├── train_model.py          # Script to train model from your CSV data
├── app.py                  # Flask API server
├── test_client.py          # API testing client
├── requirements.txt        # Python dependencies
├── gutter_price_model.pkl  # Trained model (created after training)
└── README.md              # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt --break-system-packages
```

## Usage

### Step 1: Train the Model with Your Data

```bash
# If your CSV file is named 'gutter_data.csv'
python train_model.py gutter_data.csv

# Or just run it and enter the path when prompted
python train_model.py
```

The training script will:
- Load your CSV data
- Display data statistics
- Train multiple models
- Select the best performer
- Save the model as `gutter_price_model.pkl`

### Step 2: Start the API Server

```bash
python app.py
```

The API will start on `http://localhost:5000`

### Step 3: Test the API

```bash
# In a new terminal
python test_client.py
```

## API Endpoints

### 1. Health Check
- **GET** `/health`
- Returns API status and model availability

### 2. Get Features List
- **GET** `/features`
- Returns required and optional input features

### 3. Get Model Info
- **GET** `/model_info`
- Returns model type and performance metrics

### 4. Single Prediction
- **POST** `/predict`
- Predicts price for one property

Example request:
```json
{
  "Roof Type": "Composition",
  "Home Square Footage": 1500,
  "Home Value": 1400,
  "Average Home value in Zip code": 1600,
  "Number of Stories": 1,
  "Roof Details >> Roof: >> Steepness": "8 / 12",
  "Jobsite Ladders >> Roof >> Number of Ladder Movements": "4 - 8",
  "Jobsite Ladders >> Roof >> ladder Size": "32ft",
  "Jobsite Ladders >> Gutter >> Number of Ladder Movements": "9 - 20",
  "Jobsite Ladders >> Gutter >> ladder Size": "32ft",
  "Jobsite Ladders >> Window >> Number of Ladder Movements": "9 - 20",
  "Jobsite Ladders >> Window >> ladder Size": "32ft"
}
```

Response:
```json
{
  "predicted_price": 245.50,
  "formatted_price": "$245.50",
  "confidence_interval": {
    "low": 220.95,
    "high": 270.05,
    "formatted": "$220.95 - $270.05"
  },
  "factors": {
    "home_size": 1500,
    "stories": 1,
    "roof_type": "Composition",
    "area_value_ratio": 0.875
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### 5. Batch Prediction
- **POST** `/predict_batch`
- Predicts prices for multiple properties at once

## Data Format

Your CSV file should have these columns:

### Required Columns:
- `Gutter Clearing` - The target price (what we're predicting)
- `Roof Type` - Type of roof (Composition, Tile, Metal, etc.)
- `Home Square Footage` - Size of the home
- `Home Value` - Value of the property (in thousands)
- `Average Home value in Zip code` - Average value in the area
- `Number of Stories` - 1 or 2 story home

### Optional Columns (improve accuracy):
- `Address`, `City`, `State`, `Zip` - Location information
- `Roof Details >> Roof: >> Steepness` - Roof pitch (e.g., "10 / 12")
- `Jobsite Ladders >> Roof >> Number of Ladder Movements` - Range like "4 - 8"
- `Jobsite Ladders >> Roof >> ladder Size` - Size like "40ft"
- `Jobsite Ladders >> Gutter >> Number of Ladder Movements`
- `Jobsite Ladders >> Gutter >> ladder Size`
- `Jobsite Ladders >> Window >> Number of Ladder Movements`
- `Jobsite Ladders >> Window >> ladder Size`

## How the Model Works

### Feature Engineering
The model creates several derived features:
- **Difficulty Score**: Combines stories, steepness, ladder movements, and size
- **Home Value to Area Ratio**: Indicates if property is above/below area average
- **Size per Story**: Square footage divided by number of stories
- **Total Ladder Movements**: Sum of all ladder movement requirements
- **Max Ladder Size**: Maximum ladder size needed

### Pricing Factors
The model considers:
1. **Property Size**: Larger homes require more work
2. **Number of Stories**: Multi-story homes are more complex
3. **Roof Type**: Different materials require different techniques
4. **Roof Steepness**: Steeper roofs are more dangerous/difficult
5. **Ladder Requirements**: More movements = more time
6. **Area Value**: Higher-value areas can support higher prices

### Model Selection
The system trains 11 different models:
- Linear Regression (baseline)
- Ridge, Lasso, ElasticNet (regularized linear)
- Random Forest, Extra Trees (ensemble tree methods)
- Gradient Boosting, XGBoost, LightGBM (boosting methods)
- Support Vector Regression
- Neural Network (Multi-layer Perceptron)

It automatically selects the best performer based on Mean Absolute Error (MAE).

## Python API Client Example

```python
import requests

# Make a prediction
url = "http://localhost:5000/predict"
property_data = {
    "Roof Type": "Composition",
    "Home Square Footage": 1500,
    "Home Value": 1400,
    "Average Home value in Zip code": 1600,
    "Number of Stories": 1
}

response = requests.post(url, json=property_data)
result = response.json()
print(f"Predicted Price: {result['formatted_price']}")
```

## Deployment

For production deployment:

1. **Use a production WSGI server** (instead of Flask's development server):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Set up environment variables** for configuration
3. **Add authentication** if needed
4. **Deploy behind a reverse proxy** (nginx, Apache)
5. **Monitor performance** and retrain periodically with new data

## Model Performance

The model's performance metrics are displayed during training:
- **MAE (Mean Absolute Error)**: Average price prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more
- **R² Score**: Proportion of variance explained (closer to 1 is better)

## Troubleshooting

### Model not loading
- Ensure you've trained the model first: `python train_model.py your_data.csv`
- Check that `gutter_price_model.pkl` exists

### Poor predictions
- Ensure your CSV has enough training data (at least 50-100 rows)
- Check for outliers in your pricing data
- Verify all required columns are present
- Consider adding more optional columns for better accuracy

### API not starting
- Check if port 5000 is already in use
- Ensure all dependencies are installed
- Check Python version (3.7+ required)

## License

This project is provided as-is for commercial use.

## Support

For questions or issues, please refer to the documentation above or modify the code as needed for your specific use case.