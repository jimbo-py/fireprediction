import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta

class FireRiskPredictionAI:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb)],
            voting='soft'
        )
        self.scaler = StandardScaler()
        
        # Define risk thresholds
        self.risk_levels = {
            'extreme': 0.90,
            'high': 0.70,
            'moderate': 0.50,
            'low': 0.30
        }

    def scrape_weather_data(self, location):
        """
        Fetch weather data with improved error handling and API validation.
        """
        try:
            api_key = ''  # Insert the API key directly here
            if not api_key:
                raise ValueError("OpenWeather API key not found.")

            api_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': api_key,
                'units': 'metric'
            }
            print(f"Request URL: {api_url}?q={location}&appid={api_key}&units=metric")

            response = requests.get(api_url, params=params)
            
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Content: {response.text}")
            response.raise_for_status()
            weather = response.json()
            
            print("Full JSON response: ", weather)

            # Validate response structure
            required_fields = ['main', 'wind', 'weather']
            if not all(field in weather for field in required_fields):
                raise ValueError(f"Invalid response structure: Missing fields in {weather}")
            
            weather_description = weather['weather'][0].get('description', 'No description available')   

            # Calculate drought index based on available data
            drought_index = self._calculate_drought_index(
                weather['main'].get('temp', 0),
                weather['main'].get('humidity', 0),
                weather.get('rain', {}).get('1h', 0.0)
            )

            weather_data = {
                'temperature': weather['main']['temp'],
                'humidity': weather['main']['humidity'],
                'wind_speed': weather['wind']['speed'],
                'precipitation': weather.get('rain', {}).get('1h', 0.0),
                'drought_index': drought_index,
                'weather_description': weather_description
            }
            return weather_data

        except requests.exceptions.RequestException as e:
            print(f"Network error while fetching weather data: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Data processing error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error in weather data collection: {str(e)}")
            return None

    def scrape_historical_fires(self, region, start_date, end_date):
        """
        Fetch historical fire data with corrected date formatting for ArcGIS API.
        """
        try:
            api_url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Wildland_Fire_Perimeters/FeatureServer/0/query"
            
            # Format region
            formatted_region = self._format_region(region)
            
            # Convert dates to timestamp format for ArcGIS API
            start_timestamp = f"{start_date} 00:00:00"
            end_timestamp = f"{end_date} 23:59:59"
            
            # Create the where clause with proper date formatting
            where_clause = (
                f"STATE = '{formatted_region}' AND "
                f"FireDiscoveryDateTime BETWEEN timestamp '{start_timestamp}' AND timestamp '{end_timestamp}'"
            )
            
            params = {
                'where': where_clause,
                'outFields': 'FireDiscoveryDateTime,IncidentName,DailyAcres,FireCause',
                'returnGeometry': 'false',
                'f': 'json',
                'resultOffset': 0,
                'resultRecordCount': 1000
            }

            # Debug logging
            print(f"API URL: {api_url}")
            print(f"Query parameters: {params}")

            # First try a test query to validate the connection
            test_params = params.copy()
            test_params['resultRecordCount'] = 1  # Just request one record to test
            
            response = requests.get(api_url, params=test_params)
            print(f"Test query response status: {response.status_code}")
            
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"API returned status code {response.status_code}")
                
            # Proceed with full query if test was successful
            all_features = []
            while True:
                response = requests.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    error_msg = data['error'].get('message', 'Unknown error')
                    error_details = data['error'].get('details', [])
                    print(f"API Error: {error_msg}")
                    print(f"Error Details: {error_details}")
                    # For demo purposes, return mock data
                    return self._get_mock_historical_data(formatted_region)
                
                if 'features' not in data:
                    print(f"Unexpected response format: {data}")
                    return self._get_mock_historical_data(formatted_region)

                features = data['features']
                all_features.extend(features)

                if len(features) < 1000:
                    break

                params['resultOffset'] += 1000

            # Calculate severity metrics
            if all_features:
                total_acres = sum(feature['attributes'].get('DailyAcres', 0) for feature in all_features)
                avg_acres = total_acres / len(all_features)
                severity = 'high' if avg_acres > 1000 else 'moderate' if avg_acres > 100 else 'low'
            else:
                severity = 'low'
                total_acres = 0

            return {
                'fire_incidents': all_features,
                'total_incidents': len(all_features),
                'average_severity': severity,
                'total_acres': total_acres
            }

        except Exception as e:
            print(f"Error in historical data collection: {str(e)}")
            return self._get_mock_historical_data(formatted_region)

    def _get_mock_historical_data(self, region):
        """
        Provide mock historical data when API is unavailable.
        This allows the system to continue functioning for demo purposes.
        """
        print("Warning: Using mock historical data for demonstration")
        return {
            'fire_incidents': [],
            'total_incidents': 5,  # Mock reasonable value
            'average_severity': 'moderate',
            'total_acres': 2500
        }

    def _calculate_drought_index(self, temperature, humidity, precipitation):
        """
        Calculate a simple drought index based on available weather data.
        """
        # Higher temperature increases drought risk
        temp_factor = max(0, (temperature - 20) / 30)  # Normalized around 20°C
        
        # Lower humidity increases drought risk
        humidity_factor = max(0, (100 - humidity) / 100)
        
        # Lower precipitation increases drought risk
        precip_factor = max(0, 1 - (precipitation / 10))  # Normalized for 10mm rainfall
        
        # Combine factors (weighted average)
        drought_index = (0.4 * temp_factor + 0.3 * humidity_factor + 0.3 * precip_factor)
        
        return round(drought_index, 2)

    def _format_region(self, region):
        """
        Convert region name to standardized format for API query.
        """
        # Dictionary of major cities to their state codes
        city_to_state = {
            'los angeles': 'CA',
            'san francisco': 'CA',
            'new york': 'NY',
            'chicago': 'IL',
            'houston': 'TX',
            'phoenix': 'AZ',
            'philadelphia': 'PA',
            'san diego': 'CA',
            'dallas': 'TX',
            'san jose': 'CA'
        }
        
        # Dictionary of state names to abbreviations
        state_abbrev = {
            'california': 'CA',
            'oregon': 'OR',
            'washington': 'WA',
            'arizona': 'AZ',
            'texas': 'TX',
            'new york': 'NY',
            'florida': 'FL',
            'illinois': 'IL',
            'pennsylvania': 'PA',
            'ohio': 'OH'
        }
        
        region_lower = region.lower()
        
        if region_lower in city_to_state:
            return city_to_state[region_lower]
        
        if region_lower in state_abbrev:
            return state_abbrev[region_lower]
        
        if len(region) == 2 and region.upper() in state_abbrev.values():
            return region.upper()
        
        # If we can't determine the state, log a warning and return CA as default for Los Angeles
        if 'los angeles' in region_lower:
            print(f"Warning: Defaulting to CA for location: {region}")
            return 'CA'
        
        # If we can't determine the state, raise an error
        raise ValueError(f"Unable to determine state code for location: {region}")

    def extract_features(self, weather_data, historical_data):
        """
        Extract features from weather and historical data for model input.
        """
        features = [
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation'],
            weather_data['drought_index']
        ]

        if historical_data:
            features.extend([
                historical_data['total_incidents'],
                1 if historical_data['average_severity'] == 'high' else 0
            ])

        return features

    def get_risk_level(self, probability):
        """
        Determine risk level based on probability threshold.
        """
        if probability >= self.risk_levels['extreme']:
            return "Extreme"
        elif probability >= self.risk_levels['high']:
            return "High"
        elif probability >= self.risk_levels['moderate']:
            return "Moderate"
        else:
            return "Low"

    def train_model(self, locations, historical_dates, labels):
        """
        Train the model using provided locations and their historical data.
        """
        features = []
        valid_labels = []
        training_summary = {
            'processed_locations': 0,
            'failed_locations': 0,
            'risk_distribution': {'Extreme': 0, 'High': 0, 'Moderate': 0, 'Low': 0}
        }

        for location, date, label in zip(locations, historical_dates, labels):
            weather_data = self.scrape_weather_data(location)
            historical_data = self.scrape_historical_fires(location, date, date)

            if weather_data and historical_data:
                feature_vector = self.extract_features(weather_data, historical_data)
                features.append(feature_vector)
                valid_labels.append(label)

                training_summary['processed_locations'] += 1
            else:
                print(f"Warning: Could not get data for location {location}")
                training_summary['failed_locations'] += 1

        if not features:
            print("Error: No valid data found. Cannot train the model.")
            return

        X = np.array(features)
        y = np.array(valid_labels)

        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n=== Training Summary ===")
        print(f"Total locations processed: {training_summary['processed_locations']}")
        print(f"Failed locations: {training_summary['failed_locations']}")
        print(f"\nModel accuracy: {accuracy:.2f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict_fire_risk(self, location):
        """
        Predict fire risk for a given location.
        """
        if not hasattr(self.model, 'predict'):
            return {"error": "No model loaded. Please train or load a model first."}

        weather_data = self.scrape_weather_data(location)
        current_date = datetime.now().strftime('%Y-%m-%d')
        historical_data = self.scrape_historical_fires(
            location, 
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            current_date
        )

        if not weather_data or not historical_data:
            return {"error": "Could not fetch required data"}

        features = self.extract_features(weather_data, historical_data)
        scaled_features = self.scaler.transform([features])

        probabilities = self.model.predict_proba(scaled_features)[0]
        prediction = self.model.predict(scaled_features)[0]

        risk_score = max(probabilities)
        risk_level = self.get_risk_level(risk_score)

        result = {
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "risk_score": f"{risk_score:.2%}",
            "risk_level": risk_level,
            "current_conditions": {
                "temperature": weather_data['temperature'],
                "humidity": weather_data['humidity'],
                "wind_speed": weather_data['wind_speed'],
                "precipitation": weather_data['precipitation'],
                "drought_index": weather_data['drought_index']
            },
            "historical_data": {
                "total_incidents": historical_data['total_incidents'],
                "average_severity": historical_data['average_severity']
            }
        }

        result["recommendations"] = self._generate_recommendations(result)
        return result

    def _generate_recommendations(self, result):
        """
        Generate recommendations based on risk assessment results.
        """
        recommendations = []

        if result["risk_level"] in ["Extreme", "High"]:
            recommendations.extend([
                "Implement immediate fire prevention measures",
                "Alert local fire authorities of high-risk conditions",
                "Ensure all fire breaks are maintained",
                "Restrict any activities that could spark fires"
            ])
        elif result["risk_level"] == "Moderate":
            recommendations.extend([
                "Monitor conditions closely",
                "Review fire prevention protocols",
                "Ensure fire-fighting equipment is readily available"
            ])
        else:
            recommendations.extend([
                "Maintain regular fire prevention measures",
                "Continue routine monitoring"
            ])

        if result["current_conditions"]["wind_speed"] > 20:
            recommendations.append("High winds present additional risk - take extra precautions")
        if result["current_conditions"]["humidity"] < 30:
            recommendations.append("Low humidity increases fire risk - consider additional preventive measures")

        return recommendations

    def save_model(self, filename):
        """
        Save the trained model and scaler.
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'metadata': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'risk_levels': self.risk_levels
                }
            }
            joblib.dump(model_data, filename)
            print(f"Model successfully saved to {filename}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filename):
        """
        Load a trained model and scaler.
        """
        try:
            if os.path.exists(filename):
                model_data = joblib.load(filename)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.risk_levels = model_data['metadata']['risk_levels']
                print(f"Model successfully loaded from {filename}")
                print(f"Model timestamp: {model_data['metadata']['timestamp']}")
                return True
            else:
                print(f"Error: Model file not found: {filename}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
def main():
    ai_system = FireRiskPredictionAI()
    model_filename = "fire_risk_model.joblib"

    while True:
        print("\n=== Fire Risk Prediction AI System ===")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Predict fire risk for location")
        print("4. Batch analysis for multiple locations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            # Example training data (replace with actual locations and dates)
            locations = ["Location1", "Location2", "Location3"]
            dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
            labels = [0, 1, 1]  # 0: low risk, 1: high risk
            
            print("\nInitiating model training...")
            ai_system.train_model(locations, dates, labels)
            save = input("\nWould you like to save the trained model? (y/n): ")
            if save.lower() == 'y':
                ai_system.save_model(model_filename)

        elif choice == '2':
            model_path = input("\nEnter model path (press Enter for default 'fire_risk_model.joblib'): ").strip()
            if not model_path:
                model_path = model_filename
            ai_system.load_model(model_path)

        elif choice == '3':
            if not hasattr(ai_system.model, 'predict'):
                print("\nError: No model loaded. Please train or load a model first.")
                continue

            location = input("\nEnter the location to analyze: ").strip()
            result = ai_system.predict_fire_risk(location)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
                continue
                
            print("\n=== Risk Assessment Results ===")
            print(f"Prediction: {result['prediction']}")
            print(f"Risk Level: {result['risk_level']} ({result['risk_score']})")
            print("\nCurrent Conditions:")
            for key, value in result['current_conditions'].items():
                print(f"- {key.replace('_', ' ').title()}: {value}")
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"- {rec}")

        elif choice == '4':
            if not hasattr(ai_system.model, 'predict'):
                print("\nError: No model loaded. Please train or load a model first.")
                continue

            locations = input("\nEnter locations (comma-separated): ").strip().split(',')
            
            print("\nProcessing batch analysis...")
            results = []
            for location in locations:
                location = location.strip()
                result = ai_system.predict_fire_risk(location)
                results.append((location, result))

            print("\n=== Batch Analysis Results ===")
            for location, result in results:
                if "error" in result:
                    print(f"\n{location}: Error - {result['error']}")
                else:
                    print(f"\n{location}:")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Risk Level: {result['risk_level']} ({result['risk_score']})")

        elif choice == '5':
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main()
