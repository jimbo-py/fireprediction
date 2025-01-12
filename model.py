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
        Scrape weather data for fire risk assessment.
        In practice, you would replace this with actual API calls to weather services.
        """
        try:
            # Example weather API endpoint (replace with actual weather API)
            api_url = f"https://api.weatherservice.com/data?location={location}"
            
            # Simulated weather data (replace with actual API call)
            weather_data = {
                'temperature': 25,
                'humidity': 30,
                'wind_speed': 15,
                'precipitation': 0.0,
                'drought_index': 400
            }
            
            return weather_data
            
        except Exception as e:
            print(f"Error scraping weather data: {str(e)}")
            return None

    def scrape_historical_fires(self, region, start_date, end_date):
        """
        Scrape historical fire data from relevant sources.
        In practice, you would replace this with actual API calls to fire databases.
        """
        try:
            # Example fire database API endpoint (replace with actual API)
            api_url = f"https://api.firedatabase.com/historical?region={region}&start={start_date}&end={end_date}"
            
            # Simulated historical fire data (replace with actual API call)
            historical_data = {
                'fire_incidents': [
                    {'date': '2023-01-01', 'severity': 'high', 'area_burned': 1000},
                    {'date': '2023-01-02', 'severity': 'low', 'area_burned': 50}
                ],
                'total_incidents': 2,
                'average_severity': 'moderate'
            }
            
            return historical_data
            
        except Exception as e:
            print(f"Error scraping historical fire data: {str(e)}")
            return None

    def extract_features(self, weather_data, historical_data):
        """
        Extract relevant features for fire risk prediction.
        """
        features = [
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation'],
            weather_data['drought_index']
        ]
        
        # Add historical features
        if historical_data:
            features.extend([
                historical_data['total_incidents'],
                1 if historical_data['average_severity'] == 'high' else 0
            ])
        
        return features

    def get_risk_level(self, probability):
        """
        Determine fire risk level based on prediction probability.
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
        Train the model using scraped data from multiple locations and dates.
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
                
                # Track risk metrics
                training_summary['processed_locations'] += 1
            else:
                print(f"Warning: Could not get data for location {location}")
                training_summary['failed_locations'] += 1

        if not features:
            print("Error: No valid data found. Cannot train the model.")
            return

        X = np.array(features)
        y = np.array(valid_labels)

        # Scale features
        X = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Total locations processed: {training_summary['processed_locations']}")
        print(f"Failed locations: {training_summary['failed_locations']}")
        print(f"\nModel accuracy: {accuracy:.2f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

    def predict_fire_risk(self, location):
        """
        Predict fire risk for a given location using current conditions.
        """
        if not hasattr(self.model, 'predict'):
            return {"error": "No model loaded. Please train or load a model first."}

        # Get current data
        weather_data = self.scrape_weather_data(location)
        current_date = datetime.now().strftime('%Y-%m-%d')
        historical_data = self.scrape_historical_fires(
            location, 
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            current_date
        )

        if not weather_data or not historical_data:
            return {"error": "Could not fetch required data"}

        # Extract and scale features
        features = self.extract_features(weather_data, historical_data)
        scaled_features = self.scaler.transform([features])
        
        # Get probability predictions
        probabilities = self.model.predict_proba(scaled_features)[0]
        prediction = self.model.predict(scaled_features)[0]
        
        # Calculate risk level
        risk_score = max(probabilities)
        risk_level = self.get_risk_level(risk_score)
        
        # Prepare detailed analysis report
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
        
        # Add recommendations
        result["recommendations"] = self._generate_recommendations(result)
        
        return result

    def _generate_recommendations(self, result):
        """
        Generate specific recommendations based on fire risk assessment.
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
            
        # Add specific recommendations based on conditions
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
