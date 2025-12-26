"""
AI Analytics Engine for Public Health Monitoring
Uses pre-built anomaly detection and outbreak prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import json


class HealthAnalyticsEngine:
    def __init__(self, contamination=0.1):
        """
        Initialize the analytics engine

        Args:
            contamination: Expected proportion of outliers (default 0.1 = 10%)
        """
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.baseline_mean = None
        self.baseline_std = None

    def prepare_data(self, health_records: List[Dict]) -> pd.DataFrame:
        """
        Convert health records from blockchain to DataFrame

        Expected format:
        {
            'hospital_id': 'A',
            'timestamp': '2025-01-15',
            'symptom_counts': {'fever': 45, 'cough': 32, 'fatigue': 28},
            'total_cases': 105
        }
        """
        df = pd.DataFrame(health_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Extract symptom counts into separate columns
        if 'symptom_counts' in df.columns:
            symptom_df = pd.json_normalize(df['symptom_counts'])
            df = pd.concat([df.drop('symptom_counts', axis=1), symptom_df], axis=1)

        return df

    def train_baseline(self, historical_data: pd.DataFrame):
        """
        Train anomaly detector on historical 'normal' data
        """
        # Select numeric columns for training
        numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
        X = historical_data[numeric_cols].fillna(0)

        # Store baseline statistics
        self.baseline_mean = X.mean()
        self.baseline_std = X.std()

        # Scale and train
        X_scaled = self.scaler.fit_transform(X)
        self.anomaly_detector.fit(X_scaled)
        self.is_trained = True

        print(f"✓ Baseline trained on {len(historical_data)} records")
        print(f"✓ Monitoring {len(numeric_cols)} features: {list(numeric_cols)}")

    def detect_anomalies(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in current data

        Returns DataFrame with anomaly scores and labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_baseline() first.")

        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        X = current_data[numeric_cols].fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        anomaly_scores = self.anomaly_detector.score_samples(X_scaled)
        anomaly_labels = self.anomaly_detector.predict(X_scaled)

        # Add results to dataframe
        result = current_data.copy()
        result['anomaly_score'] = anomaly_scores
        result['is_anomaly'] = anomaly_labels == -1  # -1 means anomaly

        return result

    def calculate_severity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate severity score based on deviation from baseline
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        severity_scores = []
        for idx, row in data.iterrows():
            # Calculate z-scores for each feature
            z_scores = []
            for col in numeric_cols:
                if col in self.baseline_mean.index:
                    z = (row[col] - self.baseline_mean[col]) / (self.baseline_std[col] + 1e-10)
                    z_scores.append(abs(z))

            # Severity is average absolute z-score
            severity = np.mean(z_scores) if z_scores else 0
            severity_scores.append(severity)

        data['severity'] = severity_scores
        return data

    def generate_alerts(self, analyzed_data: pd.DataFrame,
                        severity_threshold=2.0) -> List[Dict]:
        """
        Generate alerts for anomalies
        """
        alerts = []

        anomalies = analyzed_data[analyzed_data['is_anomaly'] == True]

        for idx, row in anomalies.iterrows():
            alert = {
                'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}",
                'timestamp': str(row['timestamp']),
                'hospital_id': row.get('hospital_id', 'UNKNOWN'),
                'severity': float(row.get('severity', 0)),
                'anomaly_score': float(row['anomaly_score']),
                'details': {}
            }

            # Add elevated symptoms
            numeric_cols = [col for col in row.index if col not in
                            ['timestamp', 'hospital_id', 'anomaly_score', 'is_anomaly', 'severity']]

            for col in numeric_cols:
                if col in self.baseline_mean.index:
                    expected = self.baseline_mean[col]
                    actual = row[col]
                    if actual > expected * 1.5:  # 50% increase
                        alert['details'][col] = {
                            'expected': float(expected),
                            'actual': float(actual),
                            'increase_pct': float((actual - expected) / expected * 100)
                        }

            # Set alert level
            if alert['severity'] >= severity_threshold * 1.5:
                alert['level'] = 'CRITICAL'
            elif alert['severity'] >= severity_threshold:
                alert['level'] = 'HIGH'
            else:
                alert['level'] = 'MEDIUM'

            alerts.append(alert)

        return alerts

    def predict_outbreak_risk(self, recent_data: pd.DataFrame,
                              lookback_days=7) -> Dict:
        """
        Simple outbreak prediction based on trend analysis
        """
        # Calculate daily total cases
        recent_data = recent_data.sort_values('timestamp')
        recent_data['date'] = recent_data['timestamp'].dt.date

        daily_totals = recent_data.groupby('date')['total_cases'].sum().reset_index()

        if len(daily_totals) < 3:
            return {
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'trend': 'INSUFFICIENT_DATA'
            }

        # Calculate trend (simple linear regression on last week)
        recent = daily_totals.tail(lookback_days)
        x = np.arange(len(recent))
        y = recent['total_cases'].values

        # Linear fit
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate growth rate
        avg_cases = y.mean()
        growth_rate = (slope / avg_cases) * 100 if avg_cases > 0 else 0

        # Determine risk level
        if growth_rate > 20:
            risk_level = 'CRITICAL'
            confidence = min(growth_rate / 50, 1.0)
        elif growth_rate > 10:
            risk_level = 'HIGH'
            confidence = min(growth_rate / 30, 1.0)
        elif growth_rate > 5:
            risk_level = 'MEDIUM'
            confidence = min(growth_rate / 20, 1.0)
        else:
            risk_level = 'LOW'
            confidence = 1.0 - min(abs(growth_rate) / 10, 0.8)

        # Predict next 3 days
        predictions = []
        for i in range(1, 4):
            pred_value = slope * (len(x) + i) + intercept
            predictions.append({
                'day': i,
                'predicted_cases': max(0, int(pred_value))
            })

        return {
            'risk_level': risk_level,
            'confidence': float(confidence),
            'trend': 'INCREASING' if slope > 0 else 'DECREASING',
            'growth_rate_pct': float(growth_rate),
            'current_avg_daily_cases': float(avg_cases),
            'predictions': predictions
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== AI Analytics Engine Demo ===\n")

    # Generate synthetic historical data (normal baseline)
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    hospitals = ['A', 'B', 'C']

    historical_records = []
    for date in dates:
        for hospital in hospitals:
            record = {
                'hospital_id': hospital,
                'timestamp': date,
                'fever': np.random.randint(20, 40),
                'cough': np.random.randint(15, 30),
                'fatigue': np.random.randint(10, 25),
                'headache': np.random.randint(5, 15),
                'total_cases': np.random.randint(50, 100)
            }
            historical_records.append(record)

    # Generate current data with some anomalies (outbreak simulation)
    current_dates = pd.date_range(start='2025-02-01', end='2025-02-07', freq='D')
    current_records = []
    for i, date in enumerate(current_dates):
        for hospital in hospitals:
            # Simulate outbreak starting day 4
            outbreak_multiplier = 1.0 if i < 3 else (1.0 + (i - 2) * 0.3)

            record = {
                'hospital_id': hospital,
                'timestamp': date,
                'fever': int(np.random.randint(20, 40) * outbreak_multiplier),
                'cough': int(np.random.randint(15, 30) * outbreak_multiplier),
                'fatigue': int(np.random.randint(10, 25) * outbreak_multiplier),
                'headache': int(np.random.randint(5, 15) * outbreak_multiplier),
                'total_cases': int(np.random.randint(50, 100) * outbreak_multiplier)
            }
            current_records.append(record)

    # Initialize and train engine
    engine = HealthAnalyticsEngine(contamination=0.15)

    historical_df = engine.prepare_data(historical_records)
    print("1. Training baseline model...")
    engine.train_baseline(historical_df)

    # Analyze current data
    print("\n2. Analyzing current data...")
    current_df = engine.prepare_data(current_records)
    analyzed = engine.detect_anomalies(current_df)
    analyzed = engine.calculate_severity(analyzed)

    print(f"   Found {analyzed['is_anomaly'].sum()} anomalies out of {len(analyzed)} records")

    # Generate alerts
    print("\n3. Generating alerts...")
    alerts = engine.generate_alerts(analyzed, severity_threshold=1.5)
    print(f"   Generated {len(alerts)} alerts")

    for alert in alerts[:3]:  # Show first 3
        print(f"\n   {alert['level']} Alert:")
        print(f"   - Hospital: {alert['hospital_id']}")
        print(f"   - Severity: {alert['severity']:.2f}")
        print(f"   - Elevated symptoms: {list(alert['details'].keys())}")

    # Predict outbreak risk
    print("\n4. Predicting outbreak risk...")
    outbreak_prediction = engine.predict_outbreak_risk(current_df)
    print(f"   Risk Level: {outbreak_prediction['risk_level']}")
    print(f"   Confidence: {outbreak_prediction['confidence']:.2%}")
    print(f"   Trend: {outbreak_prediction['trend']}")
    print(f"   Growth Rate: {outbreak_prediction['growth_rate_pct']:.1f}%")

    print("\n   3-Day Forecast:")
    for pred in outbreak_prediction['predictions']:
        print(f"   - Day +{pred['day']}: {pred['predicted_cases']} cases")

    print("\n=== Demo Complete ===")