import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{BASE_URL}/")
print("✓ API Status:", response.json())

# 2. Generate demo data
response = requests.post(f"{BASE_URL}/api/generate-demo-data")
data = response.json()
print("✓ Demo data generated")

# 3. Train with historical data
historical = data['historical_data']
train_request = {"records": historical}
response = requests.post(f"{BASE_URL}/api/train", json=train_request)
print("✓ Model trained:", response.json()['message'])

# 4. Analyze current data
current = data['current_data']
analyze_request = {"records": current, "severity_threshold": 1.5}
response = requests.post(f"{BASE_URL}/api/analyze", json=analyze_request)
result = response.json()
print(f"✓ Analysis complete: {result['anomaly_count']} anomalies found")
print(f"✓ Outbreak risk: {result['outbreak_prediction']['risk_level']}")
print(f"✓ Alerts generated: {len(result['alerts'])}")