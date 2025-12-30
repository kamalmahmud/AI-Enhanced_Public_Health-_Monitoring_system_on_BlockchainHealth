"""
REST API for AI Analytics Engine
Provides endpoints for the React dashboard to consume
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import json

# Import the analytics engine from separate file
from ai_analytics_engine import HealthAnalyticsEngine

# Initialize FastAPI app
app = FastAPI(
    title="Health Analytics API",
    description="AI-powered health monitoring analytics for blockchain-based public health system",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analytics engine
analytics_engine = HealthAnalyticsEngine(contamination=0.15)

# In-memory storage for demo (replace with actual blockchain data fetching)
historical_data_store = []
current_data_store = []


# Pydantic models for request/response validation
class HealthRecord(BaseModel):
    hospital_id: str
    timestamp: str
    fever: Optional[int] = 0
    cough: Optional[int] = 0
    fatigue: Optional[int] = 0
    headache: Optional[int] = 0
    total_cases: int

    class Config:
        json_schema_extra = {
            "example": {
                "hospital_id": "A",
                "timestamp": "2025-02-01",
                "fever": 35,
                "cough": 25,
                "fatigue": 20,
                "headache": 10,
                "total_cases": 90
            }
        }


class TrainingRequest(BaseModel):
    records: List[HealthRecord]


class AnalysisRequest(BaseModel):
    records: List[HealthRecord]
    severity_threshold: Optional[float] = 1.5


class Alert(BaseModel):
    alert_id: str
    timestamp: str
    hospital_id: str
    severity: float
    anomaly_score: float
    level: str
    details: Dict


class OutbreakPrediction(BaseModel):
    risk_level: str
    confidence: float
    trend: str
    growth_rate_pct: float
    current_avg_daily_cases: float
    predictions: List[Dict]


class AnalysisResponse(BaseModel):
    anomaly_count: int
    total_records: int
    alerts: List[Alert]
    outbreak_prediction: OutbreakPrediction
    timestamp: str


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Health Analytics API",
        "version": "1.0.0",
        "engine_trained": analytics_engine.is_trained
    }


@app.post("/api/train")
async def train_baseline(request: TrainingRequest):
    """
    Train the analytics engine on historical baseline data

    This should be called once with normal, non-outbreak historical data
    """
    try:
        records = [record.dict() for record in request.records]
        df = analytics_engine.prepare_data(records)
        analytics_engine.train_baseline(df)

        # Store for reference
        global historical_data_store
        historical_data_store = records

        return {
            "status": "success",
            "message": f"Baseline trained on {len(records)} records",
            "features_monitored": list(analytics_engine.baseline_mean.index),
            "baseline_stats": {
                "mean": analytics_engine.baseline_mean.to_dict(),
                "std": analytics_engine.baseline_std.to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_health_data(request: AnalysisRequest):
    """
    Analyze current health data for anomalies and outbreak risk

    Returns alerts and outbreak predictions
    """
    if not analytics_engine.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Engine not trained. Call /api/train first with baseline data"
        )

    try:
        records = [record.dict() for record in request.records]

        # Prepare and analyze data
        df = analytics_engine.prepare_data(records)
        analyzed = analytics_engine.detect_anomalies(df)
        analyzed = analytics_engine.calculate_severity(analyzed)

        # Generate alerts
        alerts = analytics_engine.generate_alerts(analyzed, request.severity_threshold)

        # Predict outbreak risk
        outbreak_pred = analytics_engine.predict_outbreak_risk(df)

        # Store for reference
        global current_data_store
        current_data_store = records

        return AnalysisResponse(
            anomaly_count=int(analyzed['is_anomaly'].sum()),
            total_records=len(analyzed),
            alerts=alerts,
            outbreak_prediction=OutbreakPrediction(**outbreak_pred),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/alerts")
async def get_recent_alerts(limit: int = 10):
    """
    Get most recent alerts
    """
    if not analytics_engine.is_trained or not current_data_store:
        return {"alerts": [], "message": "No data analyzed yet"}

    try:
        df = analytics_engine.prepare_data(current_data_store)
        analyzed = analytics_engine.detect_anomalies(df)
        analyzed = analytics_engine.calculate_severity(analyzed)
        alerts = analytics_engine.generate_alerts(analyzed)

        # Sort by severity and return top N
        sorted_alerts = sorted(alerts, key=lambda x: x['severity'], reverse=True)
        return {"alerts": sorted_alerts[:limit], "total_alerts": len(alerts)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@app.get("/api/outbreak-risk")
async def get_outbreak_risk():
    """
    Get current outbreak risk prediction
    """
    if not analytics_engine.is_trained or not current_data_store:
        return {
            "risk_level": "UNKNOWN",
            "message": "No data analyzed yet"
        }

    try:
        df = analytics_engine.prepare_data(current_data_store)
        prediction = analytics_engine.predict_outbreak_risk(df)
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict risk: {str(e)}")


@app.get("/api/statistics")
async def get_statistics():
    """
    Get overall system statistics
    """
    return {
        "engine_status": "trained" if analytics_engine.is_trained else "not_trained",
        "historical_records": len(historical_data_store),
        "current_records": len(current_data_store),
        "baseline_features": list(analytics_engine.baseline_mean.index) if analytics_engine.is_trained else []
    }


@app.post("/api/generate-demo-data")
async def generate_demo_data():
    """
    Generate synthetic data for testing (demo purposes)
    """
    np.random.seed(42)

    # Generate historical baseline data
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    hospitals = ['A', 'B', 'C']

    historical = []
    for date in dates:
        for hospital in hospitals:
            record = {
                'hospital_id': hospital,
                'timestamp': date.strftime('%Y-%m-%d'),
                'fever': int(np.random.randint(20, 40)),
                'cough': int(np.random.randint(15, 30)),
                'fatigue': int(np.random.randint(10, 25)),
                'headache': int(np.random.randint(5, 15)),
                'total_cases': int(np.random.randint(50, 100))
            }
            historical.append(record)

    # Generate current data with outbreak
    current_dates = pd.date_range(start='2025-02-01', end='2025-02-07', freq='D')
    current = []
    for i, date in enumerate(current_dates):
        for hospital in hospitals:
            outbreak_multiplier = 1.0 if i < 3 else (1.0 + (i - 2) * 0.3)
            record = {
                'hospital_id': hospital,
                'timestamp': date.strftime('%Y-%m-%d'),
                'fever': int(np.random.randint(20, 40) * outbreak_multiplier),
                'cough': int(np.random.randint(15, 30) * outbreak_multiplier),
                'fatigue': int(np.random.randint(10, 25) * outbreak_multiplier),
                'headache': int(np.random.randint(5, 15) * outbreak_multiplier),
                'total_cases': int(np.random.randint(50, 100) * outbreak_multiplier)
            }
            current.append(record)

    return {
        "historical_data": historical,
        "current_data": current,
        "message": "Demo data generated. Use /api/train with historical_data, then /api/analyze with current_data"
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Health Analytics API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)