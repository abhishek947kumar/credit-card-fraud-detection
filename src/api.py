"""
FastAPI Backend Application for Real-Time Credit Card Fraud Detection.
Provides RESTful APIs for single & batch scoring, model comparison,
live streaming simulation, explainable diagnostics, and WebSocket feeds.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    CATEGORY_LIST,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MODELS_DIR,
    STATIC_DIR,
)
from src.predictor import FraudPredictor
from src.simulator import TransactionStreamSimulator
from src.train import train_all_models

app = FastAPI(
    title="Financial Fraud Detection Engine API",
    description="Enterprise Machine Learning Platform for Real-Time Credit Card Fraud Prevention",
    version="1.0.0",
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Singletons
predictor = FraudPredictor(active_model_name="xgboost")
simulator = TransactionStreamSimulator(buffer_size=200)

# Pre-populate simulator with a batch of initial realistic transactions
for _ in range(25):
    raw = simulator.generate_random_transaction()
    scored = predictor.predict_single(raw)
    simulator.record_transaction(scored)

# Connected WebSocket clients
active_websockets: List[WebSocket] = []


# --- Pydantic Request Models ---
class TransactionInput(BaseModel):
    transaction_id: Optional[str] = Field(None, examples=["TXN-A83B91"])
    user_id: Optional[str] = Field(None, examples=["USR-1042"])
    timestamp: Optional[str] = Field(None, examples=["2026-09-30 16:40:00"])
    amount: float = Field(..., examples=[145.50], description="Transaction amount in USD")
    category: str = Field(..., examples=["online_retail"], description="Merchant category")
    distance_from_home: float = Field(5.2, examples=[12.4], description="Distance from cardholder home in km")
    distance_from_last_txn: float = Field(2.1, examples=[8.5], description="Distance from previous transaction in km")
    time_since_last_txn: float = Field(120.0, examples=[45.0], description="Minutes since last transaction")
    velocity_1h: int = Field(0, examples=[2], description="Transactions in last 1 hour")
    velocity_24h: int = Field(2, examples=[4], description="Transactions in last 24 hours")
    amount_to_user_mean_ratio: float = Field(1.0, examples=[1.35], description="Ratio to 30-day average spend")
    device_trust_score: float = Field(85.0, examples=[92.0], description="Device trust confidence 0-100")
    hour_of_day: int = Field(14, examples=[15], description="Hour of transaction 0-23")
    day_of_week: int = Field(2, examples=[3], description="Day of week 0-6")
    is_weekend: int = Field(0, examples=[0], description="Weekend flag")
    is_online: int = Field(1, examples=[1], description="Card not present / Online flag")
    foreign_transaction: int = Field(0, examples=[0], description="Foreign terminal flag")
    failed_pin_attempts_24h: int = Field(0, examples=[0], description="Failed PIN attempts")
    v1: Optional[float] = 0.0
    v2: Optional[float] = 0.0
    v3: Optional[float] = 0.0
    v4: Optional[float] = 0.0
    v5: Optional[float] = 0.0
    v6: Optional[float] = 0.0
    v7: Optional[float] = 0.0
    v8: Optional[float] = 0.0


class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., examples=["xgboost"])


class SimulatorGenerateRequest(BaseModel):
    count: int = Field(1, ge=1, le=50)
    force_fraud: Optional[bool] = None


class AnalystActionRequest(BaseModel):
    action: str = Field(..., examples=["CONFIRM_FRAUD"])  # APPROVE, CONFIRM_FRAUD, DISMISS
    analyst_notes: Optional[str] = None


# --- Web Page Route ---
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index.html not found.")
    return FileResponse(index_file)


# --- REST API Endpoints ---
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_model": predictor.active_model_name,
        "available_models": list(predictor.MODEL_FILES.keys()),
        "categories": CATEGORY_LIST,
    }


@app.post("/api/predict")
async def predict_transaction(txn: TransactionInput):
    """
    Score a single credit card transaction in real time.
    Returns fraud probability, risk score, tier, action, and explainable risk factors.
    """
    try:
        result = predictor.predict_single(txn.model_dump())
        simulator.record_transaction(result)
        
        # Broadcast to any active WebSocket listeners
        await broadcast_transaction(result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/batch-predict")
async def predict_batch(transactions: List[TransactionInput]):
    """
    Score an array of transactions in a batch.
    """
    try:
        raw_list = [t.model_dump() for t in transactions]
        results = predictor.predict_batch(raw_list)
        return {
            "total_scored": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_tier"] == "HIGH"),
            "medium_risk_count": sum(1 for r in results if r["risk_tier"] == "MEDIUM"),
            "low_risk_count": sum(1 for r in results if r["risk_tier"] == "LOW"),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/api/upload-csv")
async def upload_csv_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file containing transactions and score every record.
    """
    try:
        df = pd.read_csv(file.file)
        records = df.to_dict(orient="records")
        results = predictor.predict_batch(records)
        
        return {
            "filename": file.filename,
            "total_rows": len(results),
            "high_risk": sum(1 for r in results if r["risk_tier"] == "HIGH"),
            "medium_risk": sum(1 for r in results if r["risk_tier"] == "MEDIUM"),
            "low_risk": sum(1 for r in results if r["risk_tier"] == "LOW"),
            "sample_results": results[:20],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.get("/api/models")
async def get_models_metadata():
    """
    Retrieve performance comparison benchmark metrics, PR/ROC curves,
    and business ROI across all trained ML models.
    """
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found. Please train models first.")
    
    with open(meta_path, "r") as f:
        data = json.load(f)
        
    data["current_active_model"] = predictor.active_model_name
    return data


@app.post("/api/models/switch")
async def switch_model(req: SwitchModelRequest):
    """
    Dynamically hot-swap the active ML model serving predictions.
    """
    try:
        predictor.set_active_model(req.model_name)
        return {
            "status": "success",
            "active_model": predictor.active_model_name,
            "message": f"Successfully activated {req.model_name.upper()} model.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/transactions/recent")
async def get_recent_transactions(limit: int = 50):
    """
    Fetch the most recent transactions from the live streaming buffer.
    """
    txns = simulator.get_recent_transactions(limit=limit)
    return {
        "count": len(txns),
        "total_processed": simulator.total_processed,
        "total_fraud_detected": simulator.total_fraud_detected,
        "transactions": txns,
    }


@app.post("/api/simulator/generate")
async def generate_simulated_transactions(req: SimulatorGenerateRequest):
    """
    Generate synthetic transactions on demand and push through the ML scoring engine.
    """
    generated = []
    for _ in range(req.count):
        raw = simulator.generate_random_transaction(force_fraud=req.force_fraud)
        scored = predictor.predict_single(raw)
        simulator.record_transaction(scored)
        generated.append(scored)
        await broadcast_transaction(scored)
    
    return {
        "generated_count": len(generated),
        "transactions": generated,
    }


@app.post("/api/transactions/{txn_id}/action")
async def submit_analyst_action(txn_id: str, req: AnalystActionRequest):
    """
    Fraud Analyst manual override (Approve, Confirm Fraud, Dismiss).
    """
    # Find transaction in simulator buffer if present
    found = False
    for t in simulator.history:
        if t.get("transaction_id") == txn_id:
            t["analyst_review"] = {
                "decision": req.action,
                "notes": req.analyst_notes or "Reviewed by Fraud Operations Center",
                "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            found = True
            break
            
    return {
        "status": "success",
        "transaction_id": txn_id,
        "action": req.action,
        "message": f"Transaction {txn_id} updated with action '{req.action}'.",
    }


@app.get("/api/analytics/kpis")
async def get_kpis():
    """
    Aggregate executive KPI statistics for the live monitoring dashboard.
    """
    recent = list(simulator.history)
    total_txns = len(recent)
    high_count = sum(1 for t in recent if t.get("risk_tier") == "HIGH")
    med_count = sum(1 for t in recent if t.get("risk_tier") == "MEDIUM")
    low_count = sum(1 for t in recent if t.get("risk_tier") == "LOW")
    
    # Financial prevented
    fraud_amount_prevented = sum(
        t.get("amount", 0.0) for t in recent if t.get("risk_tier") == "HIGH"
    )
    
    avg_latency = (
        round(sum(t.get("latency_ms", 0.0) for t in recent) / total_txns, 2)
        if total_txns > 0
        else 0.0
    )

    return {
        "total_processed": simulator.total_processed,
        "buffered_count": total_txns,
        "fraud_blocked_count": high_count,
        "under_review_count": med_count,
        "auto_approved_count": low_count,
        "fraud_rate_pct": round((high_count / max(total_txns, 1)) * 100, 2),
        "fraud_amount_prevented_usd": round(fraud_amount_prevented, 2),
        "avg_latency_ms": avg_latency,
        "active_model": predictor.active_model_name,
    }


@app.post("/api/retrain")
async def retrain_models(use_smote: bool = False):
    """
    Trigger retraining pipeline of all ML models.
    """
    try:
        metadata = train_all_models(use_smote=use_smote, n_train=15000, n_test=4000)
        # Reload predictor with newly trained models
        predictor.load_model(predictor.active_model_name)
        return {
            "status": "success",
            "message": "Models successfully retrained and reloaded into memory!",
            "summary": metadata.get("models", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# --- WebSocket Handler ---
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live streaming of scored transactions.
    """
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # Keep connection alive; client can send ping or simulator commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_transaction(txn: Dict[str, Any]):
    """Send transaction update to all connected WebSocket clients."""
    if not active_websockets:
        return
    disconnected = []
    payload = json.dumps(txn)
    for ws in active_websockets:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)
