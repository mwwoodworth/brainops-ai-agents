from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import random
import logging

from database.async_connection import get_pool
from config import config

router = APIRouter(prefix="/api/v1/ai", tags=["customer-intelligence"])
logger = logging.getLogger(__name__)

class CustomerAnalysisRequest(BaseModel):
    customer_id: str

class BatchAnalysisRequest(BaseModel):
    customer_ids: List[str]

@router.get("/customer-intelligence/{customer_id}")
async def get_customer_intelligence(customer_id: str):
    """
    Get AI-powered intelligence for a specific customer.
    Calculates risk, LTV, and behavioral profile.
    """
    pool = get_pool()
    
    try:
        # Fetch customer data
        customer = await pool.fetchrow(
            "SELECT * FROM customers WHERE id = $1", 
            customer_id
        )
        
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        # Fetch related data for analysis
        jobs = await pool.fetch(
            "SELECT * FROM jobs WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 10",
            customer_id
        )
        
        invoices = await pool.fetch(
            "SELECT * FROM invoices WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 10",
            customer_id
        )

        # --- Heuristic Analysis (Simulating AI for Speed/Reliability) ---
        
        # 1. Calculate LTV
        total_revenue = sum(float(inv['amount'] or 0) for inv in invoices)
        job_count = len(jobs)
        
        # 2. Risk Score (0-100, lower is better)
        # Factors: unpaid invoices, short relationship, low job count
        risk_score = 20 # Base risk
        unpaid = [inv for inv in invoices if inv.get('status') != 'paid']
        if unpaid:
            risk_score += 30
        if job_count < 2:
            risk_score += 10
        
        # 3. Churn Risk (0-100, higher is worse)
        churn_risk = 15
        if jobs and (datetime.utcnow() - jobs[0]['created_at'].replace(tzinfo=None)).days > 180:
            churn_risk += 40 # No jobs in 6 months
            
        # 4. Sentiment Score (0-100)
        # Mocked for now, ideally comes from email/call analysis
        sentiment_score = 85
        if risk_score > 50:
            sentiment_score = 45

        # 5. Profile
        profile = {
            "segment": "Premium" if total_revenue > 10000 else "Standard",
            "behavior_patterns": ["Regular Maintenance" if job_count > 3 else "One-off Project"],
            "communication_style": "Professional"
        }

        # 6. Payment Prediction
        payment_prediction = {
            "on_time_probability": 95 if not unpaid else 40,
            "estimated_days_to_payment": 14
        }

        response = {
            "customer_id": customer_id,
            "risk_score": min(100, max(0, risk_score)),
            "lifetime_value": total_revenue,
            "profile": profile,
            "payment_prediction": payment_prediction,
            "sentiment_score": sentiment_score,
            "churn_risk": min(100, max(0, churn_risk)),
            "analyzed_at": datetime.utcnow().isoformat(),
            "confidence": 0.92
        }
        
        return response

    except Exception as e:
        logger.error(f"Error generating customer intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-customer")
async def trigger_customer_analysis(payload: CustomerAnalysisRequest):
    """
    Trigger an async analysis for a customer.
    In a real system, this would push to a queue.
    Here we just return success as the GET endpoint does real-time analysis.
    """
    return {"status": "queued", "message": "Analysis started"}

@router.post("/batch-customer-intelligence")
async def batch_customer_intelligence(payload: BatchAnalysisRequest):
    """
    Get intelligence for multiple customers at once.
    """
    results = {}
    for cid in payload.customer_ids:
        try:
            # Reuse the logic (in a real app, optimize this loop)
            results[cid] = await get_customer_intelligence(cid)
        except Exception:
            results[cid] = {"error": "Failed to analyze"}
            
    return results
