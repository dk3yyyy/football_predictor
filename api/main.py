import logging
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from db.database import Database
from models.predict import get_predictions

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Football Predictor API",
    description="Programmatic access to ML match outcomes and expected goal predictions.",
    version="1.0.0"
)

# Allow CORS for potential frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    return Database()

@app.get("/health")
def health_check():
    """Simple API health check."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/predictions/upcoming")
def upcoming_predictions(
    league_key: Optional[str] = None, 
    db: Database = Depends(get_db)
):
    """
    Returns the latest predictions for upcoming matches from the predictions_log table.
    """
    query = """
        SELECT * FROM predictions_log 
        WHERE actual_winner IS NULL 
        AND match_date >= :today
    """
    params = {"today": datetime.utcnow().isoformat()}
    
    if league_key:
        query += " AND league_key = :league"
        params["league"] = league_key
        
    query += " ORDER BY match_date ASC LIMIT 50"
    
    with db.engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
        
    predictions = [dict(r._mapping) for r in rows]
    return {"predictions": predictions}

@app.post("/predictions/generate")
def trigger_predictions():
    """
    Manually triggers the ML inference pipeline for new upcoming matches not yet predicted.
    """
    try:
        preds = get_predictions()
        return {"status": "success", "generated_count": len(preds)}
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics(model_name: str = "xgboost-v1", db: Database = Depends(get_db)):
    """
    Returns rolling accuracy stats for the given model.
    """
    accuracy = db.get_prediction_accuracy(model_name)
    if not accuracy or not accuracy.get("total"):
        return {"model_name": model_name, "message": "No evaluated predictions yet."}
        
    return {
        "model_name": model_name,
        "total_evaluations": accuracy["total"],
        "correct_predictions": accuracy["correct"],
        "accuracy_pct": accuracy["accuracy_pct"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
