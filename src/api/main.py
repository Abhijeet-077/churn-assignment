"""
FastAPI application for customer churn prediction.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
from typing import List, Dict, Any
import logging

from .models import (
    CustomerData, BatchCustomerData, PredictionRequest,
    PredictionResponse, BatchPredictionResponse, DetailedPredictionResponse,
    ModelInfo, HealthResponse, FeatureImportanceResponse, ErrorResponse
)
from .prediction_service import PredictionService
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Global prediction service instance
prediction_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global prediction_service
    
    # Startup
    logger.info("Starting Customer Churn Prediction API...")
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Churn Prediction API...")


# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Advanced ML-powered API for predicting customer churn with interpretability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    return prediction_service


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(service: PredictionService = Depends(get_prediction_service)):
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            models_loaded=service.get_available_models(),
            uptime_seconds=service.get_uptime()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/predict", response_model=DetailedPredictionResponse)
async def predict_single(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """Predict churn for a single customer."""
    try:
        logger.info(f"Single prediction request for model: {request.model_name}")
        
        prediction = service.predict_single(
            customer_data=request.customer_data,
            model_name=request.model_name,
            include_explanation=request.include_explanation
        )
        
        logger.info(f"Prediction completed: {prediction.churn_prediction} "
                   f"(probability: {prediction.churn_probability:.3f})")
        
        return prediction
        
    except ValueError as e:
        logger.error(f"Prediction validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchCustomerData,
    model_name: str = "best",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: PredictionService = Depends(get_prediction_service)
):
    """Predict churn for multiple customers."""
    try:
        logger.info(f"Batch prediction request for {len(request.customers)} customers")
        
        # For large batches, consider running in background
        if len(request.customers) > 100:
            logger.info("Large batch detected, processing in background")
        
        result = service.predict_batch(
            customers=request.customers,
            model_name=model_name
        )
        
        logger.info(f"Batch prediction completed: {result['summary']['successful_predictions']} successful")
        
        return BatchPredictionResponse(
            predictions=result['predictions'],
            summary=result['summary']
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/models", response_model=List[str])
async def get_available_models(service: PredictionService = Depends(get_prediction_service)):
    """Get list of available models."""
    try:
        return service.get_available_models()
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@app.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """Get information about a specific model."""
    try:
        available_models = service.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        return service.get_model_info(model_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/models/{model_name}/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_name: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """Get feature importance for a specific model."""
    try:
        available_models = service.get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        importance_data = service.get_feature_importance(model_name)
        
        return FeatureImportanceResponse(
            model_name=importance_data['model_name'],
            feature_importance=importance_data['feature_importance'],
            top_features=importance_data['top_features']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feature importance")


@app.get("/metrics", response_model=Dict[str, Any])
async def get_api_metrics(service: PredictionService = Depends(get_prediction_service)):
    """Get API metrics and statistics."""
    try:
        return {
            "uptime_seconds": service.get_uptime(),
            "models_loaded": len(service.get_available_models()),
            "available_models": service.get_available_models(),
            "timestamp": datetime.now().isoformat(),
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Additional utility endpoints

@app.post("/validate", response_model=Dict[str, str])
async def validate_customer_data(customer_data: CustomerData):
    """Validate customer data without making prediction."""
    try:
        # If we reach here, Pydantic validation passed
        return {
            "status": "valid",
            "message": "Customer data validation passed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@app.get("/schema", response_model=Dict[str, Any])
async def get_data_schema():
    """Get the expected data schema for customer data."""
    try:
        return {
            "schema": CustomerData.schema(),
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "One year",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 65.0,
                "total_charges": 780.0
            }
        }
    except Exception as e:
        logger.error(f"Failed to get schema: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve schema")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
