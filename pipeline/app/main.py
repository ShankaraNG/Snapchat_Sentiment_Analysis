from contextlib import asynccontextmanager
import asyncio
import os
import uvicorn
from fastapi import FastAPI
from app.routes.prediction import predict_router
from app.services.sentimentmodel import load_sentiment_model
from app.services.ldamodel import load_lda_model
from app.services.topicloading import load_topic_mapping
from ml_build.logger import get_logger

log = get_logger('Main')

async def load_models_background():
    try:
        log.info("Loading models in background")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_sentiment_model)
        await loop.run_in_executor(None, load_lda_model)
        await loop.run_in_executor(None, load_topic_mapping)
        log.info("All models loaded successfully")
    except Exception as e:
        log.error(f"Failed to load models: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting app")
    asyncio.create_task(load_models_background())
    yield
    log.info("Shutting down")

app = FastAPI(
    title="Snapchat Review Analysis API",
    description="Sentiment analysis and topic modelling for Snapchat reviews",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)