from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from app.routes.prediction import predict_router
from app.services.sentimentmodel import load_sentiment_model
from app.services.ldamodel import load_lda_model
from app.services.topicloading import load_topic_mapping
from ml_build.logger import get_logger

log = get_logger('Main')

@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.environ.get("PORT", "8000")
    log.info(f"Starting app on port {port}")
    log.info("Loading models at startup")
    try:
        load_sentiment_model()
        load_lda_model()
        load_topic_mapping()
        log.info("All models loaded successfully")
    except Exception as e:
        log.error(f"Failed to load models at startup: {str(e)}")
    yield
    log.info("Shutting down")

app = FastAPI(lifespan=lifespan)
app.include_router(predict_router)