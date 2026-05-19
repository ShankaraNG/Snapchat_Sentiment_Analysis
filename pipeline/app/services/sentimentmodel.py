import joblib
import os
from app.logger import get_logger
from app.config_loader import load_config
from ml_build.services.preprocessing import lda_preprocessing, clean_text, clean_text_embeddings, lda_preprocess
from ml_build.services.pipelinebuildersentiment import map_sentiment_labels, get_sentiment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sentiment_pipeline = None
log = get_logger('SentimentService')

def load_sentiment_model():
    try:
        log.info("Loading sentiment model")
        global sentiment_pipeline
        if sentiment_pipeline is None:
            conf = load_config()
            path = conf['model']['save_path']
            model_name = conf['model']['sentiment_name']
            full_model_path = os.path.join(BASE_DIR, path, f"{model_name}.joblib")
            if not os.path.exists(full_model_path):
                raise Exception(f"Sentiment model file not found at path: {full_model_path}")
            sentiment_pipeline = joblib.load(full_model_path)
        log.info("Sentiment model loaded successfully")
        return sentiment_pipeline
    except Exception as e:
        log.error(f"Failed to load sentiment model: {str(e)}")
        raise

def predict_sentiment(text):
    try:
        log.info("Predicting sentiment")
        pipeline = load_sentiment_model()
        log.info("Returning the sentiment prediction")
        return pipeline.transform(text)
    except Exception as e:
        log.error(f"Sentiment prediction failed: {str(e)}")
        raise
