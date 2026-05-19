import joblib
import os
from app.logger import get_logger
from ml_build.config_loader import load_config
from ml_build.services.preprocessing import lda_preprocessing, clean_text, clean_text_embeddings, lda_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

lda_pipeline = None

log = get_logger('LDAService')

def load_lda_model():
    try:
        log.info("Loading LDA model")
        global lda_pipeline
        if lda_pipeline is None:
            conf = load_config()
            path = conf['model']['save_path']
            model_name = conf['model']['lda_name']
            full_model_path = os.path.join(BASE_DIR, path, f"{model_name}.joblib")
            if not os.path.exists(full_model_path):
                raise Exception(f"LDA model file not found at path: {full_model_path}")
            lda_pipeline = joblib.load(full_model_path)
        return lda_pipeline
    except Exception as e:
        log.error(f"Failed to load LDA model: {str(e)}")
        raise

def predict_lda(text):
    try:
        log.info("Predicting LDA topics")
        pipeline = load_lda_model()
        log.info("Returning the LDA topic prediction numbers")
        return pipeline.transform(text)
    except Exception as e:
        log.error(f"LDA prediction failed: {str(e)}")
        return None