import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline as skpipe
from sklearn.preprocessing import FunctionTransformer
from ml_build.services import preprocessing as preprocessing
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

# modelconfig = load_config()
# nlpmodel = str(modelconfig['robertamodel']['model'])
# roberta_pipeline = pipeline("sentiment-analysis", model=nlpmodel)

log = get_logger('Sentiment PipelineBuilder')

_roberta_pipeline = None

def get_roberta_pipeline():
    global _roberta_pipeline
    if _roberta_pipeline is None:
        modelconfig = load_config()
        nlpmodel = str(modelconfig['robertamodel']['model'])
        _roberta_pipeline = pipeline("sentiment-analysis", model=nlpmodel)
    return _roberta_pipeline

def get_sentiment(X):
    try:
        roberta_pipeline = get_roberta_pipeline()
        safe_texts = [str(t) if (t and len(str(t).strip()) > 0) else "neutral" for t in X]
        predictions = roberta_pipeline(safe_texts, batch_size=32, truncation=True)
        return [pred['label'] for pred in predictions]
    except Exception as e:
        print(f"Error in get_sentiment: {e}")
        return ["LABEL_1"] * len(X)

def map_sentiment_labels(predictions):
    try:
        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        return [label_map.get(label, "Unknown") for label in predictions]
    except Exception as e:
        print(f"Error in mapping: {e}")
        return ["Unknown"] * len(predictions) 

def robertasentiment():
    try:
        log.info("Building Roberta sentiment analysis pipeline")
        sentiment_pipe = skpipe(steps=[
            ('cleaning', FunctionTransformer(preprocessing.clean_text_embeddings)), 
            ('roberta', FunctionTransformer(get_sentiment)),
            ('mapping', FunctionTransformer(map_sentiment_labels))
            ])
        log.info("Roberta sentiment analysis pipeline built successfully")
        return sentiment_pipe
    except Exception as e:
        log.error(f"Error building Roberta sentiment analysis pipeline: {e}")
        return None

def unload_roberta():
    global _roberta_pipeline
    _roberta_pipeline = None
    log.info("RoBERTa model unloaded from memory")