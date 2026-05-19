from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.sentimentmodel import predict_sentiment
from app.services.ldamodel import predict_lda
from app.services.topicloading import load_topic_mapping
from ml_build.services.preprocessing import translate_to_english
from app.logger import get_logger

log = get_logger('PredictionRoute')
predict_router = APIRouter()

class TextItem(BaseModel):
    id: Optional[str] = None
    text: str

class TextRequest(BaseModel):
    texts: List[TextItem]

@predict_router.get('/')
def root():
    return {'message': 'Welcome to Snapchat Review Analysis API'}

@predict_router.get('/health')
def health():
    return {'status': 'healthy'}

@predict_router.post('/predict')
def predict_route(request: TextRequest):
    try:
        log.info("Received prediction request")
        topic_mapping = load_topic_mapping()
        if topic_mapping is None:
            raise HTTPException(status_code=500, detail='Failed to load topic mapping')
        if not request.texts:
            raise HTTPException(status_code=400, detail='No texts provided')
        log.info(f"Predicting sentiment and topic for {len(request.texts)} texts")
        raw_texts = [item.text for item in request.texts]
        raw_texts = [translate_to_english(text) for text in raw_texts]
        log.info("Predicting sentiment")
        sentiment_results = predict_sentiment(raw_texts)
        log.info("Predicting topics")
        lda_results = predict_lda(raw_texts)
        response = []
        log.info("Constructing response")
        for i, item in enumerate(request.texts):
            sentiment = sentiment_results[i]
            topic = (lda_results[i].argmax() + 1).item()
            mapping_key = f"TOPIC {topic}"
            topiclabel = topic_mapping.get(mapping_key, 'Unknown')
            response.append({
                'id': item.id,
                'text': item.text,
                'sentiment': sentiment,
                'topic_number': topic,
                'topic_label': topiclabel
            })
        log.info("Prediction completed successfully")
        return {'predictions': response}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Pipeline for prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    