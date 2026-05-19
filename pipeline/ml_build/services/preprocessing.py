import regex as re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from deep_translator import GoogleTranslator
import time
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

log = get_logger('Preprocessing')

def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^\x00-\x7F]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\{.*?\}|\[.*?\]|<.*?>|\(.*?\)', '', text)
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        log.error(f"Pipeline run has failed in cleaning: {str(e)}")
        return ""

def clean_text_embeddings(text_list):
    try:
        results = []
        for text in text_list:
            text = str(text).lower()
            text = re.sub(r'[^\x00-\x7F]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
            text = re.sub(r'\{.*?\}|\[.*?\]|<.*?>|\(.*?\)', '', text)
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            text = re.sub(r'[^a-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            results.append(text)
        return results
    except Exception as e:
        log.error(f"Pipeline run has failed in cleaning: {str(e)}")
        return [""] * len(text_list)

def lda_preprocess(text,lemmatizer,stop_words,lda_custom_stopwords):
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and word not in lda_custom_stopwords]
        tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]
        return " ".join(tokens)
    except Exception as e:
        log.error(f"Pipeline run has failed in LDA preprocess step: {str(e)}")
        return None

def lda_preprocessing(text_list):
    try:
        conf = load_config()
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        lda_custom_stopwords = set(conf.get('ldacustomstopwords', []))
        if not lda_custom_stopwords:
            log.warning("No custom stopwords specified for LDA in config, proceeding with only NLTK stopwords")
        results = []
        for text in text_list:
            cleaned = clean_text(text) 
            processed = lda_preprocess(cleaned,lemmatizer,stop_words,lda_custom_stopwords)
            results.append(processed)
        return results
    except Exception as e:
        log.error(f"Pipeline run has failed in LDA preprocess step: {str(e)}")
        return None
    
def scoretosentiment(score):
    try:
        if score >= 4:
            return 'Positive'
        elif score <= 2:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        log.error(f"Pipeline run has failed in scoring sentiment: {str(e)}")
        return 'Neutral'

def translate_to_english(text):
    try:
        log.info(f"Translating text: {text}")
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        time.sleep(0.2)
        return translated
    except Exception as e:
        log.error(f"Translation failed returning the original text: {str(e)}")
        return text