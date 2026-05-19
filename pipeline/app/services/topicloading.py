import json
import os
from app.config_loader import load_config
from app.logger import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = get_logger('TopicLoadingService')

def load_topic_mapping():
    try:
        log.info("Loading topic mapping from JSON")
        conf = load_config()
        path = conf['topicmapping']['filepath']
        filename = conf['topicmapping']['filename']
        JSON_PATH = os.path.join(BASE_DIR, path, filename)
        if not os.path.exists(JSON_PATH):
            raise Exception(f"JSON file not found at path: {JSON_PATH}")
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            topic_mapping = json.load(f)
        log.info("Successfully loaded topic mappings!")
        return topic_mapping
    except Exception as e:
        log.error(f"Error loading JSON: {str(e)}")
        return {}