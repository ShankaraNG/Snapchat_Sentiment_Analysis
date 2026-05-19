import joblib
from ml_build.config_loader import load_config
from ml_build.services.preprocessing import lda_preprocessing, clean_text, clean_text_embeddings, lda_preprocess
from ml_build.services.pipelinebuildersentiment import map_sentiment_labels, get_sentiment
from ml_build.logger import get_logger
import os

log = get_logger('Testing')

BASE_DIR = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))

def lda_pipeline_testing(model_path, model_name, artifacts_path):
    try:
        log.info("Loading LDA model for testing")
        full_model_path = os.path.join(BASE_DIR, model_path, f"{model_name}.joblib")
        if not os.path.exists(full_model_path):
            raise Exception(f"LDA model file not found at path: {full_model_path}")
        loaded_pipeline = joblib.load(full_model_path)
        custom_tests = [
                    "This app keeps crashing and freezing after updating",
                    "Love the new face streaks and filter animations"
                ]
        log.info("Testing LDA pipeline with custom examples")
        custom_results = loaded_pipeline.transform(custom_tests)
        log.info("LDA pipeline testing completed, saving results")
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "lda_testing_results.txt")
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))
        with open(output_file_path, "w") as f:
            for i, text in enumerate(custom_tests):
                dom_top = custom_results[i].argmax() + 1
                f.write(f"Custom: \"{text}\" -> Topic #{dom_top}\n")
        log.info("LDA pipeline testing results saved successfully")
        return 'successfull'
    except Exception as e:
        log.error(f"LDA pipeline testing failed: {str(e)}")
        return None

def sentiment_pipeline_testing(model_path, model_name, artifacts_path):
    try:
        log.info("Loading Sentiment model for testing")
        full_model_path = os.path.join(BASE_DIR, model_path, f"{model_name}.joblib")
        if not os.path.exists(full_model_path):
            raise Exception(f"Sentiment model file not found at path: {full_model_path}")
        loaded_pipeline = joblib.load(full_model_path)
        custom_tests = [
                    "This app keeps crashing and freezing after updating",
                    "Love the new face streaks and filter animations"
                ]
        custom_results = loaded_pipeline.transform(custom_tests)
        output_file_path = os.path.join(BASE_DIR, artifacts_path, "sentiment_testing_results.txt")
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))
        with open(output_file_path, "w") as f:
            for i, text in enumerate(custom_tests):
                f.write(f"Custom: \"{text}\" -> Sentiment: {custom_results[i]}\n")
        log.info("Sentiment pipeline testing completed, results saved")
        return 'successfull'
    except Exception as e:
        log.error(f"Sentiment pipeline testing failed: {str(e)}")
        return None