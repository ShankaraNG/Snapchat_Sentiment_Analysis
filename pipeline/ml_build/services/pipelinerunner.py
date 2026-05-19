import pandas as pd
import os
from ml_build.config_loader import load_config
from ml_build.services.preprocessing import translate_to_english, clean_text_embeddings, scoretosentiment
from ml_build.logger import get_logger
from ml_build.services.pipelinebuildersentiment import robertasentiment, unload_roberta
from ml_build.services.pipelinebuilderlda import ldapipelinebuilder
from ml_build.services.training import lda_model_training, sentiment_pipeline_saving
from ml_build.services.testing import lda_pipeline_testing, sentiment_pipeline_testing
import gc
import torch

log = get_logger('PipelineRunner')
BASE_DIR = os.path.dirname(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))

def pipeline_runner():
    try:
        log.info("Starting pipeline run")
        log.info("Loading data")
        conf = load_config()
        data_path = conf['dataPath']['path']
        data_file = conf['dataPath']['file']
        full_data_path = os.path.join(BASE_DIR, data_path, data_file)
        data_df = pd.read_csv(full_data_path)
        if data_df.empty:
            raise Exception("Data file is empty")
        log.info("Data loaded successfully")
        multilingual = conf['language']['multilingual']
        log.info(f"Multilingual setting: {multilingual}")
        required_column = conf['contentcolumn']['column_name']
        if required_column not in data_df.columns:
            raise Exception(f"Required column '{required_column}' not found in data")
        comparison_column = conf['comparisoncolumn']['column_name']
        if comparison_column not in data_df.columns:
            raise Exception(f"Comparison column '{comparison_column}' not found in data")
        data_df = data_df.dropna(subset=[required_column]).reset_index(drop=True)
        if multilingual:
            log.info("Translating multilingual data to English")
            data_df['cleaned_text'] = clean_text_embeddings(data_df[required_column])
            data_df = data_df.dropna(subset=['cleaned_text']).reset_index(drop=True)
            data_df['translated_text'] = data_df['cleaned_text'].apply(translate_to_english)
            log.info("Translation completed")
        else:
            log.info("No translation needed using original text")
            data_df['translated_text'] = data_df[required_column]
        log.info("Scoring sentiment based on original labels")
        data_df['score_sentiment'] = data_df[comparison_column].apply(scoretosentiment)
        log.info("Sentiment scoring completed")
        log.info("Running Roberta sentiment analysis pipeline")
        sentiment_pipe = robertasentiment()
        if sentiment_pipe is None:
            raise Exception("Failed to create Roberta sentiment pipeline")
        artifacts_path = conf['artifacts']['save_path']
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        sentiment_model_name = conf['model']['sentiment_name']
        if not sentiment_model_name:
            raise Exception("Sentiment model name not specified in config")
        model_path = conf['model']['save_path']
        if not model_path:
            raise Exception("Model path not specified in config")
        log.info("Training and saving Roberta sentiment pipeline")
        result = sentiment_pipeline_saving(sentiment_pipe, data_df, artifacts_path, sentiment_model_name, model_path)
        if result != 'successfull':
            raise Exception("Sentiment pipeline failed during training and saving")
        log.info("Roberta sentiment pipeline trained and saved successfully") 
        log.info("RoBERTa unloaded, memory freed before LDA")
        unload_roberta()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Running LDA topic modeling pipeline")
        maxdiff = conf['countvectorizerparams']['maxdf']
        mindiff = conf['countvectorizerparams']['mindf']
        if maxdiff is None or mindiff is None:
            raise Exception("CountVectorizer parameters 'maxdf' and 'mindf' must be specified in config")
        k = conf['ldamodel']['numberoftopics']
        method = str(conf['ldamodel']['learningmethod'])
        if k is None or method is None:
            raise Exception("LDA model parameters 'numberoftopics' and 'learningmethod' must be specified in config")
        log.info(f"LDA parameters - maxdf: {maxdiff}, mindf: {mindiff}, number of topics: {k}, learning method: {method}")
        log.info("Building LDA pipeline")
        lda_pipe = ldapipelinebuilder(maxdiff, mindiff, k, method)
        if lda_pipe is None:
            raise Exception("Failed to create LDA pipeline")
        log.info("LDA pipeline built successfully")
        lda_model_name = conf['model']['lda_name']
        if not lda_model_name:
            raise Exception("LDA model name not specified in config")
        data_df = data_df.dropna(subset=['translated_text']).reset_index(drop=True)
        log.info("Training LDA pipeline")
        result = lda_model_training(lda_pipe, data_df['translated_text'], artifacts_path, lda_model_name, model_path)
        if result != 'successfull':
            raise Exception("LDA pipeline failed during training and saving")  
        log.info("LDA pipeline trained and saved successfully")
        log.info("Testing LDA pipeline with custom examples")
        result = lda_pipeline_testing(model_path, lda_model_name, artifacts_path)
        if result != 'successfull':
            raise Exception("LDA pipeline failed during testing")
        log.info("LDA pipeline testing completed successfully")
        log.info("Testing Sentiment pipeline with custom examples")
        result = sentiment_pipeline_testing(model_path, sentiment_model_name, artifacts_path)
        if result != 'successfull':
            raise Exception("Sentiment pipeline failed during testing")
        log.info("Sentiment pipeline testing completed successfully")
        log.info("Pipeline run completed successfully")
        return 'successfull'
    except Exception as e:
        log.error(f"Pipeline run has failed: {str(e)}")
        return None








